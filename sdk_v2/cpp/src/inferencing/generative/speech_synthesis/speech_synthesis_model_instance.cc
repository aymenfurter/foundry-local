// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "inferencing/generative/speech_synthesis/speech_synthesis_model_instance.h"

#include "exception.h"
#include "inferencing/generative/speech_synthesis/dpm_solver.h"
#include "utils.h"

#include <fmt/format.h>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_session_options_config_keys.h>
#include <ort_genai.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <initializer_list>
#include <optional>
#include <string_view>
#include <unordered_map>

namespace fl {

namespace {

constexpr auto kInt64 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
constexpr auto kFloat16 = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;

// A voice preset is a few MB of cached prompt. The limits stop a package from exhausting memory.
constexpr uint64_t kMaxVoiceFileBytes = 64ULL * 1024 * 1024;
constexpr uint64_t kMaxVoiceTotalBytes = 512ULL * 1024 * 1024;

// The model makes 7.5 latent frames per second of speech, and English speech needs about 2 frames per text token. The
// frame budget stops a run whose EOS classifier never fires. The token limit keeps prompt, text and speech inside the
// 8192 positions of the TTS LM.
constexpr size_t kMinFrameBudget = 32;
constexpr size_t kFramesPerToken = 4;
constexpr size_t kMaxTextTokens = 1024;

constexpr uint32_t kNoiseSeed = 0x5eed;

/// A graph tensor: name and element type.
struct TensorSpec {
  const char* name;
  ONNXTensorElementDataType type;
};

/// One ONNX graph and the names of its inputs and outputs, in the order of the values passed to Run. A language model
/// graph also takes past_key_<i>/past_value_<i> after the head inputs and returns present_key_<i>/present_value_<i>
/// after the head outputs.
struct Graph {
  std::unique_ptr<Ort::Session> session;
  std::string label;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<const char*> inputs;
  std::vector<const char*> outputs;
  size_t layers = 0;
  size_t head_outputs = 0;

  std::vector<Ort::Value> Run(std::vector<Ort::Value>& values) const {
    return session->Run(Ort::RunOptions{nullptr}, inputs.data(), values.data(), values.size(), outputs.data(),
                        outputs.size());
  }
};

const char* TypeName(ONNXTensorElementDataType type) {
  return type == kInt64 ? "int64" : type == kFloat16 ? "float16" : "another type";
}

/// The shape of the graph tensor `name`, or nullopt when the graph has no such tensor. Throws when the tensor has
/// another element type.
std::optional<std::vector<int64_t>> FindTensor(const Graph& graph, bool input, std::string_view name,
                                               ONNXTensorElementDataType type) {
  Ort::AllocatorWithDefaultOptions allocator;
  const auto& session = *graph.session;
  const auto count = input ? session.GetInputCount() : session.GetOutputCount();
  for (size_t i = 0; i < count; ++i) {
    const auto actual =
        input ? session.GetInputNameAllocated(i, allocator) : session.GetOutputNameAllocated(i, allocator);
    if (std::string_view(actual.get()) != name) {
      continue;
    }

    // The tensor info is a view into the type info, so the type info must stay alive while it is read.
    const auto type_info = input ? session.GetInputTypeInfo(i) : session.GetOutputTypeInfo(i);
    const auto tensor = type_info.GetTensorTypeAndShapeInfo();
    if (tensor.GetElementType() != type) {
      FL_THROW(FOUNDRY_LOCAL_ERROR_INVALID_ARGUMENT, fmt::format("speech synthesis graph {}: {} '{}' must be {}",
                                                                 graph.label, input ? "input" : "output", name,
                                                                 TypeName(type)));
    }

    return tensor.GetShape();
  }

  return std::nullopt;
}

std::vector<int64_t> RequireTensor(Graph& graph, bool input, const std::string& name, ONNXTensorElementDataType type) {
  auto shape = FindTensor(graph, input, name, type);
  if (!shape) {
    FL_THROW(FOUNDRY_LOCAL_ERROR_INVALID_ARGUMENT, fmt::format("speech synthesis graph {} needs {} '{}' of type {}",
                                                               graph.label, input ? "input" : "output", name,
                                                               TypeName(type)));
  }

  (input ? graph.input_names : graph.output_names).push_back(name);
  return *shape;
}

/// Load a graph and check that it has exactly the given inputs and outputs, plus a key/value cache when `kv_cache`.
void OpenGraph(Graph& graph, const char* label, Ort::Env& env, const std::filesystem::path& file,
               const Ort::SessionOptions& options, std::initializer_list<TensorSpec> inputs,
               std::initializer_list<TensorSpec> outputs, bool kv_cache) {
  graph.label = label;
  graph.session = std::make_unique<Ort::Session>(env, file.c_str(), options);
  for (const auto& spec : inputs) {
    RequireTensor(graph, true, spec.name, spec.type);
  }

  for (const auto& spec : outputs) {
    RequireTensor(graph, false, spec.name, spec.type);
  }

  graph.head_outputs = outputs.size();
  while (kv_cache && FindTensor(graph, true, fmt::format("past_key_{}", graph.layers), kFloat16)) {
    for (const char* kind : {"key", "value"}) {
      RequireTensor(graph, true, fmt::format("past_{}_{}", kind, graph.layers), kFloat16);
      RequireTensor(graph, false, fmt::format("present_{}_{}", kind, graph.layers), kFloat16);
    }

    ++graph.layers;
  }

  if (kv_cache && graph.layers == 0) {
    FL_THROW(FOUNDRY_LOCAL_ERROR_INVALID_ARGUMENT,
             fmt::format("speech synthesis graph {} needs a key/value cache (past_key_0)", label));
  }

  if (graph.session->GetInputCount() != graph.input_names.size()) {
    FL_THROW(FOUNDRY_LOCAL_ERROR_INVALID_ARGUMENT,
             fmt::format("speech synthesis graph {} has unexpected inputs", label));
  }

  for (const auto& name : graph.input_names) {
    graph.inputs.push_back(name.c_str());
  }

  for (const auto& name : graph.output_names) {
    graph.outputs.push_back(name.c_str());
  }
}

/// The static size of dimension `index` of a graph tensor, or 0 when it is symbolic or missing.
int64_t StaticDim(const std::vector<int64_t>& shape, size_t index) {
  return index < shape.size() && shape[index] > 0 ? shape[index] : 0;
}

/// Agree two sizes that each may be unknown (0).
int64_t Agree(int64_t first, int64_t second, const char* what) {
  if (first > 0 && second > 0 && first != second) {
    FL_THROW(FOUNDRY_LOCAL_ERROR_INVALID_ARGUMENT, fmt::format("speech synthesis graphs disagree on the {}", what));
  }

  return first > 0 ? first : second;
}

uint16_t ToHalf(float value) {
  return Ort::Float16_t(value).val;
}

float FromHalf(uint16_t bits) {
  return Ort::Float16_t::FromBits(bits).ToFloat();
}

Ort::Value HalfTensorView(const Ort::MemoryInfo& memory, const std::vector<uint16_t>& data,
                          const std::vector<int64_t>& shape) {
  // ORT does not write to input tensors, so a view of const data is safe.
  return Ort::Value::CreateTensor(memory, const_cast<uint16_t*>(data.data()), data.size() * sizeof(uint16_t),
                                  shape.data(), shape.size(), kFloat16);
}

Ort::Value Int64TensorView(const Ort::MemoryInfo& memory, std::vector<int64_t>& data) {
  const std::array<int64_t, 2> shape = {1, static_cast<int64_t>(data.size())};
  return Ort::Value::CreateTensor<int64_t>(memory, data.data(), data.size(), shape.data(), shape.size());
}

/// The float16 elements of a graph output.
std::vector<uint16_t> HalfData(const Ort::Value& value) {
  const auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != kFloat16) {
    FL_THROW(FOUNDRY_LOCAL_ERROR_INTERNAL, "speech synthesis graph output must be float16");
  }

  const auto* data = static_cast<const uint16_t*>(value.GetTensorRawData());
  return std::vector<uint16_t>(data, data + info.GetElementCount());
}

/// The last row (last position) of a float16 [..., width] output.
std::vector<uint16_t> LastRow(const Ort::Value& value) {
  const auto shape = value.GetTensorTypeAndShapeInfo().GetShape();
  auto data = HalfData(value);
  const auto width = shape.empty() ? 0 : static_cast<size_t>(std::max<int64_t>(shape.back(), 0));
  if (width == 0 || data.size() < width) {
    FL_THROW(FOUNDRY_LOCAL_ERROR_INTERNAL, "speech synthesis hidden state is empty");
  }

  data.erase(data.begin(), data.end() - static_cast<std::ptrdiff_t>(width));
  return data;
}

/// Run a language model step: `head` inputs followed by `cache`. Replaces `cache` with the present cache and returns
/// the head outputs.
std::vector<Ort::Value> RunLanguageModel(const Graph& graph, std::vector<Ort::Value> head,
                                         std::vector<Ort::Value>& cache) {
  head.reserve(head.size() + cache.size());
  for (auto& value : cache) {
    head.push_back(std::move(value));
  }

  auto outputs = graph.Run(head);
  cache.clear();
  for (size_t i = graph.head_outputs; i < outputs.size(); ++i) {
    cache.push_back(std::move(outputs[i]));
  }

  outputs.erase(outputs.begin() + static_cast<std::ptrdiff_t>(graph.head_outputs), outputs.end());
  return outputs;
}

/// Standard normal samples from a Mersenne Twister with the Box-Muller transform, which gives the same values on every
/// platform (std::normal_distribution does not).
std::vector<float> GaussianNoise(std::mt19937& rng, size_t count) {
  constexpr double kTwoPi = 6.283185307179586;
  std::vector<float> noise(count);
  for (size_t i = 0; i < count; i += 2) {
    const double u1 = (static_cast<double>(rng()) + 0.5) / 4294967296.0;
    const double u2 = (static_cast<double>(rng()) + 0.5) / 4294967296.0;
    const double radius = std::sqrt(-2.0 * std::log(u1));
    noise[i] = static_cast<float>(radius * std::cos(kTwoPi * u2));
    if (i + 1 < count) {
      noise[i + 1] = static_cast<float>(radius * std::sin(kTwoPi * u2));
    }
  }

  return noise;
}

std::string PathToUtf8(const std::filesystem::path& path) {
  const auto utf8 = path.u8string();
  return std::string(utf8.begin(), utf8.end());
}

}  // namespace

struct SpeechSynthesisModelInstance::Runtime {
  // Ort::Env wraps the process-wide OrtEnv singleton; this holds one reference for the session lifetime.
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "foundry_local_tts"};
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Graph text_lm;
  Graph tts_lm;
  Graph diffusion_head;
  Graph acoustic_connector;
  Graph vocoder;
  std::unique_ptr<OgaTokenizer> tokenizer;
  VoicePresetLayout layout;
  int64_t latent_dim = 0;
};

SpeechSynthesisModelInstance::SpeechSynthesisModelInstance(std::string model_id, std::string model_path,
                                                           SpeechSynthesisConfig config, ExecutionProvider ep,
                                                           ILogger& logger)
    : LoadedModelInstance(kKind, std::move(model_id), std::move(model_path), ep),
      config_(std::move(config)),
      logger_(logger),
      runtime_(std::make_unique<Runtime>()) {
  std::error_code ec;
  for (const auto* file : {&config_.text_lm_model, &config_.tts_lm_model, &config_.diffusion_head_model,
                           &config_.acoustic_connector_model, &config_.vocoder_model}) {
    if (!std::filesystem::is_regular_file(*file, ec)) {
      FL_THROW(FOUNDRY_LOCAL_ERROR_INVALID_ARGUMENT,
               fmt::format("speech synthesis graph not found: {}", file->string()));
    }
  }

  auto& rt = *runtime_;
  try {
    Ort::SessionOptions options;
    // Full optimization folds the float16 weight casts once. Without it the CPU EP casts the weights on every run,
    // which is several times slower and uses more memory at run time.
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // Synthesis alternates between five sessions, each with its own thread pool. Spinning threads in the idle pools
    // tripled the CPU time without making synthesis faster.
    options.AddConfigEntry(kOrtSessionOptionsConfigAllowIntraOpSpinning, "0");
    if (ep_ != ExecutionProvider::kDefault && ep_ != ExecutionProvider::kCPU) {
      const std::string registration(EPUtils::EPtoRegistrationName(ep_));
      std::vector<Ort::ConstEpDevice> devices;
      for (const auto& device : rt.env.GetEpDevices()) {
        if (registration == device.EpName()) {
          devices.push_back(device);
        }
      }

      if (devices.empty()) {
        FL_THROW(FOUNDRY_LOCAL_ERROR_INVALID_USAGE,
                 fmt::format("execution provider {} has no device for model {}", registration, model_id_));
      }

      options.AppendExecutionProvider_V2(rt.env, devices, std::unordered_map<std::string, std::string>{});
    }

    OpenGraph(rt.text_lm, "text_lm", rt.env, config_.text_lm_model, options, {{"input_ids", kInt64}},
              {{"hidden_states", kFloat16}}, true);
    OpenGraph(rt.tts_lm, "tts_lm", rt.env, config_.tts_lm_model, options,
              {{"inputs_embeds", kFloat16}, {"tts_text_mask", kInt64}},
              {{"hidden_states", kFloat16}, {"eos_logits", kFloat16}}, true);
    OpenGraph(rt.diffusion_head, "diffusion_head", rt.env, config_.diffusion_head_model, options,
              {{"noisy_latent", kFloat16}, {"timestep", kFloat16}, {"condition", kFloat16}},
              {{"predicted_noise", kFloat16}}, false);
    OpenGraph(rt.acoustic_connector, "acoustic_connector", rt.env, config_.acoustic_connector_model, options,
              {{"latent", kFloat16}}, {{"embedding", kFloat16}}, false);
    OpenGraph(rt.vocoder, "vocoder", rt.env, config_.vocoder_model, options, {{"latents", kFloat16}},
              {{"audio", kFloat16}}, false);
  } catch (const Ort::Exception& ex) {
    FL_LOG_AND_THROW(logger_, FOUNDRY_LOCAL_ERROR_INTERNAL, "failed to load speech synthesis model ", model_id_, ": ",
                     ex.what());
  }

  const auto text_kv = *FindTensor(rt.text_lm, true, "past_key_0", kFloat16);
  const auto tts_kv = *FindTensor(rt.tts_lm, true, "past_key_0", kFloat16);
  rt.layout.text_lm_layers = rt.text_lm.layers;
  rt.layout.tts_lm_layers = rt.tts_lm.layers;
  rt.layout.kv_heads = Agree(StaticDim(text_kv, 1), StaticDim(tts_kv, 1), "key/value heads");
  rt.layout.head_dim = Agree(StaticDim(text_kv, 3), StaticDim(tts_kv, 3), "head size");
  rt.layout.hidden_size =
      Agree(StaticDim(*FindTensor(rt.tts_lm, true, "inputs_embeds", kFloat16), 2),
            StaticDim(*FindTensor(rt.diffusion_head, true, "condition", kFloat16), 1), "hidden size");
  rt.latent_dim = Agree(StaticDim(*FindTensor(rt.diffusion_head, true, "noisy_latent", kFloat16), 1),
                        StaticDim(*FindTensor(rt.acoustic_connector, true, "latent", kFloat16), 2), "latent size");
  if (rt.latent_dim == 0) {
    FL_THROW(FOUNDRY_LOCAL_ERROR_INVALID_ARGUMENT, "speech synthesis graph diffusion_head needs a static latent size");
  }

  try {
    rt.tokenizer = OgaTokenizer::Create(PathToUtf8(config_.tokenizer_dir).c_str());
    const char* keys[] = {"literal"};
    const char* values[] = {"true"};
    rt.tokenizer->UpdateOptions(keys, values, 1);
  } catch (const std::exception& ex) {
    FL_THROW(FOUNDRY_LOCAL_ERROR_INVALID_ARGUMENT,
             fmt::format("cannot load the speech synthesis tokenizer {}: {}", config_.tokenizer_dir.string(),
                         ex.what()));
  }

  uint64_t voice_bytes = 0;
  for (const auto& voice : config_.voices) {
    auto preset = LoadVoicePreset(voice.file, rt.layout, kMaxVoiceFileBytes);
    voice_bytes += preset.ByteSize();
    if (voice_bytes > kMaxVoiceTotalBytes) {
      FL_THROW(FOUNDRY_LOCAL_ERROR_INVALID_ARGUMENT,
               fmt::format("voices of model {} use more than {} bytes", model_id_, kMaxVoiceTotalBytes));
    }

    voices_.emplace(voice.name, std::move(preset));
  }

  logger_.Log(LogLevel::Information,
              fmt::format("speech synthesis model loaded: {} ({} voices)", model_id_, config_.voices.size()));
}

SpeechSynthesisModelInstance::~SpeechSynthesisModelInstance() = default;

const SpeechSynthesisVoice& SpeechSynthesisModelInstance::ResolveVoice(const std::string& name) const {
  const auto* voice = config_.FindVoice(name.empty() ? config_.default_voice : name);
  if (!voice) {
    FL_THROW(FOUNDRY_LOCAL_ERROR_INVALID_ARGUMENT, fmt::format("unknown voice: {}", name));
  }

  return *voice;
}

std::vector<int64_t> SpeechSynthesisModelInstance::Tokenize(std::string_view text) const {
  // The VibeVoice demo replaces typographic quotes, and its processor strips the text and ends it with a line break.
  std::string prepared;
  prepared.reserve(text.size() + 1);
  for (size_t i = 0; i < text.size(); ++i) {
    if (text.substr(i, 2) == "\xE2\x80" && i + 2 < text.size()) {
      const auto third = static_cast<unsigned char>(text[i + 2]);
      if (third == 0x98 || third == 0x99 || third == 0x9C || third == 0x9D) {
        prepared += third <= 0x99 ? '\'' : '"';
        i += 2;
        continue;
      }
    }

    // The tokenizer reads a C string, so a NUL would end the text early.
    prepared += text[i] == '\0' ? ' ' : text[i];
  }

  constexpr std::string_view kSpace = " \t\r\n\f\v";
  const auto first = prepared.find_first_not_of(kSpace);
  if (first == std::string::npos) {
    return {};
  }

  prepared = prepared.substr(first, prepared.find_last_not_of(kSpace) + 1 - first) + "\n";
  auto sequences = OgaSequences::Create();
  try {
    runtime_->tokenizer->Encode(prepared.c_str(), *sequences);
  } catch (const std::exception& ex) {
    FL_THROW(FOUNDRY_LOCAL_ERROR_INVALID_ARGUMENT, fmt::format("cannot tokenize speech text: {}", ex.what()));
  }

  const auto ids = sequences->Get(0);
  return std::vector<int64_t>(ids.begin(), ids.end());
}

std::vector<float> SpeechSynthesisModelInstance::SampleLatent(const std::vector<uint16_t>& condition,
                                                              const std::vector<uint16_t>& negative_condition,
                                                              std::mt19937& rng) const {
  const auto& rt = *runtime_;
  const auto latent_dim = static_cast<size_t>(rt.latent_dim);
  const std::vector<int64_t> latent_shape = {1, rt.latent_dim};
  const std::vector<int64_t> timestep_shape = {1};
  const std::vector<int64_t> condition_shape = {1, static_cast<int64_t>(condition.size())};
  auto predict = [&](size_t step, const std::vector<float>& sample) {
    std::vector<uint16_t> noisy(sample.size());
    std::transform(sample.begin(), sample.end(), noisy.begin(), ToHalf);
    const std::vector<uint16_t> timestep = {ToHalf(config_.diffusion.timesteps[step])};

    // The diffusion head has batch size 1, so the conditioned and the unconditioned prediction are two runs.
    auto run = [&](const std::vector<uint16_t>& cond) {
      std::vector<Ort::Value> inputs;
      inputs.push_back(HalfTensorView(rt.memory_info, noisy, latent_shape));
      inputs.push_back(HalfTensorView(rt.memory_info, timestep, timestep_shape));
      inputs.push_back(HalfTensorView(rt.memory_info, cond, condition_shape));
      const auto data = HalfData(rt.diffusion_head.Run(inputs).front());
      if (data.size() != latent_dim) {
        FL_THROW(FOUNDRY_LOCAL_ERROR_INTERNAL, "speech synthesis diffusion head output has the wrong size");
      }

      return data;
    };

    const auto positive = run(condition);
    const auto negative = run(negative_condition);
    std::vector<float> velocity(latent_dim);
    for (size_t i = 0; i < latent_dim; ++i) {
      const float uncond = FromHalf(negative[i]);
      velocity[i] = uncond + config_.cfg_scale * (FromHalf(positive[i]) - uncond);
    }

    return velocity;
  };

  return SampleDpmSolver(config_.diffusion, GaussianNoise(rng, latent_dim), predict);
}

SynthesizedSpeech SpeechSynthesisModelInstance::Synthesize(std::string_view text, const SpeechSynthesisVoice& voice,
                                                           const std::function<bool()>& cancelled) const {
  auto ids = Tokenize(text);
  if (ids.empty()) {
    return {};
  }

  if (ids.size() > kMaxTextTokens) {
    FL_THROW(FOUNDRY_LOCAL_ERROR_INVALID_ARGUMENT,
             fmt::format("speech text chunk has {} tokens; at most {} are supported", ids.size(), kMaxTextTokens));
  }

  const auto& rt = *runtime_;
  const auto& preset = voices_.at(voice.name);
  auto cache_of = [&](const std::vector<HalfTensor>& tensors) {
    std::vector<Ort::Value> cache;
    for (const auto& tensor : tensors) {
      cache.push_back(HalfTensorView(rt.memory_info, tensor.data, tensor.shape));
    }

    return cache;
  };

  try {
    auto text_cache = cache_of(preset.text_lm_kv);
    auto tts_cache = cache_of(preset.tts_lm_kv);
    auto negative_cache = cache_of(preset.negative_tts_lm_kv);
    std::vector<uint16_t> condition;
    auto negative_condition = preset.negative_condition;
    std::vector<float> latents;
    std::mt19937 rng(kNoiseSeed);
    const auto text_window = static_cast<size_t>(config_.text_window_size);
    const size_t max_frames = kMinFrameBudget + kFramesPerToken * ids.size();
    size_t frames = 0;
    bool done = false;
    for (size_t begin = 0; !done; begin += text_window) {
      // Feed the next window of text through the text LM into the TTS LM (tts_text_mask 1 marks text positions).
      if (begin < ids.size()) {
        const auto end = std::min(begin + text_window, ids.size());
        std::vector<int64_t> window(ids.begin() + static_cast<std::ptrdiff_t>(begin),
                                    ids.begin() + static_cast<std::ptrdiff_t>(end));
        std::vector<Ort::Value> text_inputs;
        text_inputs.push_back(Int64TensorView(rt.memory_info, window));
        auto hidden = RunLanguageModel(rt.text_lm, std::move(text_inputs), text_cache);
        std::vector<int64_t> mask(window.size(), 1);
        std::vector<Ort::Value> tts_inputs;
        tts_inputs.push_back(std::move(hidden.front()));
        tts_inputs.push_back(Int64TensorView(rt.memory_info, mask));
        condition = LastRow(RunLanguageModel(rt.tts_lm, std::move(tts_inputs), tts_cache).front());
      }

      const bool all_text_fed = begin + text_window >= ids.size();
      for (int i = 0; i < config_.speech_window_size && !done; ++i) {
        if (cancelled && cancelled()) {
          return {};
        }

        const auto latent = SampleLatent(condition, negative_condition, rng);
        latents.insert(latents.end(), latent.begin(), latent.end());
        ++frames;

        // Feed the latent back to both TTS LMs through the acoustic connector (tts_text_mask 0 marks speech).
        std::vector<uint16_t> latent_half(latent.size());
        std::transform(latent.begin(), latent.end(), latent_half.begin(), ToHalf);
        std::vector<Ort::Value> connector_inputs;
        connector_inputs.push_back(HalfTensorView(rt.memory_info, latent_half, {1, 1, rt.latent_dim}));
        const auto embedding = HalfData(rt.acoustic_connector.Run(connector_inputs).front());
        const std::vector<int64_t> embedding_shape = {1, 1, static_cast<int64_t>(embedding.size())};
        std::vector<int64_t> speech_mask = {0};
        auto speech_inputs = [&] {
          std::vector<Ort::Value> inputs;
          inputs.push_back(HalfTensorView(rt.memory_info, embedding, embedding_shape));
          inputs.push_back(Int64TensorView(rt.memory_info, speech_mask));
          return inputs;
        };

        auto outputs = RunLanguageModel(rt.tts_lm, speech_inputs(), tts_cache);
        condition = LastRow(outputs[0]);
        const auto eos = HalfData(outputs[1]);
        // VibeVoice stops when sigmoid(eos_logit) > 0.5. Ignoring EOS until all text is fed keeps a float16 rounding
        // difference from cutting off the text.
        if ((all_text_fed && !eos.empty() && FromHalf(eos.front()) > 0.0f) || frames >= max_frames) {
          done = true;
          break;
        }

        negative_condition = LastRow(RunLanguageModel(rt.tts_lm, speech_inputs(), negative_cache).front());
      }
    }

    // Undo the latent scaling and decode all frames at once.
    std::vector<uint16_t> scaled(latents.size());
    for (size_t i = 0; i < latents.size(); ++i) {
      scaled[i] = ToHalf(latents[i] / config_.speech_scaling_factor - config_.speech_bias_factor);
    }

    std::vector<Ort::Value> vocoder_inputs;
    vocoder_inputs.push_back(
        HalfTensorView(rt.memory_info, scaled, {1, static_cast<int64_t>(frames), rt.latent_dim}));
    const auto audio = HalfData(rt.vocoder.Run(vocoder_inputs).front());
    SynthesizedSpeech result;
    result.samples.resize(audio.size());
    std::transform(audio.begin(), audio.end(), result.samples.begin(), FromHalf);
    result.text_tokens = ids.size();
    result.speech_frames = frames;
    return result;
  } catch (const Ort::Exception& ex) {
    FL_LOG_AND_THROW(logger_, FOUNDRY_LOCAL_ERROR_INTERNAL, "speech synthesis failed for model ", model_id_, ": ",
                     ex.what());
  }
}

}  // namespace fl
