// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "catalog/local_model_catalog.h"
#include "exception.h"
#include "inferencing/generative/speech_synthesis/dpm_solver.h"
#include "inferencing/generative/speech_synthesis/speech_synthesis_config.h"
#include "inferencing/generative/speech_synthesis/speech_synthesis_model_instance.h"
#include "inferencing/generative/speech_synthesis/speech_synthesis_session.h"
#include "inferencing/generative/speech_synthesis/voice_preset.h"
#include "inferencing/model_load_manager.h"
#include "inferencing/model_package.h"
#include "items/audio_item.h"
#include "items/item_queue.h"
#include "items/text_item.h"
#include "model.h"
#include "internal_api/speech_synthesis/speech_synthesis_test_helpers.h"
#include "internal_api/test_helpers.h"
#include "internal_api/test_model_cache.h"
#include "utils/temp_path.h"

#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace fl {
namespace {

using test::ErrorCode;
using test::TinyVibeVoicePath;

constexpr auto kInvalid = FOUNDRY_LOCAL_ERROR_INVALID_ARGUMENT;

float Half(uint16_t bits) {
  return Ort::Float16_t::FromBits(bits).ToFloat();
}

TEST(SpeechSynthesisPackage, ParsesTinyVibeVoiceManifest) {
  EXPECT_EQ(DetectModelPackageKind(TinyVibeVoicePath()), ModelPackageKind::kSpeechSynthesis);
  const auto config = SpeechSynthesisConfig::LoadFromDirectory(TinyVibeVoicePath());
  EXPECT_EQ(config.default_voice, "Alpha");
  ASSERT_EQ(config.voices.size(), 2u);
  EXPECT_EQ(config.voices[1].language, "en-in");
  EXPECT_EQ(config.voices[1].gender, "male");
  EXPECT_EQ(config.vocoder_model.filename(), "vocoder.onnx");
  EXPECT_EQ(config.tokenizer_dir.filename(), "tokenizer");
  EXPECT_FLOAT_EQ(config.speech_scaling_factor, 0.5f);
  EXPECT_FLOAT_EQ(config.cfg_scale, 2.0f);
  EXPECT_EQ(config.text_window_size, 5);
  EXPECT_EQ(config.speech_window_size, 6);
  EXPECT_EQ(config.diffusion.timesteps, std::vector<float>{500.0f});
  EXPECT_EQ(config.diffusion.sigmas, (std::vector<double>{1.0, 0.0}));
}

TEST(SpeechSynthesisPackage, RejectsInvalidManifests) {
  const auto dir = TinyVibeVoicePath();
  auto parse = [&](const std::string& json) { return ErrorCode([&] { SpeechSynthesisConfig::Parse(json, dir); }); };
  const std::string graphs =
      R"("graphs":{"text_lm":"a.onnx","tts_lm":"b.onnx","diffusion_head":"c.onnx","acoustic_connector":"d.onnx",)"
      R"("vocoder":"e.onnx"},"tokenizer":"tokenizer",)";
  const std::string scale = R"("speech_scaling_factor":0.5,"speech_bias_factor":0.1,)";
  const std::string diffusion = R"("diffusion":{"timesteps":[999,500],"sigmas":[10,1,0]},)";
  const std::string voices = R"("voices":[{"name":"v","file":"voices/alpha.npz","language":"en"}])";
  auto manifest = [&](const std::string& body) { return R"({"model_type":"vibevoice",)" + body + "}"; };
  const auto valid = manifest(graphs + scale + diffusion + voices);

  ASSERT_EQ(parse(valid), FOUNDRY_LOCAL_OK);
  const auto config = SpeechSynthesisConfig::Parse(valid, dir);
  EXPECT_EQ(config.default_voice, "v");
  EXPECT_FLOAT_EQ(config.cfg_scale, 1.5f);
  EXPECT_EQ(parse("[]"), kInvalid);
  EXPECT_EQ(parse(R"({"model_type":"other",)" + graphs + scale + diffusion + voices + "}"), kInvalid);
  EXPECT_EQ(parse(manifest(scale + diffusion + voices)), kInvalid);
  EXPECT_EQ(parse(manifest(R"("graphs":{"text_lm":"../x.onnx"},"tokenizer":"t",)" + scale + diffusion + voices)),
            kInvalid);
  EXPECT_EQ(parse(manifest(graphs + R"("speech_scaling_factor":0,"speech_bias_factor":0,)" + diffusion + voices)),
            kInvalid);
  EXPECT_EQ(parse(manifest(graphs + R"("speech_bias_factor":0,)" + diffusion + voices)), kInvalid);
  EXPECT_EQ(parse(manifest(graphs + scale + R"("cfg_scale":"high",)" + diffusion + voices)), kInvalid);
  EXPECT_EQ(parse(manifest(graphs + scale + R"("text_window_size":0,)" + diffusion + voices)), kInvalid);
  EXPECT_EQ(parse(manifest(graphs + scale + R"("diffusion":{"timesteps":[1],"sigmas":[1]},)" + voices)), kInvalid);
  EXPECT_EQ(parse(manifest(graphs + scale + R"("diffusion":{},)" + voices)), kInvalid);
  EXPECT_EQ(parse(manifest(graphs + scale + R"("diffusion":{"timesteps":[1]},)" + voices)), kInvalid);
  EXPECT_EQ(parse(manifest(graphs + scale + R"("diffusion":{"sigmas":[1,0]},)" + voices)), kInvalid);
  EXPECT_EQ(parse(manifest(graphs + scale + R"("diffusion":{"timesteps":[1,2],"sigmas":[1,2,0]},)" + voices)),
            kInvalid);
  EXPECT_EQ(parse(manifest(graphs + scale + diffusion + R"("default_voice":"x",)" + voices)), kInvalid);
  EXPECT_EQ(parse(manifest(graphs + scale + diffusion + R"("voices":[])")), kInvalid);
  EXPECT_EQ(parse(manifest(graphs + scale + diffusion +
                           R"("voices":[{"name":"v","file":"/etc/v.npz","language":"en"}])")),
            kInvalid);
  EXPECT_EQ(parse(manifest(graphs + scale + diffusion + R"("voices":[{"name":"v","file":"v.npz"}])")), kInvalid);
  EXPECT_EQ(parse(manifest(graphs + scale + diffusion +
                           R"("voices":[{"name":"v","file":"a.npz","language":"en"},)"
                           R"({"name":"v","file":"b.npz","language":"en"}])")),
            kInvalid);
}

TEST(SpeechSynthesisPackage, DpmSolverMatchesReferenceScheduler) {
  // The VibeVoice 5-step schedule and the values of the NumPy DPM-Solver++ in the converter's vibevoice_full_onnx.py
  // for this velocity function.
  DiffusionSchedule schedule{{999, 799, 599, 400, 200},
                             {20291.302734375, 3.103729724884033, 1.3907514810562134, 0.7402812242507935,
                              0.3374606668949127, 0.0}};
  const auto result = SampleDpmSolver(schedule, {0.5f, -1.0f, 2.0f}, [](size_t step, const std::vector<float>& x) {
    std::vector<float> velocity(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
      velocity[i] = static_cast<float>(0.3 * x[i] + 0.1 * static_cast<double>(step) + 0.05 * static_cast<double>(i));
    }

    return velocity;
  });

  ASSERT_EQ(result.size(), 3u);
  EXPECT_NEAR(result[0], -0.02794009, 1e-5);
  EXPECT_NEAR(result[1], -0.89829147, 1e-5);
  EXPECT_NEAR(result[2], 0.66338938, 1e-5);
  EXPECT_EQ(ErrorCode([&] { SampleDpmSolver({{1}, {1}}, {0.0f}, {}); }), kInvalid);
}

TEST(SpeechSynthesisPackage, LoadsStoredAndDeflatedVoicePresets) {
  const VoicePresetLayout layout{1, 1, 1, 2, 4};
  const auto voices = TinyVibeVoicePath() / "voices";
  const auto alpha = LoadVoicePreset(voices / "alpha.npz", layout, 1 << 20);
  ASSERT_EQ(alpha.tts_lm_kv.size(), 2u);
  EXPECT_EQ(alpha.tts_lm_kv[0].shape, (std::vector<int64_t>{1, 1, 3, 2}));
  EXPECT_EQ(alpha.text_lm_kv[1].shape, (std::vector<int64_t>{1, 1, 2, 2}));
  EXPECT_NEAR(Half(alpha.tts_lm_kv[1].data[0]), 0.02f, 1e-4);
  ASSERT_EQ(alpha.negative_condition.size(), 4u);
  EXPECT_NEAR(Half(alpha.negative_condition[3]), 0.05f, 1e-4);

  const auto beta = LoadVoicePreset(voices / "beta.npz", layout, 1 << 20);
  EXPECT_EQ(beta.tts_lm_kv[0].shape, (std::vector<int64_t>{1, 1, 5, 2}));
  EXPECT_NEAR(Half(beta.negative_condition[0]), -0.05f, 1e-4);

  EXPECT_EQ(ErrorCode([&] { LoadVoicePreset(voices / "alpha.npz", {1, 1, 2, 2, 4}, 1 << 20); }), kInvalid);
  EXPECT_EQ(ErrorCode([&] { LoadVoicePreset(voices / "alpha.npz", {2, 1, 1, 2, 4}, 1 << 20); }), kInvalid);
  EXPECT_EQ(ErrorCode([&] { LoadVoicePreset(voices / "alpha.npz", layout, 100); }), kInvalid);
  EXPECT_EQ(ErrorCode([&] { LoadVoicePreset(TinyVibeVoicePath() / "tts_config.json", layout, 1 << 20); }), kInvalid);
  EXPECT_EQ(ErrorCode([&] { LoadVoicePreset(voices / "missing.npz", layout, 1 << 20); }), kInvalid);
}

/// What the tiny graphs (see testdata/tiny-vibevoice/generate-model.py) produce for `text`: four samples per frame.
std::vector<float> TinySpeech(const std::string& text, int64_t tts_prompt, float negative) {
  std::vector<int> ids(text.begin(), text.end());
  ids.push_back('\n');
  const size_t max_frames = 32 + 4 * ids.size();
  int64_t length = tts_prompt;
  float condition = 0.0f;
  std::vector<float> samples;
  for (size_t begin = 0;; begin += 5) {
    if (begin < ids.size()) {
      const auto end = std::min(begin + 5, ids.size());
      condition = 0.001f * static_cast<float>(ids[end - 1]) + 0.1f;
      length += static_cast<int64_t>(end - begin);
    }

    for (int i = 0; i < 6; ++i) {
      const float latent = negative + 2.0f * (condition - negative);
      samples.insert(samples.end(), 4, latent / 0.5f - 0.1f);
      condition = 0.5f * latent;
      ++length;
      if ((begin + 5 >= ids.size() && length > 40) || samples.size() / 4 >= max_frames) {
        return samples;
      }

      negative = condition;
    }
  }
}

class SpeechSynthesisSessionTest : public ::testing::Test {
 protected:
  static constexpr const char* kModelId = "tiny-vibevoice:1";

  static void SetUpTestSuite() {
    ep_detector_ = std::make_unique<test::CpuOnlyEpDetector>();
    load_manager_ = std::make_unique<ModelLoadManager>(*ep_detector_, test::NullLog());
    const auto result = load_manager_->LoadModel(TinyVibeVoicePath().string(), kModelId);
    ASSERT_EQ(result.status, ModelLoadManager::LoadStatus::kSuccess);
    model_ = result.As<SpeechSynthesisModelInstance>();
    ASSERT_NE(model_, nullptr);
  }

  static void TearDownTestSuite() {
    if (load_manager_) {
      load_manager_->UnloadModel(kModelId);
    }

    load_manager_.reset();
    ep_detector_.reset();
    model_ = nullptr;
  }

  static void ExpectAudio(const Item& item, const std::vector<float>& expected) {
    ASSERT_EQ(item.type, FOUNDRY_LOCAL_ITEM_AUDIO);
    const auto& audio = static_cast<const AudioItem&>(item);
    EXPECT_EQ(audio.format, "pcm");
    EXPECT_EQ(audio.sample_rate, 24000);
    EXPECT_EQ(audio.channels, 1);
    ASSERT_EQ(audio.data_size, expected.size() * 2);
    const auto* bytes = static_cast<const uint8_t*>(audio.data);
    for (size_t i = 0; i < expected.size(); ++i) {
      // float16 graphs and the random diffusion noise leave a small error in every sample.
      const auto sample = static_cast<int16_t>(bytes[2 * i] | (bytes[2 * i + 1] << 8));
      EXPECT_NEAR(sample / 32767.0f, expected[i], 0.02f) << "sample " << i;
    }
  }

  std::unique_ptr<SpeechSynthesisSession> MakeSession() {
    return std::make_unique<SpeechSynthesisSession>(catalog_model_, *model_, test::NullLog(),
                                                    test::TestTelemetrySink());
  }

  static Request MakeRequest(const std::string& text, std::map<std::string, std::string> options = {}) {
    Request request;
    request.AddOwnedItem(std::make_unique<TextItem>(text));
    for (auto& [key, value] : options) {
      request.options.Add(key, value);
    }

    return request;
  }

  static inline std::unique_ptr<test::CpuOnlyEpDetector> ep_detector_;
  static inline std::unique_ptr<ModelLoadManager> load_manager_;
  static inline SpeechSynthesisModelInstance* model_ = nullptr;
  static inline test::FakeServiceBindings svc_;
  static inline Model catalog_model_ =
      Model::FromModelInfo(ModelInfo{}, "", svc_.download_manager, svc_.model_load_manager);
};

TEST_F(SpeechSynthesisSessionTest, TokenizesLiteralTextWithALineBreak) {
  auto bytes = [](const std::string& text) { return std::vector<int64_t>(text.begin(), text.end()); };
  EXPECT_EQ(model_->Tokenize("  It\xE2\x80\x99s \xE2\x80\x9Cok\xE2\x80\x9D \n"), bytes("It's \"ok\"\n"));
  EXPECT_EQ(model_->Tokenize("<|endoftext|>"), bytes("<|endoftext|>\n"));
  EXPECT_TRUE(model_->Tokenize(" \t\n").empty());
}

TEST_F(SpeechSynthesisSessionTest, SynthesizesText) {
  auto session = MakeSession();
  Response response;
  session->ProcessRequest(MakeRequest("Hello world."), response);
  ASSERT_EQ(response.items.size(), 1u);
  const auto expected = TinySpeech("Hello world.", 3, 0.05f);
  ExpectAudio(*response.items[0], expected);
  EXPECT_STREQ(response.items[0]->GetMetadata().Find("voice"), "Alpha");
  EXPECT_EQ(response.finish_reason, FOUNDRY_LOCAL_FINISH_STOP);
  EXPECT_EQ(response.usage.prompt_tokens, 13);
  EXPECT_EQ(response.usage.completion_tokens, static_cast<int64_t>(expected.size() / 4));
  EXPECT_EQ(response.usage.total_tokens, 13 + static_cast<int64_t>(expected.size() / 4));

  Response again;
  session->ProcessRequest(MakeRequest("Hello world."), again);
  const auto& first = static_cast<const AudioItem&>(*response.items[0]);
  const auto& second = static_cast<const AudioItem&>(*again.items[0]);
  ASSERT_EQ(first.data_size, second.data_size);
  EXPECT_EQ(std::memcmp(first.data, second.data, first.data_size), 0) << "fixed noise seed";
}

TEST_F(SpeechSynthesisSessionTest, AppliesVoiceAndAcceptsSpeedOne) {
  auto session = MakeSession();
  KeyValuePairs defaults;
  defaults.Add(FOUNDRY_LOCAL_PARAM_VOICE, "Beta");
  session->SetSessionOptions(defaults);

  Response beta;
  session->ProcessRequest(MakeRequest("Hello world.", {{FOUNDRY_LOCAL_PARAM_SPEED, "1.0"}}), beta);
  ExpectAudio(*beta.items[0], TinySpeech("Hello world.", 5, -0.05f));
  EXPECT_STREQ(beta.items[0]->GetMetadata().Find("voice"), "Beta");

  Response alpha;
  session->ProcessRequest(MakeRequest("Hello world.", {{FOUNDRY_LOCAL_PARAM_VOICE, "Alpha"}}), alpha);
  ExpectAudio(*alpha.items[0], TinySpeech("Hello world.", 3, 0.05f));
}

TEST_F(SpeechSynthesisSessionTest, StreamsOneAudioItemPerChunk) {
  auto session = MakeSession();
  std::vector<std::unique_ptr<Item>> streamed;
  session->SetStreamingCallback([&](flStreamingCallbackData event, void*) {
    auto* queue = reinterpret_cast<ItemQueue*>(event.item_queue);
    while (auto item = queue->TryPop()) {
      streamed.push_back(std::move(item));
    }

    return 0;
  });

  const std::vector<std::string> chunks = {"The first sentence is long enough.", "Hi. The second one is long too!",
                                           "A third line, also long."};
  Response response;
  session->ProcessRequest(MakeRequest(chunks[0] + " " + chunks[1] + "\n" + chunks[2]), response);
  ASSERT_EQ(streamed.size(), 3u);
  std::vector<float> combined;
  for (size_t i = 0; i < chunks.size(); ++i) {
    const auto expected = TinySpeech(chunks[i], 3, 0.05f);
    ExpectAudio(*streamed[i], expected);
    EXPECT_EQ(streamed[i]->GetMetadata().Find("text"), chunks[i]);
    combined.insert(combined.end(), expected.begin(), expected.end());
  }

  ExpectAudio(*response.items[0], combined);
}

TEST_F(SpeechSynthesisSessionTest, StopsAfterCancellation) {
  auto session = MakeSession();
  size_t streamed = 0;
  session->SetStreamingCallback([&](flStreamingCallbackData event, void*) {
    auto* queue = reinterpret_cast<ItemQueue*>(event.item_queue);
    while (queue->TryPop()) {
      ++streamed;
    }

    return 1;
  });

  // The callback runs on its own thread, so synthesis may finish more chunks before it sees the cancellation.
  Response response;
  const std::string line = "This sentence is long enough to be a chunk.\n";
  const auto code = ErrorCode([&] { session->ProcessRequest(MakeRequest(line + line + line + line), response); });
  EXPECT_EQ(code, FOUNDRY_LOCAL_ERROR_OPERATION_CANCELLED);
  EXPECT_LE(streamed, 4u);

  streamed = 0;
  auto cancelled = MakeRequest(line + line);
  cancelled.CancelCurrentOrNext();
  Response cancelled_response;
  EXPECT_EQ(ErrorCode([&] { session->ProcessRequest(cancelled, cancelled_response); }),
            FOUNDRY_LOCAL_ERROR_OPERATION_CANCELLED);
  EXPECT_EQ(streamed, 0u);
  EXPECT_TRUE(model_->Synthesize("Hello world.", model_->ResolveVoice(""), [] { return true; }).samples.empty());
}

TEST_F(SpeechSynthesisSessionTest, RejectsInvalidRequests) {
  auto session = MakeSession();
  auto run = [&](Request request) {
    return ErrorCode([&] {
      Response response;
      session->ProcessRequest(request, response);
    });
  };

  EXPECT_EQ(run(Request{}), kInvalid);
  EXPECT_EQ(run(MakeRequest("hi", {{FOUNDRY_LOCAL_PARAM_SPEED, "1.5"}})), kInvalid);
  EXPECT_EQ(run(MakeRequest("hi", {{FOUNDRY_LOCAL_PARAM_SPEED, "fast"}})), kInvalid);
  EXPECT_EQ(run(MakeRequest("hi", {{FOUNDRY_LOCAL_PARAM_VOICE, "nobody"}})), kInvalid);
  EXPECT_EQ(run(MakeRequest(std::string(16385, 'a'))), kInvalid);
  // Continuation bytes alone count as no characters, so invalid UTF-8 must not get past the length limit.
  EXPECT_EQ(run(MakeRequest(std::string(20000, '\x80'))), kInvalid);
  EXPECT_EQ(run(MakeRequest("   \n  ")), FOUNDRY_LOCAL_OK);
  EXPECT_EQ(ErrorCode([] { model_->Synthesize(std::string(1100, 'a'), model_->ResolveVoice("")); }), kInvalid);

  Request image;
  image.AddOwnedItem(Item::Create(FOUNDRY_LOCAL_ITEM_IMAGE));
  EXPECT_EQ(run(std::move(image)), kInvalid);
}

TEST_F(SpeechSynthesisSessionTest, RefusesUnloadWhileSessionIsAlive) {
  auto session = MakeSession();
  EXPECT_EQ(ErrorCode([] { load_manager_->UnloadModel(kModelId); }), FOUNDRY_LOCAL_ERROR_INVALID_USAGE);
  session.reset();
  EXPECT_EQ(model_->SessionRefCount(), 0);
}

TEST(SpeechSynthesisRegistration, RegistersPackageInLocalCatalog) {
  auto root = test::TempPath::CreateTempDir("speech_synthesis_catalog");
  test::FakeServiceBindings bindings;
  LocalModelCatalog catalog(
      root.path() / "cache" / "models",
      [&](ModelInfo info, std::string path) {
        return Model::FromLocalRegistration(std::move(info), std::move(path), bindings.download_manager,
                                            bindings.model_load_manager);
      },
      test::NullLog());

  ModelInfo metadata;
  metadata.SetPropertyStr(FOUNDRY_LOCAL_MODEL_PROP_TASK_STR, "text-to-speech");
  auto* model = catalog.RegisterModel(TinyVibeVoicePath().string(), "vibevoice-test:1", metadata);
  ASSERT_NE(model, nullptr);
  EXPECT_TRUE(model->IsCached());
  EXPECT_EQ(model->Info().task, "text-to-speech");
  EXPECT_EQ(*model->Info().GetPropertyStr(FOUNDRY_LOCAL_MODEL_PROP_INPUT_MODALITIES_STR), "text");
  EXPECT_EQ(*model->Info().GetPropertyStr(FOUNDRY_LOCAL_MODEL_PROP_OUTPUT_MODALITIES_STR), "audio");
  const auto io = model->GetInputOutputInfo();
  ASSERT_EQ(io.num_inputs, 1u);
  ASSERT_EQ(io.num_outputs, 1u);
  EXPECT_EQ(io.inputs[0]->type, FOUNDRY_LOCAL_ITEM_TEXT);
  EXPECT_EQ(io.outputs[0]->type, FOUNDRY_LOCAL_ITEM_AUDIO);

  ModelInfo chat;
  chat.SetPropertyStr(FOUNDRY_LOCAL_MODEL_PROP_TASK_STR, "chat-completion");
  EXPECT_EQ(ErrorCode([&] { catalog.RegisterModel(TinyVibeVoicePath().string(), "vibevoice-chat:1", chat); }),
            kInvalid);

  // Broken packages fail the load with INVALID_ARGUMENT: a voice that is not an archive, a tokenizer without the
  // genai_config.json that ORT GenAI needs, and a missing graph.
  auto broken = [&](const std::string& name, const std::function<void(const std::filesystem::path&)>& damage) {
    const auto package = root.path() / name;
    std::filesystem::copy(TinyVibeVoicePath(), package, std::filesystem::copy_options::recursive);
    damage(package);
    return ErrorCode([&] { bindings.model_load_manager.LoadModel(package.string(), name + ":1"); });
  };

  EXPECT_EQ(broken("bad-voice",
                   [](const auto& dir) { std::ofstream(dir / "voices" / "beta.npz", std::ios::trunc) << "123456"; }),
            kInvalid);
  EXPECT_EQ(broken("bad-tokenizer",
                   [](const auto& dir) { std::filesystem::remove(dir / "tokenizer" / "genai_config.json"); }),
            kInvalid);
  EXPECT_EQ(broken("bad-graph", [](const auto& dir) { std::filesystem::remove(dir / "vocoder.onnx"); }), kInvalid);
}

}  // namespace
}  // namespace fl
