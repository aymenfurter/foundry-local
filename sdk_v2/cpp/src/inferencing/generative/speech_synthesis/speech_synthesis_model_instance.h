// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "inferencing/generative/speech_synthesis/speech_synthesis_config.h"
#include "inferencing/generative/speech_synthesis/voice_preset.h"
#include "inferencing/loaded_model_instance.h"
#include "logger.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace fl {

/// The audio of one synthesized text chunk.
struct SynthesizedSpeech {
  std::vector<float> samples;  ///< Mono, SpeechSynthesisModelInstance::kSampleRate.
  size_t text_tokens = 0;
  size_t speech_frames = 0;
};

/// A VibeVoice-Realtime text-to-speech model in plain ONNX Runtime sessions. All tensors are float16, except token ids
/// and masks (int64):
///   text_lm             input_ids [1, n] + past_key/value_<i>                  -> hidden_states [1, n, H] + present_*
///   tts_lm              inputs_embeds [1, n, H] + tts_text_mask [1, n] + past -> hidden_states, eos_logits [1, 1]
///   diffusion_head      noisy_latent [1, L] + timestep [1] + condition [1, H]  -> predicted_noise [1, L]
///   acoustic_connector  latent [1, 1, L]                                        -> embedding [1, 1, H]
///   vocoder             latents [1, F, L]                                       -> audio [1, S] at 24 kHz
/// Text is tokenized with the package tokenizer through ORT GenAI, as literal text. The text LM encodes windows of
/// text tokens, and after each window the TTS LM generates a window of speech latents with the diffusion head and
/// classifier-free guidance against the negative (unconditioned) TTS LM of the voice. Generation ends when the EOS
/// classifier fires after all text is fed, and the vocoder decodes all latents at once.
///
/// Sessions, the tokenizer and voices are loaded in the constructor and are immutable afterward, so concurrent
/// synthesis needs no lock.
class SpeechSynthesisModelInstance : public LoadedModelInstance {
 public:
  static constexpr ModelPackageKind kKind = ModelPackageKind::kSpeechSynthesis;
  static constexpr int kSampleRate = 24000;

  ~SpeechSynthesisModelInstance() override;

  const SpeechSynthesisConfig& Config() const { return config_; }

  /// The named voice, or the default voice for an empty name. @throws fl::Exception(INVALID_ARGUMENT) if unknown.
  const SpeechSynthesisVoice& ResolveVoice(const std::string& name) const;

  /// The token ids that VibeVoice reads for `text`: curly quotes become straight quotes, surrounding white space is
  /// removed and a line break ends the text. Empty when the text has nothing to speak.
  std::vector<int64_t> Tokenize(std::string_view text) const;

  /// Synthesize one chunk of text. `cancelled` is checked before each speech frame; when it returns true, synthesis
  /// stops and the result is empty. The noise of the diffusion head comes from a fixed seed, so the same text and
  /// voice give the same audio on one platform.
  SynthesizedSpeech Synthesize(std::string_view text, const SpeechSynthesisVoice& voice,
                               const std::function<bool()>& cancelled = {}) const;

 private:
  friend class ModelLoadManager;

  SpeechSynthesisModelInstance(std::string model_id, std::string model_path, SpeechSynthesisConfig config,
                               ExecutionProvider ep, ILogger& logger);

  struct Runtime;

  /// One speech latent: DPM-Solver++ over the diffusion head with classifier-free guidance.
  std::vector<float> SampleLatent(const std::vector<uint16_t>& condition,
                                  const std::vector<uint16_t>& negative_condition, std::mt19937& rng) const;

  SpeechSynthesisConfig config_;
  ILogger& logger_;
  std::unique_ptr<Runtime> runtime_;
  std::map<std::string, VoicePreset> voices_;
};

}  // namespace fl
