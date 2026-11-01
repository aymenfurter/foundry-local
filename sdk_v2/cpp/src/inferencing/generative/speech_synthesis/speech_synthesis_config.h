// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace fl {

/// One voice of a speech-synthesis package: a VibeVoice voice preset (.npz) that holds the cached prompt of the voice.
struct SpeechSynthesisVoice {
  std::string name;
  std::filesystem::path file;
  std::string language;  ///< BCP 47 tag, for example "en".
  std::string gender;    ///< Optional: "female" or "male".
};

/// The DPM-Solver++ schedule of the diffusion head: one model timestep per step and one sigma more than steps. The
/// last sigma is the end of the trajectory (0 for a clean sample).
struct DiffusionSchedule {
  std::vector<float> timesteps;
  std::vector<double> sigmas;
};

/// Parsed tts_config.json of a VibeVoice-Realtime package. All paths are absolute and inside the package directory,
/// also after symbolic links are resolved.
///
/// {
///   "model_type": "vibevoice",
///   "graphs": {"text_lm": "text_lm_kv.onnx", "tts_lm": "tts_lm_kv.onnx", "diffusion_head": "diffusion_head.onnx",
///              "acoustic_connector": "acoustic_connector.onnx", "vocoder": "vocoder.onnx"},
///   "tokenizer": "tokenizer",
///   "speech_scaling_factor": 0.2333984375,
///   "speech_bias_factor": -0.0703125,
///   "text_window_size": 5,
///   "speech_window_size": 6,
///   "cfg_scale": 1.5,
///   "diffusion": {"timesteps": [999, 799, 599, 400, 200], "sigmas": [20291.3, 3.1037, 1.3908, 0.7403, 0.3375, 0]},
///   "default_voice": "Emma",
///   "voices": [{"name": "Emma", "file": "voices/en-Emma_woman.npz", "language": "en", "gender": "female"}]
/// }
struct SpeechSynthesisConfig {
  std::filesystem::path text_lm_model;
  std::filesystem::path tts_lm_model;
  std::filesystem::path diffusion_head_model;
  std::filesystem::path acoustic_connector_model;
  std::filesystem::path vocoder_model;
  std::filesystem::path tokenizer_dir;  ///< tokenizer.json, tokenizer_config.json and genai_config.json (ORT GenAI).
  float speech_scaling_factor = 1.0f;   ///< Acoustic latents are divided by this and then reduced by the bias.
  float speech_bias_factor = 0.0f;
  int text_window_size = 5;    ///< Text tokens fed to the language models per window.
  int speech_window_size = 6;  ///< Speech latents generated per window.
  float cfg_scale = 1.5f;      ///< Classifier-free guidance scale of the diffusion head.
  DiffusionSchedule diffusion;
  std::string default_voice;  ///< The first voice when the manifest does not name one.
  std::vector<SpeechSynthesisVoice> voices;

  /// Load and validate `<package_dir>/tts_config.json`. Does not check that the referenced files exist.
  /// @throws fl::Exception(INVALID_ARGUMENT) on invalid content.
  static SpeechSynthesisConfig LoadFromDirectory(const std::filesystem::path& package_dir);
  static SpeechSynthesisConfig Parse(const std::string& json_text, const std::filesystem::path& package_dir);

  const SpeechSynthesisVoice* FindVoice(const std::string& name) const;
};

}  // namespace fl
