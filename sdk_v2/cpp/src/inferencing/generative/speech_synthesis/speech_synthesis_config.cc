// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "inferencing/generative/speech_synthesis/speech_synthesis_config.h"

#include "exception.h"
#include "inferencing/model_package.h"
#include "util/file_utils.h"
#include "util/path_safety.h"

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <optional>

namespace fl {

namespace {

constexpr int kMaxWindowSize = 64;
constexpr size_t kMaxDiffusionSteps = 100;

[[noreturn]] void Invalid(const std::string& message) {
  FL_THROW(FOUNDRY_LOCAL_ERROR_INVALID_ARGUMENT, fmt::format("tts_config.json: {}", message));
}

/// A string field. Missing or null gives an empty string, or an error when `required`.
std::string String(const nlohmann::json& object, const char* key, bool required = false) {
  const auto it = object.find(key);
  if (it == object.end() || it->is_null()) {
    if (required) {
      Invalid(fmt::format("'{}' is required", key));
    }

    return {};
  }

  if (!it->is_string()) {
    Invalid(fmt::format("'{}' must be a string", key));
  }

  return it->get<std::string>();
}

/// A finite number field, or nullopt when it is missing and not `required`.
std::optional<double> Number(const nlohmann::json& object, const char* key, bool required = false) {
  const auto it = object.find(key);
  if (it == object.end() || it->is_null()) {
    if (required) {
      Invalid(fmt::format("'{}' is required", key));
    }

    return std::nullopt;
  }

  if (!it->is_number() || !std::isfinite(it->get<double>())) {
    Invalid(fmt::format("'{}' must be a finite number", key));
  }

  return it->get<double>();
}

int WindowSize(const nlohmann::json& object, const char* key, int fallback) {
  const auto it = object.find(key);
  if (it == object.end()) {
    return fallback;
  }

  if (!it->is_number_integer() || it->get<int64_t>() < 1 || it->get<int64_t>() > kMaxWindowSize) {
    Invalid(fmt::format("'{}' must be an integer from 1 to {}", key, kMaxWindowSize));
  }

  return it->get<int>();
}

/// Resolve a manifest path. Empty and absolute paths and paths that leave the package directory are rejected.
std::filesystem::path PackagePath(const std::filesystem::path& package_dir, const std::string& relative) {
  const std::filesystem::path value(relative);
  const auto resolved = (package_dir / value).lexically_normal();
  if (relative.empty() || value.has_root_name() || value.has_root_directory() ||
      !IsPathWithinDirectory(resolved, package_dir)) {
    Invalid(fmt::format("path must stay inside the package: '{}'", relative));
  }

  return resolved;
}

DiffusionSchedule ParseDiffusion(const nlohmann::json& root) {
  const auto it = root.find("diffusion");
  if (it == root.end() || !it->is_object()) {
    Invalid("'diffusion' must be an object with 'timesteps' and 'sigmas'");
  }

  // find(), not const operator[], which asserts that the key exists.
  const auto timesteps_it = it->find("timesteps");
  const auto sigmas_it = it->find("sigmas");
  const auto has_arrays = timesteps_it != it->end() && sigmas_it != it->end();
  static const nlohmann::json kMissing;
  const auto& timesteps = has_arrays ? *timesteps_it : kMissing;
  const auto& sigmas = has_arrays ? *sigmas_it : kMissing;
  if (!timesteps.is_array() || timesteps.empty() || timesteps.size() > kMaxDiffusionSteps || !sigmas.is_array() ||
      sigmas.size() != timesteps.size() + 1) {
    Invalid(fmt::format("'diffusion' needs 1 to {} timesteps and one sigma more than timesteps", kMaxDiffusionSteps));
  }

  DiffusionSchedule schedule;
  for (const auto& value : timesteps) {
    if (!value.is_number() || !std::isfinite(value.get<double>()) || value.get<double>() < 0) {
      Invalid("diffusion timesteps must be numbers >= 0");
    }

    schedule.timesteps.push_back(value.get<float>());
  }

  for (const auto& value : sigmas) {
    if (!value.is_number() || !std::isfinite(value.get<double>())) {
      Invalid("diffusion sigmas must be finite numbers");
    }

    schedule.sigmas.push_back(value.get<double>());
  }

  // The solver divides by every sigma but the last, and needs a strictly decreasing noise level.
  for (size_t i = 0; i + 1 < schedule.sigmas.size(); ++i) {
    if (schedule.sigmas[i] <= 0 || schedule.sigmas[i + 1] < 0 || schedule.sigmas[i + 1] >= schedule.sigmas[i]) {
      Invalid("diffusion sigmas must decrease strictly, be > 0 and end at a value >= 0");
    }
  }

  return schedule;
}

}  // namespace

SpeechSynthesisConfig SpeechSynthesisConfig::LoadFromDirectory(const std::filesystem::path& package_dir) {
  const auto file = package_dir / kSpeechSynthesisConfigFileName;
  const auto text = ReadFileToString(file);
  if (!text) {
    FL_THROW(FOUNDRY_LOCAL_ERROR_INVALID_ARGUMENT, fmt::format("cannot read {}", file.string()));
  }

  return Parse(*text, package_dir);
}

SpeechSynthesisConfig SpeechSynthesisConfig::Parse(const std::string& json_text,
                                                   const std::filesystem::path& package_dir) {
  nlohmann::json root;
  try {
    root = nlohmann::json::parse(json_text);
  } catch (const nlohmann::json::exception& ex) {
    FL_THROW(FOUNDRY_LOCAL_ERROR_INVALID_ARGUMENT, fmt::format("invalid tts_config.json: {}", ex.what()));
  }

  if (!root.is_object() || String(root, "model_type", true) != "vibevoice") {
    FL_THROW(FOUNDRY_LOCAL_ERROR_INVALID_ARGUMENT, "tts_config.json must be an object with model_type 'vibevoice'");
  }

  const auto dir = std::filesystem::absolute(package_dir).lexically_normal();
  const auto graphs = root.find("graphs");
  if (graphs == root.end() || !graphs->is_object()) {
    Invalid("'graphs' must be an object");
  }

  SpeechSynthesisConfig config;
  config.text_lm_model = PackagePath(dir, String(*graphs, "text_lm", true));
  config.tts_lm_model = PackagePath(dir, String(*graphs, "tts_lm", true));
  config.diffusion_head_model = PackagePath(dir, String(*graphs, "diffusion_head", true));
  config.acoustic_connector_model = PackagePath(dir, String(*graphs, "acoustic_connector", true));
  config.vocoder_model = PackagePath(dir, String(*graphs, "vocoder", true));
  config.tokenizer_dir = PackagePath(dir, String(root, "tokenizer", true));

  config.speech_scaling_factor = static_cast<float>(*Number(root, "speech_scaling_factor", true));
  config.speech_bias_factor = static_cast<float>(*Number(root, "speech_bias_factor", true));
  if (config.speech_scaling_factor == 0.0f) {
    Invalid("'speech_scaling_factor' must not be 0");
  }

  config.text_window_size = WindowSize(root, "text_window_size", config.text_window_size);
  config.speech_window_size = WindowSize(root, "speech_window_size", config.speech_window_size);
  config.cfg_scale = static_cast<float>(Number(root, "cfg_scale").value_or(config.cfg_scale));
  if (config.cfg_scale < 0.0f || config.cfg_scale > 20.0f) {
    Invalid("'cfg_scale' must be from 0 to 20");
  }

  config.diffusion = ParseDiffusion(root);

  const auto voices = root.find("voices");
  if (voices == root.end() || !voices->is_array() || voices->empty()) {
    Invalid("'voices' must be a non-empty array");
  }

  for (const auto& value : *voices) {
    if (!value.is_object()) {
      Invalid("voices must be objects");
    }

    SpeechSynthesisVoice voice{String(value, "name", true), PackagePath(dir, String(value, "file", true)),
                               String(value, "language", true), String(value, "gender")};
    if (voice.name.empty() || voice.language.empty() || config.FindVoice(voice.name)) {
      Invalid(fmt::format("voice '{}' must have a unique name and a language", voice.name));
    }

    config.voices.push_back(std::move(voice));
  }

  config.default_voice = String(root, "default_voice");
  config.default_voice = config.default_voice.empty() ? config.voices.front().name : config.default_voice;
  if (!config.FindVoice(config.default_voice)) {
    Invalid(fmt::format("default_voice '{}' is not in voices", config.default_voice));
  }

  return config;
}

const SpeechSynthesisVoice* SpeechSynthesisConfig::FindVoice(const std::string& name) const {
  const auto it = std::find_if(voices.begin(), voices.end(), [&](const auto& voice) { return voice.name == name; });
  return it == voices.end() ? nullptr : &*it;
}

}  // namespace fl
