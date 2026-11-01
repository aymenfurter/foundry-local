// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "inferencing/generative/genai_config.h"
#include "inferencing/generative/speech_synthesis/speech_synthesis_config.h"

#include <filesystem>
#include <optional>
#include <string_view>
#include <variant>

namespace fl {

/// The runtime that serves a model package directory. This header is the single owner of the rules that map a task
/// to a package kind and a package directory to its manifest.
enum class ModelPackageKind {
  kGenAI,            ///< genai_config.json. ORT GenAI loads the model.
  kSpeechSynthesis,  ///< tts_config.json. A plain ORT session loads the model.
};

inline constexpr const char* kGenAIConfigFileName = "genai_config.json";
inline constexpr const char* kSpeechSynthesisConfigFileName = "tts_config.json";

/// The manifest file name of a package kind.
const char* ManifestFileName(ModelPackageKind kind) noexcept;

/// The package kind that serves a catalog task. "text-to-speech" needs a speech-synthesis package; all other tasks
/// need a GenAI package.
ModelPackageKind PackageKindForTask(std::string_view task) noexcept;

/// Detect the package kind from the manifest in `dir`. Returns nullopt when `dir` has no known manifest.
/// A directory that contains both manifests is a GenAI package.
std::optional<ModelPackageKind> DetectModelPackageKind(const std::filesystem::path& dir) noexcept;

/// True when `dir` contains a manifest of a supported model package kind.
inline bool IsModelPackageDirectory(const std::filesystem::path& dir) noexcept {
  return DetectModelPackageKind(dir).has_value();
}

/// A parsed package manifest. The alternative index matches ModelPackageKind.
using ModelManifest = std::variant<GenAIConfig, SpeechSynthesisConfig>;

/// Parse the manifest of a package of the given kind.
/// @throws fl::Exception(INVALID_ARGUMENT) when the manifest is missing or invalid.
ModelManifest LoadModelManifest(const std::filesystem::path& dir, ModelPackageKind kind);

}  // namespace fl
