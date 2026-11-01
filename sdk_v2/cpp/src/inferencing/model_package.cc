// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "inferencing/model_package.h"

#include "exception.h"

#include <fmt/format.h>

#include <system_error>

namespace fl {

const char* ManifestFileName(ModelPackageKind kind) noexcept {
  return kind == ModelPackageKind::kSpeechSynthesis ? kSpeechSynthesisConfigFileName : kGenAIConfigFileName;
}

ModelPackageKind PackageKindForTask(std::string_view task) noexcept {
  return task == "text-to-speech" ? ModelPackageKind::kSpeechSynthesis : ModelPackageKind::kGenAI;
}

std::optional<ModelPackageKind> DetectModelPackageKind(const std::filesystem::path& dir) noexcept {
  for (const auto kind : {ModelPackageKind::kGenAI, ModelPackageKind::kSpeechSynthesis}) {
    std::error_code ec;
    if (std::filesystem::is_regular_file(dir / ManifestFileName(kind), ec)) {
      return kind;
    }
  }

  return std::nullopt;
}

ModelManifest LoadModelManifest(const std::filesystem::path& dir, ModelPackageKind kind) {
  const auto manifest = dir / ManifestFileName(kind);
  std::error_code ec;
  if (!std::filesystem::is_regular_file(manifest, ec)) {
    FL_THROW(FOUNDRY_LOCAL_ERROR_INVALID_ARGUMENT,
             fmt::format("model package must contain a regular {} file: {}", ManifestFileName(kind), dir.string()));
  }

  if (kind == ModelPackageKind::kSpeechSynthesis) {
    return SpeechSynthesisConfig::LoadFromDirectory(dir);
  }

  return GenAIConfig::LoadFromFile(manifest.string());
}

}  // namespace fl
