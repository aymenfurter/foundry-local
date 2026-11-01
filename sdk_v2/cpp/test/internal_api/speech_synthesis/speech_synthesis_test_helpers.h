// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "exception.h"
#include "internal_api/test_model_cache.h"

#include <filesystem>
#include <functional>

namespace fl::test {

/// The tiny VibeVoice-shaped test package: five deterministic float16 graphs, a byte tokenizer and two voices.
inline std::filesystem::path TinyVibeVoicePath() {
  return GetTestDataPath("tiny-vibevoice");
}

/// The fl::Exception code that `action` throws, or FOUNDRY_LOCAL_OK.
inline flErrorCode ErrorCode(const std::function<void()>& action) {
  try {
    action();
  } catch (const Exception& ex) {
    return ex.code();
  }

  return FOUNDRY_LOCAL_OK;
}

}  // namespace fl::test
