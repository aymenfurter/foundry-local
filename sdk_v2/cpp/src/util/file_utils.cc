// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "util/file_utils.h"

#include <fstream>
#include <system_error>
#include <sstream>

namespace fl {

std::optional<std::string> ReadFileToString(const std::filesystem::path& path) {
  // Opening a FIFO or device can block or never end, so only regular files are read.
  std::error_code ec;
  if (!std::filesystem::is_regular_file(path, ec)) {
    return std::nullopt;
  }

  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    return std::nullopt;
  }

  std::ostringstream content;
  content << file.rdbuf();
  return content.str();
}

}  // namespace fl
