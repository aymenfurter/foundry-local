// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <filesystem>
#include <optional>
#include <string>

namespace fl {

/// Read a whole regular file in binary mode. Returns nullopt when the path is not a regular file or cannot be opened,
/// so the caller can report the error in its own terms.
std::optional<std::string> ReadFileToString(const std::filesystem::path& path);

}  // namespace fl
