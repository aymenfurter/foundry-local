// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace fl {

/// Decode UTF-8 to code points. Lenient: a lead byte without enough following bytes, a stray continuation byte or an
/// invalid lead byte decodes as one code point with the byte's value. Use IsValidUtf8 (util/string_utils.h) to reject
/// malformed input.
std::u32string DecodeUtf8(std::string_view utf8);

/// Encode code points as UTF-8.
std::string EncodeUtf8(std::u32string_view code_points);

/// Count the code points of UTF-8 text: every byte that is not a continuation byte (10xxxxxx) starts one.
size_t CountCodePoints(std::string_view utf8);

}  // namespace fl
