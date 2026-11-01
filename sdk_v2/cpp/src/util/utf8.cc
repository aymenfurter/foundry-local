// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "util/utf8.h"

#include <algorithm>

namespace fl {

std::u32string DecodeUtf8(std::string_view utf8) {
  std::u32string out;
  out.reserve(utf8.size());
  auto byte = [&](size_t index) { return static_cast<unsigned char>(utf8[index]); };

  size_t i = 0;
  while (i < utf8.size()) {
    const auto b0 = byte(i);
    char32_t cp = b0;
    size_t length = 1;
    if (b0 >= 0xC0 && b0 < 0xE0 && i + 1 < utf8.size()) {
      cp = ((b0 & 0x1F) << 6) | (byte(i + 1) & 0x3F);
      length = 2;
    } else if (b0 >= 0xE0 && b0 < 0xF0 && i + 2 < utf8.size()) {
      cp = ((b0 & 0x0F) << 12) | ((byte(i + 1) & 0x3F) << 6) | (byte(i + 2) & 0x3F);
      length = 3;
    } else if (b0 >= 0xF0 && i + 3 < utf8.size()) {
      cp = ((b0 & 0x07) << 18) | ((byte(i + 1) & 0x3F) << 12) | ((byte(i + 2) & 0x3F) << 6) | (byte(i + 3) & 0x3F);
      length = 4;
    }

    out.push_back(cp);
    i += length;
  }

  return out;
}

std::string EncodeUtf8(std::u32string_view code_points) {
  std::string out;
  out.reserve(code_points.size());
  for (const auto cp : code_points) {
    if (cp < 0x80) {
      out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
      out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
      out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
      out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
  }

  return out;
}

size_t CountCodePoints(std::string_view utf8) {
  return static_cast<size_t>(
      std::count_if(utf8.begin(), utf8.end(), [](char c) { return (static_cast<unsigned char>(c) & 0xC0) != 0x80; }));
}

}  // namespace fl
