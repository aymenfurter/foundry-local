// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "util/utf8.h"

#include <gtest/gtest.h>

#include <string>

namespace fl {
namespace {

TEST(Utf8Test, DecodesAndEncodesCodePoints) {
  const std::string text = "aé€😀";
  EXPECT_EQ(DecodeUtf8(text), (std::u32string{U'a', U'é', U'€', U'😀'}));
  EXPECT_EQ(EncodeUtf8(DecodeUtf8(text)), text);
  EXPECT_EQ(CountCodePoints(text), 4u);
}

TEST(Utf8Test, DecodesMalformedInputByteByByte) {
  // A truncated sequence and a stray continuation byte each decode as one code point with the byte value.
  EXPECT_EQ(DecodeUtf8("\xE2\x82"), (std::u32string{0xE2, 0x82}));
  EXPECT_EQ(DecodeUtf8("a\x80" "b"), (std::u32string{U'a', 0x80, U'b'}));
  EXPECT_TRUE(DecodeUtf8("").empty());
  EXPECT_EQ(CountCodePoints(""), 0u);
}

}  // namespace
}  // namespace fl
