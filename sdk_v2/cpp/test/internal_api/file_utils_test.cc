// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "util/file_utils.h"
#include "utils/temp_path.h"

#include <gtest/gtest.h>

#include <fstream>
#include <string>

#ifndef _WIN32
#include <sys/stat.h>
#endif

namespace fl {
namespace {

TEST(FileUtilsTest, ReadsRegularFilesOnly) {
  const auto dir = test::TempPath::CreateTempDir("file_utils");
  const std::string content("a\0b\r\nc", 6);
  std::ofstream(dir.path() / "data.bin", std::ios::binary) << content;
  EXPECT_EQ(ReadFileToString(dir.path() / "data.bin"), content);
  EXPECT_FALSE(ReadFileToString(dir.path() / "missing.txt").has_value());
  EXPECT_FALSE(ReadFileToString(dir.path()).has_value());
#ifndef _WIN32
  // Opening a FIFO without a writer blocks, so the reader must refuse it.
  ASSERT_EQ(mkfifo((dir.path() / "pipe").c_str(), 0600), 0);
  EXPECT_FALSE(ReadFileToString(dir.path() / "pipe").has_value());
#endif
}

}  // namespace
}  // namespace fl
