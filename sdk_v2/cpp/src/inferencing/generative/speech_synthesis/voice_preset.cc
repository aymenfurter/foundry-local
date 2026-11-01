// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "inferencing/generative/speech_synthesis/voice_preset.h"

#include "exception.h"

#include <archive.h>
#include <archive_entry.h>
#include <fmt/format.h>
#include <onnxruntime_cxx_api.h>

#include <array>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>

namespace fl {

namespace {

struct ArchiveDeleter {
  void operator()(archive* value) const { archive_read_free(value); }
};

using ArchivePtr = std::unique_ptr<archive, ArchiveDeleter>;

/// One array of an .npz archive: the .npy dtype descriptor, the shape and the raw little-endian data.
struct NpyArray {
  std::string descr;
  std::vector<int64_t> shape;
  std::string data;
};

[[noreturn]] void Invalid(const std::filesystem::path& file, const std::string& message) {
  FL_THROW(FOUNDRY_LOCAL_ERROR_INVALID_ARGUMENT, fmt::format("voice preset {}: {}", file.string(), message));
}

/// The text after `'key':` in an .npy header dictionary, or an empty view when the key is missing.
std::string_view HeaderValue(std::string_view header, std::string_view key) {
  const auto at = header.find(fmt::format("'{}':", key));
  if (at == std::string_view::npos) {
    return {};
  }

  auto value = header.substr(at + key.size() + 3);
  while (!value.empty() && value.front() == ' ') {
    value.remove_prefix(1);
  }

  return value;
}

NpyArray ParseNpy(const std::string& bytes, const std::string& name, const std::filesystem::path& file) {
  // Format: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
  constexpr std::string_view kMagic("\x93NUMPY", 6);
  const auto* raw = reinterpret_cast<const unsigned char*>(bytes.data());
  if (bytes.size() < 10 || std::string_view(bytes).substr(0, kMagic.size()) != kMagic) {
    Invalid(file, fmt::format("array '{}' is not in .npy format", name));
  }

  size_t header_size = static_cast<size_t>(raw[8]) | (static_cast<size_t>(raw[9]) << 8);
  size_t offset = 10;
  if (raw[6] == 2 || raw[6] == 3) {
    if (bytes.size() < 12) {
      Invalid(file, fmt::format("array '{}' has a truncated header", name));
    }

    header_size |= (static_cast<size_t>(raw[10]) << 16) | (static_cast<size_t>(raw[11]) << 24);
    offset = 12;
  } else if (raw[6] != 1) {
    Invalid(file, fmt::format("array '{}' has unsupported .npy version {}", name, raw[6]));
  }

  if (header_size > bytes.size() - offset) {
    Invalid(file, fmt::format("array '{}' has a truncated header", name));
  }

  const std::string_view header(bytes.data() + offset, header_size);
  NpyArray array;
  const auto descr = HeaderValue(header, "descr");
  if (descr.size() < 2 || descr.front() != '\'' || descr.find('\'', 1) == std::string_view::npos) {
    Invalid(file, fmt::format("array '{}' has no dtype", name));
  }

  array.descr = std::string(descr.substr(1, descr.find('\'', 1) - 1));
  if (HeaderValue(header, "fortran_order").substr(0, 5) != "False") {
    Invalid(file, fmt::format("array '{}' must be in C order", name));
  }

  const auto shape = HeaderValue(header, "shape");
  const auto close = shape.find(')');
  if (shape.empty() || shape.front() != '(' || close == std::string_view::npos) {
    Invalid(file, fmt::format("array '{}' has no shape", name));
  }

  uint64_t count = 1;
  for (size_t at = 1; at < close;) {
    while (at < close && (shape[at] == ' ' || shape[at] == ',')) {
      ++at;
    }

    if (at == close) {
      break;
    }

    int64_t dim = 0;
    const auto start = at;
    while (at < close && shape[at] >= '0' && shape[at] <= '9' && dim <= std::numeric_limits<int32_t>::max()) {
      dim = dim * 10 + (shape[at++] - '0');
    }

    if (at == start || dim > std::numeric_limits<int32_t>::max() ||
        (at < close && shape[at] != ',' && shape[at] != ' ') ||
        (dim != 0 && count > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dim))) {
      Invalid(file, fmt::format("array '{}' has an invalid shape", name));
    }

    array.shape.push_back(dim);
    count *= static_cast<uint64_t>(dim);
  }

  const size_t item_size = array.descr == "<f2" ? 2 : array.descr == "<f4" ? 4 : 0;
  if (item_size == 0) {
    Invalid(file, fmt::format("array '{}' has dtype '{}'; float16 or float32 is required", name, array.descr));
  }

  const auto data_offset = offset + header_size;
  if (count > (bytes.size() - data_offset) / item_size || count * item_size != bytes.size() - data_offset) {
    Invalid(file, fmt::format("array '{}' does not hold the data of its shape", name));
  }

  array.data = bytes.substr(data_offset);
  return array;
}

/// Read every .npy array of an .npz archive. `max_bytes` limits the archive and its total uncompressed size.
std::map<std::string, NpyArray> ReadNpz(const std::filesystem::path& file, uint64_t max_bytes) {
  // Opening a FIFO or device can block, so only regular files are read.
  std::error_code ec;
  if (!std::filesystem::is_regular_file(file, ec) || std::filesystem::file_size(file, ec) > max_bytes || ec) {
    Invalid(file, fmt::format("must be a regular file of at most {} bytes", max_bytes));
  }

  ArchivePtr reader(archive_read_new());
  if (!reader) {
    Invalid(file, "cannot create an archive reader");
  }

  archive_read_support_format_zip(reader.get());
#ifdef _WIN32
  const auto opened = archive_read_open_filename_w(reader.get(), file.c_str(), 64 * 1024);
#else
  const auto opened = archive_read_open_filename(reader.get(), file.c_str(), 64 * 1024);
#endif
  if (opened != ARCHIVE_OK) {
    Invalid(file, fmt::format("not a zip archive: {}", archive_error_string(reader.get())));
  }

  std::map<std::string, NpyArray> arrays;
  uint64_t total = 0;
  archive_entry* entry = nullptr;
  int result = ARCHIVE_OK;
  while ((result = archive_read_next_header(reader.get(), &entry)) == ARCHIVE_OK) {
    const char* raw_name = archive_entry_pathname_utf8(entry);
    std::string name = raw_name ? raw_name : "";
    if (name.size() <= 4 || name.compare(name.size() - 4, 4, ".npy") != 0) {
      archive_read_data_skip(reader.get());
      continue;
    }

    name.resize(name.size() - 4);
    std::string bytes;
    std::array<char, 64 * 1024> buffer{};
    la_ssize_t count = 0;
    while ((count = archive_read_data(reader.get(), buffer.data(), buffer.size())) > 0) {
      total += static_cast<uint64_t>(count);
      if (total > max_bytes) {
        Invalid(file, fmt::format("arrays are larger than {} bytes", max_bytes));
      }

      bytes.append(buffer.data(), static_cast<size_t>(count));
    }

    if (count < 0) {
      Invalid(file, fmt::format("cannot read '{}': {}", name, archive_error_string(reader.get())));
    }

    auto array = ParseNpy(bytes, name, file);
    if (!arrays.emplace(name, std::move(array)).second) {
      Invalid(file, fmt::format("array '{}' appears twice", name));
    }
  }

  if (result != ARCHIVE_EOF) {
    Invalid(file, fmt::format("invalid zip archive: {}", archive_error_string(reader.get())));
  }

  return arrays;
}

HalfTensor ToHalf(NpyArray&& array) {
  HalfTensor tensor{std::move(array.shape), {}};
  if (array.descr == "<f2") {
    tensor.data.resize(array.data.size() / 2);
    std::memcpy(tensor.data.data(), array.data.data(), array.data.size());
    return tensor;
  }

  tensor.data.resize(array.data.size() / 4);
  for (size_t i = 0; i < tensor.data.size(); ++i) {
    float value = 0;
    std::memcpy(&value, array.data.data() + i * 4, 4);
    tensor.data[i] = Ort::Float16_t(value).val;
  }

  return tensor;
}

}  // namespace

size_t VoicePreset::ByteSize() const {
  size_t bytes = negative_condition.size() * 2;
  for (const auto* group : {&text_lm_kv, &tts_lm_kv, &negative_tts_lm_kv}) {
    for (const auto& tensor : *group) {
      bytes += tensor.data.size() * 2;
    }
  }

  return bytes;
}

VoicePreset LoadVoicePreset(const std::filesystem::path& file, const VoicePresetLayout& layout, uint64_t max_bytes) {
  auto arrays = ReadNpz(file, max_bytes);
  auto take = [&](const std::string& name) {
    const auto it = arrays.find(name);
    if (it == arrays.end()) {
      Invalid(file, fmt::format("array '{}' is missing", name));
    }

    return ToHalf(std::move(it->second));
  };

  auto matches = [](int64_t actual, int64_t expected) { return expected <= 0 || actual == expected; };

  // A cache is [1, kv_heads, sequence, head_dim] per layer, with one sequence length for all layers.
  auto load_cache = [&](const std::string& prefix, size_t layers) {
    std::vector<HalfTensor> cache;
    for (size_t layer = 0; layer < layers; ++layer) {
      for (const char* kind : {"key", "value"}) {
        auto tensor = take(fmt::format("{}_{}_{}", prefix, kind, layer));
        const auto& shape = tensor.shape;
        if (shape.size() != 4 || shape[0] != 1 || !matches(shape[1], layout.kv_heads) || shape[2] < 1 ||
            !matches(shape[3], layout.head_dim) || (!cache.empty() && shape[2] != cache.front().shape[2])) {
          Invalid(file, fmt::format("array '{}_{}_{}' must have shape [1, {}, sequence, {}]", prefix, kind, layer,
                                    layout.kv_heads, layout.head_dim));
        }

        cache.push_back(std::move(tensor));
      }
    }

    return cache;
  };

  VoicePreset preset;
  preset.text_lm_kv = load_cache("lm", layout.text_lm_layers);
  preset.tts_lm_kv = load_cache("tts_lm", layout.tts_lm_layers);
  preset.negative_tts_lm_kv = load_cache("neg_tts_lm", layout.tts_lm_layers);
  auto hidden = take("neg_tts_lm_hidden");
  if (hidden.shape.size() != 3 || hidden.shape[0] != 1 || hidden.shape[1] < 1 || hidden.shape[2] < 1 ||
      !matches(hidden.shape[2], layout.hidden_size)) {
    Invalid(file, fmt::format("array 'neg_tts_lm_hidden' must have shape [1, sequence, {}]", layout.hidden_size));
  }

  const auto width = static_cast<size_t>(hidden.shape[2]);
  preset.negative_condition.assign(hidden.data.end() - static_cast<std::ptrdiff_t>(width), hidden.data.end());
  return preset;
}

}  // namespace fl
