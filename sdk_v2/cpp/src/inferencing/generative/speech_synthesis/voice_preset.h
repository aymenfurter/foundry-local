// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace fl {

/// A float16 tensor: IEEE half-precision bits in row-major order.
struct HalfTensor {
  std::vector<int64_t> shape;
  std::vector<uint16_t> data;
};

/// The tensor shapes that the model graphs expect from a voice preset.
struct VoicePresetLayout {
  size_t text_lm_layers = 0;
  size_t tts_lm_layers = 0;
  int64_t kv_heads = 0;
  int64_t head_dim = 0;
  int64_t hidden_size = 0;
};

/// The cached prompt of one VibeVoice voice: the key/value caches of the text LM, the TTS LM and the negative
/// (unconditioned) TTS LM after the voice prompt, and the last hidden state of the negative TTS LM. A preset holds no
/// audio and cannot be made from audio with this SDK.
struct VoicePreset {
  std::vector<HalfTensor> text_lm_kv;          ///< key_0, value_0, key_1, value_1, ...; each [1, heads, seq, dim].
  std::vector<HalfTensor> tts_lm_kv;           ///< Same order.
  std::vector<HalfTensor> negative_tts_lm_kv;  ///< Same order.
  std::vector<uint16_t> negative_condition;    ///< [hidden_size].

  size_t ByteSize() const;
};

/// Load a voice preset from a NumPy .npz archive (stored or deflated) with the arrays lm_key_<i>, lm_value_<i>,
/// tts_lm_key_<i>, tts_lm_value_<i>, neg_tts_lm_key_<i>, neg_tts_lm_value_<i> and neg_tts_lm_hidden, of type float16
/// or float32. Other arrays are ignored.
/// @throws fl::Exception(INVALID_ARGUMENT) when the file is larger than `max_bytes`, is not a valid archive, or an
/// array is missing or does not match `layout`.
VoicePreset LoadVoicePreset(const std::filesystem::path& file, const VoicePresetLayout& layout, uint64_t max_bytes);

}  // namespace fl
