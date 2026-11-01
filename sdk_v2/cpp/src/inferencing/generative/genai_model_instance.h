// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "inferencing/execution_provider.h"
#include "inferencing/generative/genai_config.h"
#include "inferencing/generative/preprocessor.h"
#include "inferencing/loaded_model_instance.h"
#include "logger.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// Forward declarations for ORT GenAI types (defined in ort_genai.h)
struct OgaModel;

namespace fl {

class OnnxChatEngine;

struct ModelCapabilities {
  bool native_qwen_xml_tool_calls = false;
  bool positional_tool_results = false;
  bool supports_reasoning_history = false;
  bool supports_preserve_thinking = false;
  bool supports_reasoning_controls = false;
};

namespace model_capabilities_internal {

ModelCapabilities ResolveRenderedProbes(std::string_view model_type,
                                        std::string_view tool_call_projection,
                                        std::string_view tool_result_projection,
                                        std::string_view reasoning_projection = {},
                                        std::string_view no_preserve_reasoning_projection = {},
                                        std::string_view reasoning_start = {},
                                        std::string_view reasoning_end = {}) noexcept;

}  // namespace model_capabilities_internal

/// A model that has been loaded into the ORT GenAI runtime.
/// Owns the OgaModel and its preprocessing resources.
/// Non-copyable, non-movable. Owned by ModelLoadManager via std::unique_ptr.
class GenAIModelInstance : public LoadedModelInstance {
 public:
  static constexpr ModelPackageKind kKind = ModelPackageKind::kGenAI;

  ~GenAIModelInstance() override;

  const GenAIConfig& GetGenAIConfig() const { return genai_config_; }
  ILogger& GetLogger() const { return logger_; }
  bool IsMultiModal() const;
  const std::string& ModelType() const { return model_type_; }
  bool HasNativeQwenXmlToolCalls() const { return capabilities_.native_qwen_xml_tool_calls; }
  bool HasPositionalToolResults() const { return capabilities_.positional_tool_results; }
  bool SupportsReasoningHistory() const { return capabilities_.supports_reasoning_history; }
  bool SupportsPreserveThinking() const { return capabilities_.supports_preserve_thinking; }
  bool SupportsReasoningControls() const { return capabilities_.supports_reasoning_controls; }

  /// Cached tag token IDs and decoded strings for tool/reasoning detection.
  /// Populated once on first access using OGA tokenizer APIs.
  struct TagInfo {
    std::optional<int32_t> bot_id;
    std::optional<int32_t> eot_id;
    std::optional<int32_t> bor_id;
    std::optional<int32_t> eor_id;
    std::string bot_str;
    std::string eot_str;
    std::string bor_str;
    std::string eor_str;
  };
  const TagInfo& GetTagInfo();

  /// Token IDs `text` encodes to with this model's tokenizer, or an empty sequence when it cannot be encoded.
  /// Encoding is best-effort by design: a marker the tokenizer cannot represent leaves callers matching decoded
  /// text instead of token IDs, which is the same fallback a model that publishes no marker IDs already uses.
  std::vector<int32_t> EncodeText(const std::string& text);

  /// Access the underlying OGA objects.
  OgaModel& GetOgaModel();
  Preprocessor& GetPreprocessor();
  OnnxChatEngine* GetChatEngine() { return chat_engine_.get(); }

  /// Get the last-activity timestamp.
  std::chrono::steady_clock::time_point LastActivity() const { return last_activity_; }

 private:
  friend class ModelLoadManager;

  GenAIModelInstance(std::string model_id,
                     std::string effective_model_path,
                     GenAIConfig genai_config,
                     ExecutionProvider resolved_ep,
                     ILogger& logger);

  GenAIConfig genai_config_;
  ILogger& logger_;
  std::unique_ptr<OgaModel> oga_model_;
  std::string model_type_;
  std::unique_ptr<Preprocessor> preprocessor_;
  ModelCapabilities capabilities_;
  std::unique_ptr<OnnxChatEngine> chat_engine_;
  TagInfo tag_info_;
  std::once_flag tag_info_init_flag_;
  std::chrono::steady_clock::time_point last_activity_;
};

}  // namespace fl
