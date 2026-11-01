// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "inferencing/execution_provider.h"
#include "inferencing/generative/genai_model_instance.h"
#include "inferencing/loaded_model_instance.h"
#include "logger.h"
#include "ep_detection/ep_detector.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace fl {

/// Manages the lifecycle of loaded models: ORT GenAI packages (genai_config.json) and speech-synthesis packages
/// (tts_config.json). Thread-safe. Owns all LoadedModelInstance instances.
///
/// Returned instance pointers are non-owning and valid only while the
/// model remains loaded. The caller must ensure the model is not unloaded (via
/// UnloadModel) while the pointer is in use. Session factories use AcquireSessionModel
/// to acquire their construction-time reference atomically with lookup.
class ModelLoadManager {
 public:
  enum class LoadStatus {
    kSuccess,
    kModelNotFound,
    kModelAlreadyLoaded,
  };

  struct LoadResult {
    LoadStatus status;
    LoadedModelInstance* instance = nullptr;  // non-owning; nullptr when the model was not found

    /// The loaded instance as `T`, or nullptr when it is absent or of another package kind.
    template <class T>
    T* As() const noexcept {
      return instance ? instance->As<T>() : nullptr;
    }
  };

  ModelLoadManager(IEpDetector& ep_detector, ILogger& logger);
  ~ModelLoadManager();

  ModelLoadManager(const ModelLoadManager&) = delete;
  ModelLoadManager& operator=(const ModelLoadManager&) = delete;

  /// Load a model from the given path. The package manifest selects the runtime: genai_config.json loads with
  /// ORT GenAI and tts_config.json loads a speech-synthesis model with a plain ORT session.
  /// @param model_path  Path to the model directory (must contain genai_config.json or tts_config.json).
  /// @param model_id    Unique identifier for the model.
  /// @param ep_override Execution provider override (kDefault = use the manifest default,
  ///                    or auto-select CUDA for generic-gpu GenAI models if available).
  /// @returns LoadResult with status and non-owning pointer to the loaded model.
  LoadResult LoadModel(std::string_view model_path,
                       std::string_view model_id,
                       ExecutionProvider ep_override = ExecutionProvider::kDefault);

  /// Unload a previously loaded model.
  /// @returns true if the model was found and unloaded; false if the model was not loaded
  ///          (idempotent no-op).
  /// @throws  FoundryLocalException(INVALID_USAGE) if the model still has live sessions.
  ///          Drop the sessions before unloading — using a model while unloading it is a
  ///          contract violation.
  bool UnloadModel(std::string_view model_id);

  /// Unload only when the loaded model's normalized path matches `model_path`.
  /// Returns false when the ID is absent or belongs to a different path.
  bool UnloadModel(std::string_view model_id, std::string_view model_path);

  /// Get a loaded GenAI model by ID. Returns nullptr if not loaded or if the loaded package is not a GenAI package.
  /// The returned pointer is valid until UnloadModel is called for this model_id.
  GenAIModelInstance* GetLoadedModel(std::string_view model_id);

  /// Get a loaded GenAI model for an immediate query only when its normalized path matches `model_path`.
  GenAIModelInstance* GetLoadedModel(std::string_view model_id, std::string_view model_path);

  /// Get a loaded model of any package kind for an immediate query, qualified by its normalized path.
  LoadedModelInstance* GetLoadedInstance(std::string_view model_id, std::string_view model_path);

  /// True when a model of any package kind is loaded with this ID from this path.
  bool IsModelLoaded(std::string_view model_id, std::string_view model_path);

  struct SessionReferenceDeleter {
    void operator()(LoadedModelInstance* model) const noexcept { model->ReleaseSession(); }
  };
  using SessionModelReference = std::unique_ptr<LoadedModelInstance, SessionReferenceDeleter>;

  /// Acquire the construction-time session reference under the lifecycle lock. The session factory retains it until
  /// the constructed session has acquired its own reference, or releases it on construction failure.
  SessionModelReference AcquireSessionModel(std::string_view model_id, std::string_view model_path);

  struct LoadedModelState {
    std::optional<GenAIConfig> config;  // nullopt for packages that are not GenAI packages
    std::string model_path;
    std::optional<uint64_t> context_length;
  };

  /// Copy loaded configuration, actual path, and effective context under the lock, qualified by ID and artifact path.
  /// Engine limits come from its already-initialized native capabilities, without creating a request or session.
  std::optional<LoadedModelState> GetLoadedModelState(std::string_view model_id, std::string_view model_path);

  /// Reject operations on a directory containing any loaded model, regardless of ID or revision.
  /// Holds the lifecycle mutex through the operation so another handle cannot load files being removed.
  /// The operation must not call back into this ModelLoadManager.
  void RunWithUnloadedModelsInDirectory(std::string_view directory, const std::function<void()>& operation);

  /// Reject all future LoadModel calls. Called by Manager::Shutdown().
  /// Idempotent and thread-safe.
  void RejectNewLoads();

  /// Unload every loaded model, waiting for live sessions to finish.
  /// Called by Manager::Shutdown after the session manager has cancelled HTTP-tracked
  /// sessions; this covers direct-API sessions (which only register via the per-model
  /// session refcount on GenAIModelInstance).
  ///
  /// The timeout is an *overall* deadline shared across all models — total drain time is
  /// bounded by `timeout` regardless of how many models are loaded. Models whose sessions
  /// have not been released by the deadline are left loaded and a warning is logged; we
  /// do not block process shutdown indefinitely on a stuck caller.
  void UnloadAll(std::chrono::milliseconds timeout = std::chrono::seconds(10));

 private:
  bool HasEP(const std::string& ep_name) const;
  void EnsureEpReadyLocked(const std::string& required_ep, const std::string& model_id);
  std::unique_ptr<LoadedModelInstance> LoadGenAIModelLocked(const std::string& path, const std::string& model_id,
                                                            GenAIConfig config, ExecutionProvider ep_override);
  std::unique_ptr<LoadedModelInstance> LoadSpeechSynthesisModelLocked(const std::string& path,
                                                                     const std::string& model_id,
                                                                     SpeechSynthesisConfig config,
                                                                     ExecutionProvider ep_override);
  bool UnloadModel(std::string_view model_id, const std::string* normalized_model_path);
  LoadedModelInstance* GetLoadedInstance(std::string_view model_id, const std::string* normalized_model_path);
  LoadedModelInstance* FindLoadedModelLocked(std::string_view model_id, const std::string* normalized_model_path);

  IEpDetector& ep_detector_;
  ILogger& logger_;
  std::atomic<bool> shutdown_{false};
  mutable std::mutex mutex_;
  std::map<std::string, std::unique_ptr<LoadedModelInstance>> loaded_models_;
};

}  // namespace fl
