// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "inferencing/execution_provider.h"
#include "inferencing/model_package.h"

#include <atomic>
#include <string>
#include <utility>

namespace fl {

/// A model whose runtime resources are loaded. ModelLoadManager owns every instance and manages the shared lifecycle:
/// identity, artifact path, execution provider and the live-session reference count. Derived classes own the runtime
/// objects of one package kind.
class LoadedModelInstance {
 public:
  virtual ~LoadedModelInstance() = default;

  LoadedModelInstance(const LoadedModelInstance&) = delete;
  LoadedModelInstance& operator=(const LoadedModelInstance&) = delete;
  LoadedModelInstance(LoadedModelInstance&&) = delete;
  LoadedModelInstance& operator=(LoadedModelInstance&&) = delete;

  ModelPackageKind Kind() const { return kind_; }
  const std::string& ModelId() const { return model_id_; }
  const std::string& ModelPath() const { return model_path_; }
  ExecutionProvider EP() const { return ep_; }

  /// Typed view: `this` as `T` when the instance is of `T::kKind`, else nullptr.
  template <class T>
  T* As() noexcept {
    return kind_ == T::kKind ? static_cast<T*>(this) : nullptr;
  }

  template <class T>
  const T* As() const noexcept {
    return kind_ == T::kKind ? static_cast<const T*>(this) : nullptr;
  }

  /// Live-session reference counting. Sessions call AcquireSession() on construction and ReleaseSession() on
  /// destruction. ModelLoadManager::UnloadModel refuses to unload while the count is > 0 to prevent use-after-free of
  /// the runtime objects.
  void AcquireSession() { session_ref_count_.fetch_add(1, std::memory_order_acq_rel); }
  void ReleaseSession() { session_ref_count_.fetch_sub(1, std::memory_order_acq_rel); }
  int SessionRefCount() const { return session_ref_count_.load(std::memory_order_acquire); }

 protected:
  LoadedModelInstance(ModelPackageKind kind, std::string model_id, std::string model_path, ExecutionProvider ep)
      : model_id_(std::move(model_id)), model_path_(std::move(model_path)), ep_(ep), kind_(kind) {}

  const std::string model_id_;
  const std::string model_path_;
  const ExecutionProvider ep_;

 private:
  const ModelPackageKind kind_;
  mutable std::atomic<int> session_ref_count_{0};
};

}  // namespace fl
