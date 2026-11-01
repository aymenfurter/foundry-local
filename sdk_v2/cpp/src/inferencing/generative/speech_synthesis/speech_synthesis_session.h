// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "inferencing/session/session.h"

#include <string>

namespace fl {

class SpeechSynthesisModelInstance;

/// Text to speech. A request holds TEXT items; each item, and each line in it, is a paragraph. The response holds one
/// AUDIO item: signed 16-bit little-endian PCM ("pcm"), mono, 24 kHz. With a streaming callback, each chunk (a
/// sentence; short sentences are joined) also arrives as its own AUDIO item. Options: FOUNDRY_LOCAL_PARAM_VOICE, and
/// FOUNDRY_LOCAL_PARAM_SPEED, which VibeVoice supports only at 1.0.
class SpeechSynthesisSession : public Session {
 public:
  SpeechSynthesisSession(const fl::Model& catalog_model, SpeechSynthesisModelInstance& model, ILogger& logger,
                         ITelemetry& telemetry);
  ~SpeechSynthesisSession() override;

  SpeechSynthesisSession(SpeechSynthesisSession&&) = delete;
  SpeechSynthesisSession& operator=(SpeechSynthesisSession&&) = delete;

  SessionType Type() const override;

 protected:
  void ProcessRequestImpl(const Request& request, Response& response) override;

 private:
  std::string ExecutionProvider() const override;

  SpeechSynthesisModelInstance& model_;
};

}  // namespace fl
