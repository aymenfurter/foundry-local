// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "inferencing/generative/speech_synthesis/speech_synthesis_session.h"

#include "exception.h"
#include "inferencing/execution_provider.h"
#include "inferencing/generative/audio/pcm_utils.h"
#include "inferencing/generative/speech_synthesis/speech_synthesis_model_instance.h"
#include "inferencing/generative/speech_synthesis/speech_text.h"
#include "items/audio_item.h"
#include "items/text_item.h"
#include "util/string_utils.h"
#include "util/utf8.h"

#include <fmt/format.h>

#include <cmath>
#include <cstdlib>

namespace fl {

namespace {

/// Input limit of one request, in characters. The audio of a request stays in memory until it ends (about 2.9 MB per
/// minute as PCM16 plus float samples), and 16384 characters are about 15 to 20 minutes of speech.
constexpr size_t kMaxInputCharacters = 16384;

/// Chunk sizes, in characters. VibeVoice speaks a very short text (one to three words) unreliably, so short sentences
/// are joined. A long chunk delays its first audio: the CPU makes speech about three times faster than real time.
constexpr size_t kMinChunkCharacters = 24;
constexpr size_t kMaxChunkCharacters = 400;

std::unique_ptr<AudioItem> MakeAudioItem(const std::vector<float>& samples) {
  auto item = std::make_unique<AudioItem>(ConvertFloatToS16LE(samples), "pcm");
  item->sample_rate = SpeechSynthesisModelInstance::kSampleRate;
  item->channels = 1;
  return item;
}

}  // namespace

SpeechSynthesisSession::SpeechSynthesisSession(const fl::Model& catalog_model, SpeechSynthesisModelInstance& model,
                                               ILogger& logger, ITelemetry& telemetry)
    : Session(catalog_model, logger, telemetry), model_(model) {
  // Last so a throw above does not leak a refcount; nothing below can throw.
  model_.AcquireSession();
}

SpeechSynthesisSession::~SpeechSynthesisSession() {
  model_.ReleaseSession();
}

SessionType SpeechSynthesisSession::Type() const {
  return SessionType::kSpeechSynthesis;
}

std::string SpeechSynthesisSession::ExecutionProvider() const {
  return std::string(EPUtils::EPtoTelemetryName(model_.EP(), ""));
}

void SpeechSynthesisSession::ProcessRequestImpl(const Request& request, Response& response) {
  if (request.items.empty()) {
    FL_THROW(FOUNDRY_LOCAL_ERROR_INVALID_ARGUMENT, "speech synthesis request has no input items");
  }

  std::string input;
  for (const auto* item : request.items) {
    if (!item || item->type != FOUNDRY_LOCAL_ITEM_TEXT) {
      FL_THROW(FOUNDRY_LOCAL_ERROR_INVALID_ARGUMENT, "speech synthesis expects TEXT items");
    }

    input += (input.empty() ? "" : "\n") + static_cast<const TextItem&>(*item).text;
  }

  if (!IsValidUtf8(input) || CountCodePoints(input) > kMaxInputCharacters) {
    FL_THROW(FOUNDRY_LOCAL_ERROR_INVALID_ARGUMENT,
             fmt::format("speech input must be valid UTF-8 of at most {} characters", kMaxInputCharacters));
  }

  const auto options = MergedOptions(request.options);
  const char* voice_name = options.Find(FOUNDRY_LOCAL_PARAM_VOICE);
  const auto& voice = model_.ResolveVoice(voice_name ? voice_name : "");
  if (const char* value = options.Find(FOUNDRY_LOCAL_PARAM_SPEED)) {
    char* end = nullptr;
    const double parsed = std::strtod(value, &end);
    if (end == value || *end != '\0' || !std::isfinite(parsed) || parsed != 1.0) {
      FL_THROW(FOUNDRY_LOCAL_ERROR_INVALID_ARGUMENT,
               fmt::format("this speech synthesis model supports only speed 1, got '{}'", value));
    }
  }

  // Each chunk is synthesized on its own, so that a long request starts streaming early and observes cancellation
  // between chunks and between speech frames.
  auto callback = CreateCallbackHandler(request);
  const auto cancelled = [&request] { return request.IsCancellationRequested(); };
  std::vector<float> samples;
  int64_t text_tokens = 0;
  int64_t speech_frames = 0;
  for (const auto& chunk : SplitSpeechText(input, kMinChunkCharacters, kMaxChunkCharacters)) {
    if (cancelled()) {
      break;
    }

    const auto speech = model_.Synthesize(chunk, voice, cancelled);
    text_tokens += static_cast<int64_t>(speech.text_tokens);
    speech_frames += static_cast<int64_t>(speech.speech_frames);
    if (callback && !speech.samples.empty()) {
      auto item = MakeAudioItem(speech.samples);
      item->GetMetadata().Add("text", chunk);
      callback->PushItem(std::move(item));
    }

    samples.insert(samples.end(), speech.samples.begin(), speech.samples.end());
  }

  response.items.push_back(MakeAudioItem(samples));
  response.items.back()->GetMetadata().Add("voice", voice.name);
  response.usage.prompt_tokens = text_tokens;
  response.usage.completion_tokens = speech_frames;
  response.usage.total_tokens = text_tokens + speech_frames;
  response.finish_reason = FOUNDRY_LOCAL_FINISH_STOP;
}

}  // namespace fl
