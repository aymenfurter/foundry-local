# Speech synthesis (text to speech)

Foundry Local runs text-to-speech models with the task `text-to-speech`. The supported model family is
[VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) (MIT) by Microsoft Research, in
the ONNX FP16 export [nenad1002/microsoft-vibevoice-0.5B-onnx-fp16](https://huggingface.co/nenad1002/microsoft-vibevoice-0.5B-onnx-fp16)
(MIT, third party). Source code: [microsoft/VibeVoice](https://github.com/microsoft/VibeVoice). Technical report:
[arXiv:2508.19205](https://arxiv.org/abs/2508.19205).

## Architecture

- `inferencing/model_package.h` owns the package rules: `genai_config.json` is an ORT GenAI package, and
  `tts_config.json` is a speech-synthesis package. The load manager, cache scanner, download path resolution, local
  registration, `Session::Create` and `Model::IsCached` use it.
- `SpeechSynthesisModelInstance` (a `LoadedModelInstance`, like `GenAIModelInstance`) runs the five VibeVoice graphs
  in plain ONNX Runtime sessions. It loads the graphs, the tokenizer and all voice presets when the model loads. After
  that it is immutable, so sessions synthesize without a lock. The execution provider is the load override; the
  default is CPU.
- `SpeechSynthesisSession` uses the generic request/response API. There are no new C ABI functions, so the shared host
  and all bindings work without IPC changes.

## How VibeVoice generates speech

VibeVoice reads text tokens: the package tokenizer (Qwen2.5 byte-level BPE) runs through ORT GenAI
(`OgaCreateTokenizerFromPath`) with the `literal` option, so text such as `<|endoftext|>` cannot inject control
tokens. As in the VibeVoice processor, typographic quotes become straight quotes and the text ends with a line break.

| Graph | Inputs (float16 unless noted) | Outputs |
|---|---|---|
| `text_lm` (4-layer Qwen2) | `input_ids` int64 [1, n], `past_key/value_<i>` | `hidden_states` [1, n, 896], `present_*` |
| `tts_lm` (20-layer Qwen2) | `inputs_embeds` [1, n, 896], `tts_text_mask` int64 [1, n], cache | `hidden_states`, `eos_logits` [1, 1], cache |
| `diffusion_head` | `noisy_latent` [1, 64], `timestep` [1], `condition` [1, 896] | `predicted_noise` [1, 64] (v-prediction) |
| `acoustic_connector` | `latent` [1, 1, 64] | `embedding` [1, 1, 896] |
| `vocoder` | `latents` [1, F, 64] | `audio` [1, 3200 F], 24 kHz |

A voice preset holds the key/value caches of the text LM, the TTS LM and the negative (unconditioned) TTS LM after a
voice prompt. For each chunk of text:

1. Windows of 5 text tokens go through the text LM into the TTS LM (`tts_text_mask` 1).
2. After each window, the TTS LM makes 6 speech latents. Each latent is 5 steps of multistep DPM-Solver++
   (`dpm_solver.cc`) over the diffusion head with classifier-free guidance (scale 1.5) between the TTS LM and the
   negative TTS LM. The latent goes back into both LMs through the acoustic connector (`tts_text_mask` 0).
3. Generation ends when the EOS classifier fires after all text is fed, or after a frame budget (32 + 4 per token).
4. The vocoder decodes all latents of the chunk at once (7.5 frames per second of speech).

The noise comes from a fixed seed, so the same text and voice give the same audio on one platform. The VibeVoice
acoustic encoder is not part of the release, so voices cannot be made from audio; only the shipped presets exist.

## Contract

| | |
|---|---|
| Input | TEXT items. Each item, and each line in it, is a paragraph. At most 16384 characters in total. |
| Output | One AUDIO item: signed 16-bit little-endian PCM (`pcm`), mono, 24 kHz. Metadata `voice`. |
| Streaming | One AUDIO item per chunk. Metadata `text` is the chunk. A chunk is a sentence; a sentence shorter than 24 characters is joined with the next sentence of its line, and a sentence longer than 400 characters is split at clause punctuation, then between words. |
| Usage | `prompt_tokens` = text tokens, `completion_tokens` = speech frames. |
| Options | `voice` (`FOUNDRY_LOCAL_PARAM_VOICE`, a name from the manifest, default `default_voice`). `speed` (`FOUNDRY_LOCAL_PARAM_SPEED`) accepts only 1.0: VibeVoice has no rate control. |

## Quality, speed and memory

Measured on an Apple Silicon Mac, CPU execution provider, ONNX Runtime 1.31 (see the test notes below):

- Whisper (base.en) transcribed sentences of 5 to 20 words correctly for all seven English voices (apart from
  single-word confusions such as "bill" for "build"). Text of one to three words is unreliable with several voices
  (Carter read "OK, thanks." as "Have a good day.", Mike made no clear speech for "Hi."); Emma and Grace read it
  correctly, so Emma is the default. This is why the session joins short sentences.
- Synthesis runs at about 0.35 of real time: a sentence of 34 characters takes 0.8 s, 110 characters 2.1 s, 216
  characters 4.8 s and 339 characters 7.3 s. The vocoder decodes a chunk at the end, so the first audio of a chunk
  arrives when the whole chunk is ready. Loading takes about 3.5 s.
- The package is 1.7 GB. The process peaks at about 4.9 GB resident memory: full graph optimization converts the
  float16 weights for the CPU. With basic optimization the CPU converts them on every run instead, which is 3 to 5
  times slower and uses more memory at run time.

## Model package

A package is a directory. All manifest paths are relative to it and must stay inside it, also after symbolic links
are resolved.

```text
tts_config.json             manifest (below)
text_lm_kv.onnx             graphs of the ONNX FP16 export; .onnx.data files stay next to their graphs
tts_lm_kv.onnx
diffusion_head.onnx (+ .data)
acoustic_connector.onnx (+ .data)
vocoder.onnx (+ .data)
tokenizer/                  tokenizer.json, tokenizer_config.json and a genai_config.json for ORT GenAI
voices/<name>.npz           voice presets: NumPy .npz (stored or deflated), float16 or float32, at most 64 MiB each
LICENSE, NOTICE.md          VibeVoice MIT license and sources
```

ORT GenAI reads a tokenizer through a `genai_config.json`. Only its token ids are used:

```json
{"model": {"type": "qwen2", "bos_token_id": 151643, "eos_token_id": 151643, "pad_token_id": 151643,
           "vocab_size": 151936, "context_length": 8192}}
```

The manifest takes its numbers from `config.json` and `schedule.json` of the export:

```json
{
  "model_type": "vibevoice",
  "graphs": {"text_lm": "text_lm_kv.onnx", "tts_lm": "tts_lm_kv.onnx", "diffusion_head": "diffusion_head.onnx",
             "acoustic_connector": "acoustic_connector.onnx", "vocoder": "vocoder.onnx"},
  "tokenizer": "tokenizer",
  "speech_scaling_factor": 0.2333984375,
  "speech_bias_factor": -0.0703125,
  "text_window_size": 5,
  "speech_window_size": 6,
  "cfg_scale": 1.5,
  "diffusion": {"timesteps": [999, 799, 599, 400, 200],
                "sigmas": [20291.302734375, 3.103729724884033, 1.3907514810562134, 0.7402812242507935,
                           0.3374606668949127, 0.0]},
  "default_voice": "Emma",
  "voices": [
    {"name": "Emma", "file": "voices/en-Emma_woman.npz", "language": "en", "gender": "female"},
    {"name": "Grace", "file": "voices/en-Grace_woman.npz", "language": "en", "gender": "female"},
    {"name": "Carter", "file": "voices/en-Carter_man.npz", "language": "en", "gender": "male"},
    {"name": "Davis", "file": "voices/en-Davis_man.npz", "language": "en", "gender": "male"},
    {"name": "Frank", "file": "voices/en-Frank_man.npz", "language": "en", "gender": "male"},
    {"name": "Mike", "file": "voices/en-Mike_man.npz", "language": "en", "gender": "male"},
    {"name": "Samuel", "file": "voices/in-Samuel_man.npz", "language": "en-in", "gender": "male"}
  ]
}
```

`text_window_size`, `speech_window_size`, `cfg_scale` and `default_voice` are optional. The export also has presets
for nine other languages; the package leaves them out because the model card supports English only. The export's
model card lists a voice "Wayne", but neither the export nor microsoft/VibeVoice has it.

The converted files were checked against their sources: the graph inputs and outputs match the table above, and the
voice presets equal the BF16 presets in microsoft/VibeVoice `demo/voices/streaming_model` (largest difference 3e-8).
The export, the VibeVoice code and the model are MIT; the Qwen2.5 tokenizer is Apache-2.0.

A catalog entry needs `inferenceTasks: ["text-to-speech"]`, the package directory as the downloadable asset
(`tts_config.json` at the root or in one sub-directory, like `genai_config.json`), and `minFLVersion` set to the
first release with speech synthesis. To use a local package without a catalog, register it in the local catalog
with task `text-to-speech` (for example Rust `catalog.register_model(path, "vibevoice-realtime-0.5b:1", metadata)`).

## Tests

`test/internal_api/speech_synthesis/` and `sdk_v2/rust/tests/integration/speech_test.rs` use the deterministic
`test/testdata/tiny-vibevoice` package: five tiny float16 graphs with the VibeVoice signatures, a byte-level tokenizer
and two voices (no download). The DPM-Solver++ test compares against the NumPy solver of the export.
