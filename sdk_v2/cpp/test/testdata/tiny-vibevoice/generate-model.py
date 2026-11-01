# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Regenerate tiny-vibevoice, a deterministic, untrained stand-in for a VibeVoice-Realtime package that speech synthesis
tests load on CPU without downloads. Run `python -m pip install -r requirements.txt && python generate-model.py`.

The five float16 graphs have the VibeVoice signatures with hidden size 4, latent size 2 and one layer per language
model, and compute simple functions that tests can follow:

- text_lm: hidden = 0.001 * token id in every channel. The tokenizer has no merges, so a token id is a UTF-8 byte.
- tts_lm: hidden = inputs_embeds + 0.1 * tts_text_mask; eos_logit = cache length + new length - 40.
- diffusion_head: predicted_noise = noisy_latent - sqrt(2) * condition[:2]. With the one-step schedule (sigma 1 to 0)
  the solver returns condition[:2], so a latent is negative + cfg_scale * (positive - negative) of the conditions.
- acoustic_connector: embedding = 0.5 * [latent, latent].
- vocoder: four samples per frame, [l0, l1, l0, l1] of the unscaled latent.

Voice Alpha (stored .npz) has a TTS prompt of 3 positions and negative condition 0.05; voice Beta (deflated .npz) has
5 positions and -0.05.
"""

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

ROOT = Path(__file__).resolve().parent
F16 = TensorProto.FLOAT16
HIDDEN = 4
LATENT = 2
EOS_LENGTH = 40


def const(name, values, dtype=F16):
    if dtype == F16:
        array = np.array(values, dtype=np.float16)
        return helper.make_tensor(name, F16, list(array.shape), array.tobytes(), raw=True)
    array = np.array(values, dtype=np.int64)
    return helper.make_tensor(name, TensorProto.INT64, list(array.shape), array.flatten().tolist())


def kv(prefix, dims):
    return [helper.make_tensor_value_info(f"{prefix}_{kind}_0", F16, dims) for kind in ("key", "value")]


def save(graph, name, initializers):
    graph.initializer.extend(initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="foundry-local-test")
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, ROOT / f"{name}.onnx")


def cache_nodes(hidden):
    """present = Concat(past, hidden[..., :2] as [1, 1, n, 2]) for key and value."""
    return [
        helper.make_node("Slice", [hidden, "zero", "two", "axis2"], ["new_kv_3d"]),
        helper.make_node("Unsqueeze", ["new_kv_3d", "axis1"], ["new_kv"]),
        helper.make_node("Concat", ["past_key_0", "new_kv"], ["present_key_0"], axis=2),
        helper.make_node("Concat", ["past_value_0", "new_kv"], ["present_value_0"], axis=2),
    ]


def cache_initializers():
    return slice_initializers() + [const("axis2", [2], TensorProto.INT64)]


def slice_initializers():
    return [const("zero", [0], TensorProto.INT64), const("two", [2], TensorProto.INT64),
            const("axis1", [1], TensorProto.INT64)]


def text_lm():
    nodes = [
        helper.make_node("Cast", ["input_ids"], ["ids"], to=F16),
        helper.make_node("Mul", ["ids", "scale"], ["scaled"]),
        helper.make_node("Unsqueeze", ["scaled", "axis2"], ["column"]),
        helper.make_node("Mul", ["column", "ones"], ["hidden_states"]),
    ] + cache_nodes("hidden_states")
    inputs = [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, "new_seq"])] + \
        kv("past", [1, 1, "past_seq", 2])
    outputs = [helper.make_tensor_value_info("hidden_states", F16, [1, "new_seq", HIDDEN])] + \
        kv("present", [1, 1, "total_seq", 2])
    graph = helper.make_graph(nodes, "tiny_text_lm", inputs, outputs)
    save(graph, "text_lm", [const("scale", 0.001), const("ones", np.ones((1, 1, HIDDEN)))] + cache_initializers())


def tts_lm():
    nodes = [
        helper.make_node("Cast", ["tts_text_mask"], ["mask"], to=F16),
        helper.make_node("Unsqueeze", ["mask", "axis2"], ["mask_column"]),
        helper.make_node("Mul", ["mask_column", "mask_scale"], ["mask_offset"]),
        helper.make_node("Add", ["inputs_embeds", "mask_offset"], ["hidden_states"]),
        helper.make_node("Shape", ["past_key_0"], ["past_length"], start=2, end=3),
        helper.make_node("Shape", ["inputs_embeds"], ["new_length"], start=1, end=2),
        helper.make_node("Add", ["past_length", "new_length"], ["length"]),
        helper.make_node("Cast", ["length"], ["length_half"], to=F16),
        helper.make_node("Sub", ["length_half", "eos_length"], ["eos_flat"]),
        helper.make_node("Reshape", ["eos_flat", "eos_shape"], ["eos_logits"]),
    ] + cache_nodes("hidden_states")
    inputs = [helper.make_tensor_value_info("inputs_embeds", F16, [1, "new_seq", HIDDEN]),
              helper.make_tensor_value_info("tts_text_mask", TensorProto.INT64, [1, "new_seq"])] + \
        kv("past", [1, 1, "past_seq", 2])
    outputs = [helper.make_tensor_value_info("hidden_states", F16, [1, "new_seq", HIDDEN]),
               helper.make_tensor_value_info("eos_logits", F16, [1, 1])] + kv("present", [1, 1, "total_seq", 2])
    graph = helper.make_graph(nodes, "tiny_tts_lm", inputs, outputs)
    save(graph, "tts_lm", [const("mask_scale", 0.1), const("eos_length", [EOS_LENGTH]),
                           const("eos_shape", [1, 1], TensorProto.INT64)] + cache_initializers())


def diffusion_head():
    nodes = [
        helper.make_node("Slice", ["condition", "zero", "two", "axis1"], ["target"]),
        helper.make_node("Mul", ["target", "sqrt2"], ["scaled_target"]),
        helper.make_node("Sub", ["noisy_latent", "scaled_target"], ["velocity"]),
        helper.make_node("Mul", ["timestep", "zero_half"], ["no_time"]),
        helper.make_node("Add", ["velocity", "no_time"], ["predicted_noise"]),
    ]
    inputs = [helper.make_tensor_value_info("noisy_latent", F16, [1, LATENT]),
              helper.make_tensor_value_info("timestep", F16, [1]),
              helper.make_tensor_value_info("condition", F16, [1, HIDDEN])]
    outputs = [helper.make_tensor_value_info("predicted_noise", F16, [1, LATENT])]
    graph = helper.make_graph(nodes, "tiny_diffusion_head", inputs, outputs)
    save(graph, "diffusion_head", [const("sqrt2", np.sqrt(2.0)), const("zero_half", 0.0)] + slice_initializers())


def acoustic_connector():
    nodes = [helper.make_node("Concat", ["latent", "latent"], ["pair"], axis=2),
             helper.make_node("Mul", ["pair", "half"], ["embedding"])]
    inputs = [helper.make_tensor_value_info("latent", F16, [1, 1, LATENT])]
    outputs = [helper.make_tensor_value_info("embedding", F16, [1, 1, HIDDEN])]
    save(helper.make_graph(nodes, "tiny_acoustic_connector", inputs, outputs), "acoustic_connector",
         [const("half", 0.5)])


def vocoder():
    nodes = [helper.make_node("Concat", ["latents", "latents"], ["pairs"], axis=2),
             helper.make_node("Reshape", ["pairs", "flat"], ["audio"])]
    inputs = [helper.make_tensor_value_info("latents", F16, [1, "num_frames", LATENT])]
    outputs = [helper.make_tensor_value_info("audio", F16, [1, "num_samples"])]
    save(helper.make_graph(nodes, "tiny_vocoder", inputs, outputs), "vocoder",
         [const("flat", [1, -1], TensorProto.INT64)])


def bytes_to_unicode():
    """GPT-2 byte-level alphabet: byte b is the character that the tokenizer uses for it."""
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    chars = printable[:]
    extra = 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            chars.append(256 + extra)
            extra += 1
    return {b: chr(c) for b, c in zip(printable, chars)}


def tokenizer():
    """A Qwen2-style byte-level BPE tokenizer without merges: every UTF-8 byte is one token whose id is the byte."""
    alphabet = bytes_to_unicode()
    pattern = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|"
               r"\s*[\r\n]+|\s+(?!\S)|\s+")
    end = {"id": 256, "content": "<|endoftext|>", "single_word": False, "lstrip": False, "rstrip": False,
           "normalized": False, "special": True}
    write_json(ROOT / "tokenizer" / "tokenizer.json", {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": [end],
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}]},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
        "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "",
                  "end_of_word_suffix": "", "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": {alphabet[b]: b for b in range(256)}, "merges": []},
    })
    write_json(ROOT / "tokenizer" / "tokenizer_config.json", {
        "add_bos_token": False, "add_prefix_space": False, "bos_token": None, "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>", "unk_token": None, "model_max_length": 8192, "tokenizer_class": "Qwen2Tokenizer",
        "added_tokens_decoder": {"256": {k: v for k, v in end.items() if k != "id"}},
    })
    # ORT GenAI reads tokenizers through a genai_config.json. Only the token ids are used.
    write_json(ROOT / "tokenizer" / "genai_config.json", {"model": {
        "type": "qwen2", "bos_token_id": 256, "eos_token_id": 256, "pad_token_id": 256, "vocab_size": 257,
        "context_length": 8192}})


def voice(name, tts_prompt, negative, compressed):
    arrays = {"neg_tts_lm_hidden": np.full((1, 1, HIDDEN), negative, dtype=np.float16),
              "lm_hidden": np.zeros((1, 2, HIDDEN), dtype=np.float16)}
    for prefix, length, value in (("lm", 2, 0.01), ("tts_lm", tts_prompt, 0.02), ("neg_tts_lm", 1, 0.0)):
        for kind in ("key", "value"):
            arrays[f"{prefix}_{kind}_0"] = np.full((1, 1, length, 2), value, dtype=np.float16)
    (ROOT / "voices").mkdir(exist_ok=True)
    (np.savez_compressed if compressed else np.savez)(ROOT / "voices" / f"{name}.npz", **arrays)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def main():
    text_lm()
    tts_lm()
    diffusion_head()
    acoustic_connector()
    vocoder()
    tokenizer()
    voice("alpha", 3, 0.05, compressed=False)
    voice("beta", 5, -0.05, compressed=True)
    write_json(ROOT / "tts_config.json", {
        "model_type": "vibevoice",
        "graphs": {"text_lm": "text_lm.onnx", "tts_lm": "tts_lm.onnx", "diffusion_head": "diffusion_head.onnx",
                   "acoustic_connector": "acoustic_connector.onnx", "vocoder": "vocoder.onnx"},
        "tokenizer": "tokenizer", "speech_scaling_factor": 0.5, "speech_bias_factor": 0.1, "text_window_size": 5,
        "speech_window_size": 6, "cfg_scale": 2.0, "diffusion": {"timesteps": [500], "sigmas": [1.0, 0.0]},
        "default_voice": "Alpha",
        "voices": [{"name": "Alpha", "file": "voices/alpha.npz", "language": "en", "gender": "female"},
                   {"name": "Beta", "file": "voices/beta.npz", "language": "en-in", "gender": "male"}],
    })


if __name__ == "__main__":
    main()
