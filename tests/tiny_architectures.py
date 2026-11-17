from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from transformers import (
    GptOssConfig,
    GptOssForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2Config,
    Qwen2ForCausalLM,
)

FAMILIES = ("llama", "qwen2", "gpt_oss")
GPT_OSS_SLIDING_WINDOW = 8


def tiny_config(family: str, vocab_size: int, **special_tokens: Any) -> PretrainedConfig:
    if family == "llama":
        return LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            attn_implementation="eager",
            tie_word_embeddings=False,
            **special_tokens,
        )
    if family == "qwen2":
        return Qwen2Config(
            vocab_size=vocab_size,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            tie_word_embeddings=False,
            **special_tokens,
        )
    if family == "gpt_oss":
        return GptOssConfig(
            vocab_size=vocab_size,
            hidden_size=64,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_local_experts=4,
            num_experts_per_tok=2,
            sliding_window=GPT_OSS_SLIDING_WINDOW,
            tie_word_embeddings=False,
            experts_implementation="eager",
            **special_tokens,
        )
    raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")


def tiny_causal_lm(
    family: str, vocab_size: int, seed: int = 0, dtype: torch.dtype = torch.float64, **special_tokens: Any
) -> PreTrainedModel:
    config = tiny_config(family, vocab_size, **special_tokens)
    model_class = {"llama": LlamaForCausalLM, "qwen2": Qwen2ForCausalLM, "gpt_oss": GptOssForCausalLM}[family]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = model_class(config)
    return model.to(dtype).eval()


def save_mxfp4_gpt_oss(directory: str | os.PathLike[str], tokenizer: PreTrainedTokenizerBase, seed: int = 0) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    special = {"pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id}
    model = tiny_causal_lm("gpt_oss", len(tokenizer), seed=seed, dtype=torch.bfloat16, **special)
    config = model.config
    state = {name: tensor.contiguous() for name, tensor in model.state_dict().items()}
    generator = torch.Generator().manual_seed(seed)
    hidden, intermediate, experts = config.hidden_size, config.intermediate_size, config.num_local_experts
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}.mlp.experts."
        for name, rows, cols in (("gate_up_proj", 2 * intermediate, hidden), ("down_proj", hidden, intermediate)):
            del state[prefix + name]
            shape = (experts, rows, cols // 32)
            blocks = torch.randint(0, 256, (*shape, 16), generator=generator, dtype=torch.uint8)
            state[prefix + name + "_blocks"] = blocks
            state[prefix + name + "_scales"] = torch.randint(118, 122, shape, generator=generator, dtype=torch.uint8)
    save_file(state, str(directory / "model.safetensors"), metadata={"format": "pt"})
    serialized = config.to_dict()
    unquantized = ["model.layers.*.self_attn", "model.layers.*.mlp.router", "model.embed_tokens", "lm_head"]
    serialized["quantization_config"] = {"quant_method": "mxfp4", "modules_to_not_convert": unquantized}
    (directory / "config.json").write_text(json.dumps(serialized, indent=2), encoding="utf-8")
    tokenizer.save_pretrained(directory)
    return directory
