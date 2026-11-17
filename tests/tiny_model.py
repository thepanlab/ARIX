from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

PAD_TOKEN, EOS_TOKEN, BOS_TOKEN = "<pad>", "<eos>", "<bos>"
VOCAB_SIZE = 400
MAX_POSITIONS = 32768
CHAT_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}assistant: {% endif %}"
)


def train_tokenizer(texts: Iterable[str], vocab_size: int = VOCAB_SIZE) -> PreTrainedTokenizerFast:
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[PAD_TOKEN, EOS_TOKEN, BOS_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    backend.train_from_iterator(texts, trainer=trainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token=PAD_TOKEN, eos_token=EOS_TOKEN, bos_token=BOS_TOKEN
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


def build_tiny_model(tmp_path: str | os.PathLike[str], texts: Iterable[str], seed: int = 0) -> Path:
    directory = Path(tmp_path)
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer = train_tokenizer(texts)
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=MAX_POSITIONS,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=False,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = LlamaForCausalLM(config).to(torch.float32)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return directory
