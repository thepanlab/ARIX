from __future__ import annotations

import contextlib
import itertools
from collections.abc import Iterator, Sequence
from typing import Any

import torch
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from arix.config import GenerationConfig


def stop_token_ids(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> list[int]:
    ids: set[int] = set()
    configured = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if isinstance(configured, int):
        ids.add(configured)
    elif configured is not None:
        ids.update(int(i) for i in configured)
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    if not ids:
        raise ValueError("the model and tokenizer define no EOS token")
    return sorted(ids)


@torch.no_grad()
def generate_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[Sequence[int]],
    gen: GenerationConfig,
    generation_batch_size: int,
    synced_gpus: bool = False,
) -> list[list[int]]:
    if generation_batch_size < 1:
        raise ValueError("generation_batch_size must be >= 1")
    if gen.prefill_chunk_size is not None and gen.prefill_chunk_size < 1:
        raise ValueError("generation.prefill_chunk_size must be >= 1 or null")
    stops = stop_token_ids(model, tokenizer)
    kwargs = _generate_kwargs(gen, stops, tokenizer.pad_token_id, synced_gpus)
    responses: list[list[int]] = []
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(prompts), generation_batch_size):
            chunk = prompts[start : start + generation_batch_size]
            width = max(len(p) for p in chunk)
            if synced_gpus and gen.prefill_chunk_size is not None:
                width = _widest_over_processes(width, model.device)
            input_ids, attention_mask = _left_pad(chunk, tokenizer.pad_token_id, width)
            output = model.generate(  # type: ignore[operator]
                input_ids=input_ids.to(model.device), attention_mask=attention_mask.to(model.device), **kwargs
            )
            for row in output[:, input_ids.shape[1] :].tolist():
                responses.append(_cut_after_stop(row, stops))
    finally:
        model.train(was_training)
    return responses


def _generate_kwargs(gen: GenerationConfig, stops: list[int], pad_id: int, synced_gpus: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": gen.max_new_tokens,
        "eos_token_id": stops,
        "pad_token_id": pad_id,
        "repetition_penalty": 1.0,
        "synced_gpus": synced_gpus,
        "use_cache": True,
    }
    if gen.prefill_chunk_size is not None:
        kwargs["prefill_chunk_size"] = gen.prefill_chunk_size
    if gen.temperature > 0:
        kwargs.update(do_sample=True, temperature=gen.temperature, top_p=gen.top_p, top_k=gen.top_k)
    else:
        kwargs.update(do_sample=False)
    return kwargs


def _widest_over_processes(width: int, device: torch.device) -> int:
    widest = torch.tensor([width], dtype=torch.long, device=device)
    torch.distributed.all_reduce(widest, op=torch.distributed.ReduceOp.MAX)
    return int(widest.item())


def _left_pad(
    prompts: Sequence[Sequence[int]], pad_id: int, width: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    if any(len(p) == 0 for p in prompts):
        raise ValueError("cannot generate from an empty prompt")
    width = max(max(len(p) for p in prompts), width or 0)
    input_ids = torch.full((len(prompts), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(prompts), width), dtype=torch.long)
    for row, prompt in enumerate(prompts):
        input_ids[row, width - len(prompt) :] = torch.tensor(list(prompt), dtype=torch.long)
        attention_mask[row, width - len(prompt) :] = 1
    return input_ids, attention_mask


def _cut_after_stop(tokens: list[int], stops: Sequence[int]) -> list[int]:
    stop_set = set(stops)
    for index, token in enumerate(tokens):
        if token in stop_set:
            return tokens[: index + 1]
    return tokens


@contextlib.contextmanager
def zero3_generation(engine: nn.Module, gather: bool) -> Iterator[bool]:
    if not gather:
        yield False
        return
    import deepspeed

    with deepspeed.zero.GatheredParameters(list(engine.parameters())):
        _remove_zero3_hooks(engine)
        try:
            yield True
        finally:
            _add_zero3_hooks(engine)


def _zero3_offload(engine: nn.Module) -> Any:
    if not hasattr(engine, "optimizer"):
        return None
    optimizer = engine.optimizer
    if optimizer is None:
        raise RuntimeError("The model optimizer is None, which is not yet supported.")
    return optimizer.parameter_offload if hasattr(optimizer, "parameter_offload") else optimizer


def _zero3_parameters(module: nn.Module) -> list[nn.Parameter]:
    named = itertools.chain(module.named_parameters(recurse=True), module.ds_external_parameters())  # type: ignore[operator]
    return [parameter for _, parameter in named]


def _remove_zero3_hooks(engine: nn.Module) -> None:
    offload = _zero3_offload(engine)
    if offload is None:
        return
    for parameter in _zero3_parameters(offload.module):
        parameter.ds_active_sub_modules.clear()  # type: ignore[attr-defined]
    for hook in [*offload.forward_hooks, *offload.backward_hooks]:
        hook.remove()
    offload.forward_hooks = []
    offload.backward_hooks = []


def _add_zero3_hooks(engine: nn.Module) -> None:
    import deepspeed
    from packaging import version

    offload = _zero3_offload(engine)
    if offload is None:
        return
    coordinator = getattr(offload, "param_coordinator", None)
    if coordinator is not None and not coordinator.is_invalid_trace():
        coordinator._invalidate_trace()
    if version.parse(deepspeed.__version__) >= version.parse("0.16.4"):
        offload._register_deepspeed_module(offload.module)
    else:
        offload._register_hooks_recursively(offload.module)
