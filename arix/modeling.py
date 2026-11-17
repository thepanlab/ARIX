from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedType
from torch import Tensor, nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from transformers.utils.quantization_config import Mxfp4Config

from arix.config import ModelConfig
from arix.task import Example

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
LOGPROB_CHUNK = 256


def resolve_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    try:
        return _DTYPES[name]
    except KeyError:
        raise ValueError(f"unknown dtype {name!r}; expected one of {[*_DTYPES, 'auto']}") from None


def load_tokenizer(cfg: ModelConfig) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer or cfg.name_or_path, trust_remote_code=cfg.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def encode_prompt(tokenizer: PreTrainedTokenizerBase, example: Example) -> list[int]:
    add_special_tokens = bool(example.meta.get("add_special_tokens", False))
    return list(tokenizer(example.prompt, add_special_tokens=add_special_tokens).input_ids)


def disable_dropout(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
        for name, value in list(vars(module).items()):
            if name.endswith("dropout") and isinstance(value, float):
                setattr(module, name, 0.0)


def enable_gradient_checkpointing(model: PreTrainedModel) -> None:
    kwargs = {"use_reentrant": False}
    if hasattr(model, "activation_checkpointing_enable"):
        model.activation_checkpointing_enable(activation_checkpointing_kwargs=kwargs)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)


def dequantization_kwargs(cfg: ModelConfig) -> dict[str, Any]:
    config = AutoConfig.from_pretrained(cfg.name_or_path, trust_remote_code=cfg.trust_remote_code)
    quantization = getattr(config, "quantization_config", None)
    if isinstance(quantization, Mapping):
        method = quantization.get("quant_method")
    else:
        method = getattr(quantization, "quant_method", None)
    return {"quantization_config": Mxfp4Config(dequantize=True)} if method == "mxfp4" else {}


_MXFP4_SERIALIZED_SUFFIXES = ("_blocks", "_scales")


def forget_mxfp4_conversions(model: PreTrainedModel) -> None:
    conversions = getattr(model, "_weight_conversions", None)
    if not conversions:
        return

    def reads_mxfp4_tensors(conversion: Any) -> bool:
        sources = getattr(conversion, "_original_source_patterns", ()) or getattr(conversion, "source_patterns", ())
        return any(str(source).rstrip("$").endswith(_MXFP4_SERIALIZED_SUFFIXES) for source in sources)

    kept = [conversion for conversion in conversions if not reads_mxfp4_tensors(conversion)]
    model._weight_conversions = kept  # type: ignore[assignment]


def _load_causal_lm(cfg: ModelConfig) -> PreTrainedModel:
    kwargs: dict[str, Any] = {"dtype": resolve_dtype(cfg.torch_dtype), "trust_remote_code": cfg.trust_remote_code}
    if cfg.attn_implementation is not None:
        kwargs["attn_implementation"] = cfg.attn_implementation
    dequantize = dequantization_kwargs(cfg)
    kwargs.update(dequantize)
    model = AutoModelForCausalLM.from_pretrained(cfg.name_or_path, **kwargs)
    if dequantize:
        forget_mxfp4_conversions(model)
    disable_dropout(model)
    return model


def load_policy_and_tokenizer(cfg: ModelConfig) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = load_tokenizer(cfg)
    model = _load_causal_lm(cfg)
    if cfg.gradient_checkpointing:
        enable_gradient_checkpointing(model)
    return model, tokenizer


def load_reference_model(cfg: ModelConfig) -> PreTrainedModel:
    model = _load_causal_lm(cfg)
    model.requires_grad_(False)
    model.eval()
    return model


def prepare_reference_model(model: PreTrainedModel, accelerator: Accelerator) -> nn.Module:
    model.requires_grad_(False)
    if accelerator.distributed_type != DistributedType.DEEPSPEED:
        prepared = accelerator.prepare_model(model, evaluation_mode=True)
        prepared.eval()
        return prepared
    import deepspeed

    config = copy.deepcopy(accelerator.state.deepspeed_plugin.deepspeed_config)
    config.pop("optimizer", None)
    config.pop("scheduler", None)
    zero = config.setdefault("zero_optimization", {})
    if zero.get("stage", 0) == 3:
        hidden_size = model.config.hidden_size
        defaults = {
            "reduce_bucket_size": hidden_size * hidden_size,
            "stage3_param_persistence_threshold": 10 * hidden_size,
            "stage3_prefetch_bucket_size": int(0.9 * hidden_size * hidden_size),
        }
        for key, value in defaults.items():
            if zero.get(key, "auto") == "auto":
                zero[key] = value
    else:
        zero["stage"] = 0
    engine, *_ = deepspeed.initialize(model=model, config=config)
    engine.eval()
    return engine


@dataclass
class SequenceBatch:
    input_ids: Tensor
    attention_mask: Tensor
    response_mask: Tensor
    prompt_lens: list[int]
    response_lens: list[int]

    def to(self, device: torch.device | str) -> SequenceBatch:
        return SequenceBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            response_mask=self.response_mask.to(device),
            prompt_lens=self.prompt_lens,
            response_lens=self.response_lens,
        )


def build_sequence_batch(
    prompt_ids: Sequence[Sequence[int]], response_ids: Sequence[Sequence[int]], pad_id: int
) -> SequenceBatch:
    if len(prompt_ids) != len(response_ids):
        raise ValueError(f"{len(prompt_ids)} prompts but {len(response_ids)} responses")
    if not prompt_ids:
        raise ValueError("empty batch")
    prompt_lens = [len(p) for p in prompt_ids]
    response_lens = [len(r) for r in response_ids]
    if min(prompt_lens) == 0 or min(response_lens) == 0:
        raise ValueError("prompts and responses must contain at least one token")
    width = max(p + r for p, r in zip(prompt_lens, response_lens, strict=True))
    batch_size = len(prompt_ids)
    input_ids = torch.full((batch_size, width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, width), dtype=torch.long)
    response_mask = torch.zeros((batch_size, width), dtype=torch.bool)
    for row, (prompt, response) in enumerate(zip(prompt_ids, response_ids, strict=True)):
        start, end = len(prompt), len(prompt) + len(response)
        input_ids[row, :end] = torch.tensor([*prompt, *response], dtype=torch.long)
        attention_mask[row, :end] = 1
        response_mask[row, start:end] = True
    return SequenceBatch(input_ids, attention_mask, response_mask, prompt_lens, response_lens)


@dataclass
class ForwardOutput:
    token_logprobs: Tensor
    hidden: Tensor | None
    logits: Tensor | None
    logits_start: int = 0


def first_response_logits_position(batch: SequenceBatch) -> int:
    return min(batch.prompt_lens) - 1


def forward_outputs(
    model: nn.Module, batch: SequenceBatch, need_hidden: bool = False, need_logits: bool = False
) -> ForwardOutput:
    width = batch.input_ids.shape[1]
    outputs = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        output_hidden_states=-1 if need_hidden else False,
        logits_to_keep=width - first_response_logits_position(batch),
        use_cache=False,
    )
    logits = outputs.logits
    start = width - logits.shape[1]
    return ForwardOutput(
        token_logprobs=_response_logprobs(logits, batch, start),
        hidden=outputs.hidden_states[-1] if need_hidden else None,
        logits=logits if need_logits else None,
        logits_start=start,
    )


def forward_logprobs_hidden(model: nn.Module, batch: SequenceBatch, need_hidden: bool) -> tuple[Tensor, Tensor | None]:
    out = forward_outputs(model, batch, need_hidden=need_hidden)
    return out.token_logprobs, out.hidden


def _selected_logprobs(logits: Tensor, labels: Tensor) -> Tensor:
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def _exact_kl(policy_logits: Tensor, ref_logits: Tensor) -> Tensor:
    log_p = F.log_softmax(policy_logits.float(), dim=-1)
    log_q = F.log_softmax(ref_logits.float(), dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1)


class _ChunkedRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, fn: Callable[..., Tensor], chunk: int, rows: Tensor, *tables: Tensor) -> Tensor:
        ctx.fn, ctx.chunk = fn, chunk
        ctx.save_for_backward(rows, *tables)
        parts = [fn(*(table[rows[s : s + chunk]] for table in tables)) for s in range(0, rows.numel(), chunk)]
        return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.float32, device=rows.device)

    @staticmethod
    def backward(ctx: Any, grad: Tensor) -> tuple[Tensor | None, ...]:
        rows, *tables = ctx.saved_tensors
        needed = ctx.needs_input_grad[3:]
        buffers = [torch.zeros_like(table) if need else None for table, need in zip(tables, needed, strict=True)]
        for s in range(0, rows.numel(), ctx.chunk):
            index = rows[s : s + ctx.chunk]
            inputs = [table[index].requires_grad_(need) for table, need in zip(tables, needed, strict=True)]
            with torch.enable_grad():
                values = ctx.fn(*inputs)
            targets = [(x, b) for x, b in zip(inputs, buffers, strict=True) if b is not None]
            chunk_grads = torch.autograd.grad(
                values, [x for x, _ in targets], grad[s : s + ctx.chunk], allow_unused=True
            )
            for (_, buffer), chunk_grad in zip(targets, chunk_grads, strict=True):
                if chunk_grad is not None:
                    buffer[index] = chunk_grad
        return (None, None, None, *buffers)


def _response_logprobs(logits: Tensor, batch: SequenceBatch, start: int) -> Tensor:
    keep, vocab = logits.shape[1], logits.shape[2]
    rows_b, rows_t = batch.response_mask.nonzero(as_tuple=True)
    rows = rows_b * keep + (rows_t - 1 - start)
    next_tokens = F.pad(batch.input_ids[:, start + 1 :], (0, 1))
    values = _ChunkedRows.apply(
        _selected_logprobs, LOGPROB_CHUNK, rows, logits.reshape(-1, vocab), next_tokens.reshape(-1)
    )
    zeros = torch.zeros(batch.input_ids.shape, dtype=torch.float32, device=logits.device)
    return zeros.index_put((rows_b, rows_t), values)


def full_kl_per_token(policy_logits: Tensor, ref_logits: Tensor, mask: Tensor, chunk: int, start: int = 0) -> Tensor:
    if policy_logits.shape != ref_logits.shape:
        raise ValueError(f"policy logits {tuple(policy_logits.shape)} and reference logits {tuple(ref_logits.shape)}")
    keep, vocab = policy_logits.shape[1], policy_logits.shape[2]
    rows_b, rows_t = mask[:, start + 1 :].nonzero(as_tuple=True)
    rows_t = rows_t + start + 1
    rows = rows_b * keep + (rows_t - 1 - start)
    values = _ChunkedRows.apply(_exact_kl, chunk, rows, policy_logits.reshape(-1, vocab), ref_logits.reshape(-1, vocab))
    return torch.zeros(mask.shape, dtype=torch.float32, device=policy_logits.device).index_put((rows_b, rows_t), values)
