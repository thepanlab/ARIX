from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from tiny_architectures import FAMILIES, tiny_causal_lm
from torch import Tensor
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from transformers import PreTrainedModel

import arix.modeling as modeling
from arix.config import ARIXConfig, ModelConfig, OutputConfig, TrainingConfig
from arix.embedding import RationaleProjector, rationale_embedding
from arix.evaluation import embed_responses
from arix.modeling import (
    SequenceBatch,
    build_sequence_batch,
    forward_outputs,
    full_kl_per_token,
    load_policy_and_tokenizer,
    load_reference_model,
)
from arix.task import Example
from arix.trainer import ARIXTrainer

VOCAB, PAD_ID, PROJ_DIM = 97, 0, 8
PROMPTS = [[5, 6, 7, 8, 9, 10, 11], [12, 13], [14, 15, 16, 17], [18, 19, 20, 21, 22, 23, 24, 25, 26]]
RESPONSES = [[30, 31, 32, 33, 1], [34, 35, 36, 37, 38, 39, 40, 41, 42], [43], [44, 45, 46, 47, 48, 49]]
SMALL_CHUNK = 3


def batch_of(prompts: list[list[int]] = PROMPTS, responses: list[list[int]] = RESPONSES) -> SequenceBatch:
    return build_sequence_batch(prompts, responses, PAD_ID)


def reference_forward(model: PreTrainedModel, batch: SequenceBatch) -> tuple[Tensor, Tensor, Tensor]:
    out = model(
        input_ids=batch.input_ids, attention_mask=batch.attention_mask, output_hidden_states=True, use_cache=False
    )
    logp = torch.log_softmax(out.logits[:, :-1].float(), dim=-1).gather(-1, batch.input_ids[:, 1:, None]).squeeze(-1)
    return F.pad(logp, (1, 0)).masked_fill(~batch.response_mask, 0.0), out.hidden_states[-1], out.logits


def surrogate(token_logprobs: Tensor, hidden: Tensor, batch: SequenceBatch, projector: RationaleProjector) -> Tensor:
    generator = torch.Generator().manual_seed(7)
    weights = torch.randn(token_logprobs.shape, generator=generator)
    direction = torch.randn(len(batch.prompt_lens), projector.proj_dim, generator=generator)
    emb = rationale_embedding(hidden, token_logprobs, batch.response_mask, projector)
    return (token_logprobs * weights).sum() + (emb * direction).sum()


def gradients(model: PreTrainedModel) -> dict[str, Tensor]:
    grads = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    return grads


@pytest.mark.parametrize("chunk", [SMALL_CHUNK, modeling.LOGPROB_CHUNK])
@pytest.mark.parametrize("family", FAMILIES)
def test_forward_outputs_match_the_full_logits_computation(
    family: str, chunk: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(modeling, "LOGPROB_CHUNK", chunk)
    model = tiny_causal_lm(family, VOCAB, seed=0)
    projector = RationaleProjector(model.config.hidden_size, PROJ_DIM, seed=0)
    batch = batch_of()
    mask = batch.response_mask

    out = forward_outputs(model, batch, need_hidden=True, need_logits=True)
    assert out.hidden is not None and out.logits is not None
    surrogate(out.token_logprobs, out.hidden, batch, projector).backward()
    new_grads = gradients(model)

    logp, hidden, logits = reference_forward(model, batch)
    surrogate(logp, hidden, batch, projector).backward()
    reference_grads = gradients(model)

    start = min(batch.prompt_lens) - 1
    assert out.logits_start == start and out.logits.shape == (4, batch.input_ids.shape[1] - start, VOCAB)
    torch.testing.assert_close(out.logits, logits[:, start:], rtol=1e-12, atol=1e-12)
    assert torch.equal(out.hidden, hidden)
    torch.testing.assert_close(out.token_logprobs, logp, rtol=0.0, atol=1e-6)
    assert torch.all(out.token_logprobs[~mask] == 0) and torch.all(out.token_logprobs[mask] < 0)
    assert new_grads.keys() == reference_grads.keys() and len(new_grads) == len(list(model.parameters()))
    for name, expected in reference_grads.items():
        assert float(expected.abs().max()) > 0, name
        torch.testing.assert_close(new_grads[name], expected, rtol=1e-5, atol=1e-9, msg=lambda m, n=name: f"{n}: {m}")


@pytest.mark.parametrize("family", FAMILIES)
def test_exact_kl_matches_the_full_logits_computation(family: str) -> None:
    policy = tiny_causal_lm(family, VOCAB, seed=0)
    ref = tiny_causal_lm(family, VOCAB, seed=1).requires_grad_(False)
    batch = batch_of()
    mask = batch.response_mask
    weights = torch.randn(mask.shape, generator=torch.Generator().manual_seed(3))

    out = forward_outputs(policy, batch, need_logits=True)
    with torch.no_grad():
        ref_out = forward_outputs(ref, batch, need_logits=True)
    assert out.logits is not None and ref_out.logits is not None and out.logits_start == ref_out.logits_start
    kl = full_kl_per_token(out.logits, ref_out.logits, mask, SMALL_CHUNK, start=out.logits_start)
    (kl * weights).sum().backward()
    new_grads = gradients(policy)

    _, _, logits = reference_forward(policy, batch)
    with torch.no_grad():
        _, _, ref_logits = reference_forward(ref, batch)
    log_p, log_q = (F.log_softmax(x[:, :-1].float(), dim=-1) for x in (logits, ref_logits))
    expected = F.pad((log_p.exp() * (log_p - log_q)).sum(dim=-1), (1, 0)).masked_fill(~mask, 0.0)
    (expected * weights).sum().backward()
    reference_grads = gradients(policy)

    torch.testing.assert_close(kl, expected, rtol=0.0, atol=1e-6)
    assert torch.all(kl[mask] > 0) and torch.all(kl[~mask] == 0)
    for name, value in reference_grads.items():
        torch.testing.assert_close(new_grads[name], value, rtol=1e-5, atol=1e-9, msg=lambda m, n=name: f"{n}: {m}")
    torch.testing.assert_close(full_kl_per_token(logits, ref_logits, mask, SMALL_CHUNK), expected, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("family", FAMILIES)
def test_evaluation_embeddings_match_the_full_logits_computation(family: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(modeling, "LOGPROB_CHUNK", SMALL_CHUNK)
    model = tiny_causal_lm(family, VOCAB, seed=0)
    projector = RationaleProjector(model.config.hidden_size, PROJ_DIM, seed=0)
    embeddings = embed_responses(model, PROMPTS, RESPONSES, projector, "softmax", batch_size=3, pad_id=PAD_ID)
    expected = []
    for start in (0, 3):
        batch = batch_of(PROMPTS[start : start + 3], RESPONSES[start : start + 3])
        with torch.no_grad():
            logp, hidden, _ = reference_forward(model, batch)
        expected.append(rationale_embedding(hidden, logp, batch.response_mask, projector))
    torch.testing.assert_close(embeddings, torch.cat(expected), rtol=0.0, atol=1e-6)


class VocabularySizedTensors(TorchDispatchMode):
    def __init__(self, vocab: int, hidden: int) -> None:
        super().__init__()
        self.vocab, self.hidden = vocab, hidden
        self.seen: list[tuple[tuple[int, ...], torch.dtype]] = []

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        out = func(*args, **(kwargs or {}))
        for value in tree_flatten(out)[0]:
            if isinstance(value, Tensor) and value.dim() >= 2 and value.shape[-1] == self.vocab:
                if tuple(value.shape) != (self.hidden, self.vocab):
                    self.seen.append((tuple(value.shape), value.dtype))
        return out


@pytest.mark.parametrize("family", FAMILIES)
def test_no_logits_sized_tensor_beyond_the_kept_positions(family: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(modeling, "LOGPROB_CHUNK", SMALL_CHUNK)
    model = tiny_causal_lm(family, VOCAB, seed=0, dtype=torch.bfloat16)
    projector = RationaleProjector(model.config.hidden_size, PROJ_DIM, seed=0)
    prompts = [[60 + i for i in range(length)] for length in (10, 8, 12, 9)]
    batch = batch_of(prompts, RESPONSES)
    width = batch.input_ids.shape[1]
    kept = width - (min(batch.prompt_lens) - 1)
    calls: list[dict[str, Any]] = []
    model.register_forward_hook(
        lambda _module, _args, kwargs, output: calls.append({**kwargs, "output": output}), with_kwargs=True
    )
    assert len(PROMPTS) * kept != model.config.hidden_size
    tracker = VocabularySizedTensors(VOCAB, model.config.hidden_size)
    with tracker:
        out = forward_outputs(model, batch, need_hidden=True)
        assert out.hidden is not None
        surrogate(out.token_logprobs, out.hidden, batch, projector).backward()

    (call,) = calls
    assert call["logits_to_keep"] == kept == width - 7 and call["output"].logits.shape == (4, kept, VOCAB)
    *earlier, last = call["output"].hidden_states
    assert all(state is None for state in earlier) and last.shape == (4, width, model.config.hidden_size)
    assert max(torch.Size(shape).numel() for shape, _ in tracker.seen) <= 4 * kept * VOCAB
    float32 = [shape for shape, dtype in tracker.seen if dtype == torch.float32]
    assert float32 and max(torch.Size(shape).numel() for shape in float32) <= SMALL_CHUNK * VOCAB


def test_the_zero3_config_offloads_the_optimizer_to_the_cpu(code_root: Path) -> None:
    config = yaml.safe_load((code_root / "configs" / "accelerate" / "zero3_6gpu.yaml").read_text(encoding="utf-8"))
    deepspeed = config["deepspeed_config"]
    assert deepspeed["zero_stage"] == 3 and config["num_processes"] == 6
    assert deepspeed["offload_optimizer_device"] == "cpu" and deepspeed["offload_param_device"] == "none"


class AcceptEverything:
    name = "accept_everything"

    def verify_batch(self, _examples: list[Example], responses: list[str]) -> list[bool]:
        return [True] * len(responses)

    def extract_batch(self, _examples: list[Example], responses: list[str]) -> list[Any]:
        return [None] * len(responses)


@pytest.mark.filterwarnings("ignore:Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`")
def test_the_learning_rate_schedule_drives_the_optimizer_that_is_stepped(
    tiny_model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = ARIXConfig(
        model=ModelConfig(name_or_path=str(tiny_model_dir), torch_dtype="float32", gradient_checkpointing=False),
        training=TrainingConfig(
            epochs=1, batch_size=1, group_size=2, micro_batch_size=2, generation_batch_size=2, warmup_steps=2
        ),
        output=OutputConfig(output_dir=str(tmp_path / "run")),
    )
    accelerator = Accelerator(cpu=True)
    prepare = accelerator.prepare
    replacements: list[torch.optim.Optimizer] = []

    def replace_optimizer(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> Any:
        groups = [{**group, "params": list(group["params"])} for group in optimizer.param_groups]
        replacement = torch.optim.AdamW(groups)
        replacements.append(replacement)
        return prepare(model, replacement)

    monkeypatch.setattr(accelerator, "prepare", replace_optimizer)
    model, tokenizer = load_policy_and_tokenizer(cfg.model)
    example = Example(uid="u0", prompt="Question:\n1 + 1?\n", reference="2", target="2")
    trainer = ARIXTrainer(
        cfg, AcceptEverything(), [example], model, tokenizer, load_reference_model(cfg.model), accelerator
    )
    (replacement,) = replacements
    learning_rates = []
    for _ in range(3):
        learning_rates.append({group["lr"] for group in replacement.param_groups})
        trainer.scheduler.step()
    rate = cfg.training.learning_rate
    assert learning_rates == [{0.0}, {rate / 2}, {rate}]
