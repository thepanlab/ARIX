from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase

import arix.trainer as trainer_module
from arix.bank import ReferenceBank
from arix.config import (
    ARIXConfig,
    EvaluationConfig,
    config_from_dict,
    load_config,
    load_run_config,
    read_yaml_mapping,
    save_config,
)
from arix.embedding import RationaleProjector
from arix.evaluation import load_projector, read_run_config
from arix.losses import grpo_sequence_losses
from arix.modeling import load_policy_and_tokenizer, load_reference_model
from arix.task import Example, ResponseDecoder, Task, decode_for_verification
from arix.trainer import (
    TRAIN_LOG,
    ARIXTrainer,
    LossSettings,
    StepRollouts,
    _collective_flags,
    _prompts_per_process,
    _shard,
    check_output_dir,
)

NUM_PROMPTS, GROUP_SIZE = 2, 4
EXAMPLES = [
    Example(uid=f"u{i}", prompt=f"Question:\n{i} + 3?\nAnswer reasoning:\n", reference=f"x = {i} + 3", target=i + 3)
    for i in range(NUM_PROMPTS)
]


class AcceptEverything:
    name = "accept_everything"

    def verify_batch(self, examples: list[Example], responses: list[str]) -> list[bool]:
        return [True] * len(responses)

    def extract_batch(self, examples: list[Example], responses: list[str]) -> list[Any]:
        return [None] * len(responses)


class AcceptEveryOther(AcceptEverything):
    def verify_batch(self, examples: list[Example], responses: list[str]) -> list[bool]:
        return [index % 2 == 0 for index in range(len(responses))]


class DecodingTask(AcceptEverything):
    def __init__(self) -> None:
        self.decoded: list[list[list[int]]] = []
        self.verified: list[str] = []

    def decode_responses(self, tokenizer: Any, responses: list[list[int]]) -> list[str]:
        self.decoded.append(responses)
        return [f"[{len(ids)} tokens] " + tokenizer.decode(ids, skip_special_tokens=False) for ids in responses]

    def verify_batch(self, examples: list[Example], responses: list[str]) -> list[bool]:
        self.verified.extend(responses)
        return super().verify_batch(examples, responses)


def run_config(directory: Path, model_dir: Path, *overrides: str) -> ARIXConfig:
    defaults = [
        f"model.name_or_path={model_dir}",
        "model.torch_dtype=float32",
        "model.gradient_checkpointing=false",
        "training.epochs=1",
        f"training.batch_size={NUM_PROMPTS}",
        f"training.group_size={GROUP_SIZE}",
        "training.micro_batch_size=3",
        "training.generation_batch_size=8",
        "training.max_steps=1",
        "generation.max_new_tokens=6",
        "arix.bank_size=3",
        f"output.output_dir={directory / 'run'}",
        "output.save_every_steps=0",
    ]
    return load_config(None, [*defaults, *overrides])


def make_trainer(
    cfg: ARIXConfig,
    task: Task,
    *,
    examples: Sequence[Example] = EXAMPLES,
    ref_model: PreTrainedModel | None = None,
    accelerator: Accelerator | None = None,
) -> ARIXTrainer:
    model, tokenizer = load_policy_and_tokenizer(cfg.model)
    if ref_model is None and cfg.training.kl_beta > 0:
        ref_model = load_reference_model(cfg.model)
    return ARIXTrainer(cfg, task, list(examples), model, tokenizer, ref_model, accelerator or Accelerator(cpu=True))


def read_log(cfg: ARIXConfig) -> list[dict[str, Any]]:
    lines = (Path(cfg.output.output_dir) / TRAIN_LOG).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


@dataclass
class CapturedStep:
    rollouts: StepRollouts
    epoch: int
    entries: dict[str, list[list[int]]] = field(default_factory=dict)
    anchors: Tensor | None = None
    rewards: Tensor | None = None
    params: dict[str, Tensor] = field(default_factory=dict)
    grads: dict[str, Tensor] = field(default_factory=dict)


def capture_steps(trainer: ARIXTrainer, monkeypatch: pytest.MonkeyPatch) -> list[CapturedStep]:
    steps: list[CapturedStep] = []
    sample, signals, clip = ARIXTrainer._sample, ARIXTrainer._signals, trainer.accelerator.clip_grad_norm_

    def on_sample(self: ARIXTrainer, examples: list[Example]) -> StepRollouts:
        rollouts = sample(self, examples)
        steps.append(CapturedStep(rollouts=rollouts, epoch=self.epoch))
        return rollouts

    def on_signals(self: ARIXTrainer, rollouts: StepRollouts, *args: Any) -> Any:
        result = signals(self, rollouts, *args)
        assert result.anchors is not None
        banks = self.banks.banks
        steps[-1].entries = {ex.uid: [list(e.token_ids) for e in banks[ex.uid].entries] for ex in rollouts.examples}
        steps[-1].anchors, steps[-1].rewards = result.anchors.clone(), result.rewards.clone()
        return result

    def on_clip(parameters: Any, max_norm: float, norm_type: float = 2.0) -> Any:
        named = list(trainer.policy.named_parameters())
        steps[-1].params = {name: p.detach().clone() for name, p in named}
        steps[-1].grads = {name: torch.zeros_like(p) if p.grad is None else p.grad.clone() for name, p in named}
        return clip(parameters, max_norm, norm_type)

    monkeypatch.setattr(ARIXTrainer, "_sample", on_sample)
    monkeypatch.setattr(ARIXTrainer, "_signals", on_signals)
    monkeypatch.setattr(trainer.accelerator, "clip_grad_norm_", on_clip)
    return steps


def perturbed_reference_model(cfg: ARIXConfig, scale: float = 0.05) -> PreTrainedModel:
    ref_model = load_reference_model(cfg.model)
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for parameter in ref_model.parameters():
            parameter.add_(scale * torch.randn(parameter.shape, generator=generator))
    return ref_model


def response_forward(model: nn.Module, prompt: list[int], response: list[int]) -> tuple[Tensor, Tensor]:
    out = model(input_ids=torch.tensor([prompt + response]), output_hidden_states=True, use_cache=False)
    log_probs = torch.log_softmax(out.logits[0, len(prompt) - 1 : -1].float(), dim=-1)
    return log_probs, out.hidden_states[-1][0, len(prompt) :]


def sampled_logprobs(log_probs: Tensor, response: list[int]) -> Tensor:
    return log_probs[torch.arange(len(response)), torch.tensor(response)]


def embed(log_probs: Tensor, hidden: Tensor, response: list[int], projector: RationaleProjector) -> Tensor:
    weights = torch.softmax(sampled_logprobs(log_probs, response), dim=0)
    return F.normalize(weights @ projector(hidden), dim=0)


def direct_gradients(
    policy: nn.Module,
    ref_model: nn.Module,
    projector: RationaleProjector,
    tokenizer: PreTrainedTokenizerBase,
    cfg: ARIXConfig,
    step: CapturedStep,
) -> tuple[dict[str, dict[str, Tensor]], Tensor]:
    t, a, group = cfg.training, cfg.arix, cfg.training.group_size
    assert step.anchors is not None
    surrogate, penalty, contrastive, rewards = [], [], [], []
    for p, example in enumerate(step.rollouts.examples):
        prompt = list(tokenizer(example.prompt, add_special_tokens=False).input_ids)
        responses = [list(step.rollouts.responses[i]) for i in range(p * group, (p + 1) * group)]
        correct = torch.tensor(step.rollouts.verdicts[p * group : (p + 1) * group])
        with torch.no_grad():
            entries = step.entries[example.uid]
            entry_emb = torch.stack([embed(*response_forward(policy, prompt, b), b, projector) for b in entries])
            ref_log_probs = [response_forward(ref_model, prompt, response)[0] for response in responses]
        forwards = [response_forward(policy, prompt, response) for response in responses]
        u = [sampled_logprobs(forward[0], response) for forward, response in zip(forwards, responses, strict=True)]
        emb = torch.stack([embed(*forward, o, projector) for forward, o in zip(forwards, responses, strict=True)])
        anchors = step.anchors[p * group : (p + 1) * group].cpu()
        similarity = emb.detach() @ entry_emb.T
        if step.epoch < a.retrieval_start_epoch:
            assert anchors.tolist() == [0] * group
        else:
            assert bool((similarity.gather(1, anchors[:, None])[:, 0] >= similarity.max(dim=1).values - 1e-5).all())
        anchor_emb = entry_emb[anchors]
        with torch.no_grad():
            reward = torch.where(correct, 1.0, a.partial_scale * (emb * anchor_emb).sum(dim=-1) + a.partial_bias)
            advantage = (reward - reward.mean()) / (reward.std() + t.adv_eps)
        clipped, kl = [], []
        for j, response in enumerate(responses):
            ratio = torch.exp(u[j] - u[j].detach())
            unclipped = ratio * advantage[j]
            clipped.append(torch.minimum(unclipped, ratio.clamp(1 - t.clip_eps, 1 + t.clip_eps) * advantage[j]).mean())
            if t.kl_estimator == "k3":
                log_ratio = sampled_logprobs(ref_log_probs[j], response) - u[j]
                kl.append((log_ratio.exp() - log_ratio - 1.0).mean())
            else:
                log_probs = forwards[j][0]
                kl.append((log_probs.exp() * (log_probs - ref_log_probs[j])).sum(dim=-1).mean())
        surrogate.append(-torch.stack(clipped).mean())
        penalty.append(t.kl_beta * torch.stack(kl).mean())
        logits = anchor_emb @ emb.T / a.tau
        l_pos = (torch.logsumexp(logits, dim=1) - logits.diagonal())[correct].mean() if bool(correct.any()) else 0.0
        contrastive.append(a.alpha_pos * l_pos)
        rewards.append(reward)
    named = list(policy.named_parameters())
    parts = {}
    for name, terms in (("surrogate", surrogate), ("kl", penalty), ("contrastive", contrastive)):
        value = torch.stack([torch.as_tensor(term) for term in terms]).mean()
        grads = torch.autograd.grad(value, [p for _, p in named], retain_graph=True, allow_unused=True)
        parts[name] = {n: torch.zeros_like(p) if g is None else g for (n, p), g in zip(named, grads, strict=True)}
    return parts, torch.cat(rewards)


@pytest.mark.parametrize(("kl_estimator", "micro_batch_size"), [("k3", 3), ("full", 5)])
def test_the_accumulated_gradient_of_a_step_is_that_of_equation_12(
    kl_estimator: str, micro_batch_size: int, tiny_model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    overrides = ["training.epochs=2", "training.max_steps=2", "training.learning_rate=1e-3"]
    overrides += [f"training.kl_estimator={kl_estimator}", f"training.micro_batch_size={micro_batch_size}"]
    cfg = run_config(tmp_path, tiny_model_dir, *overrides)
    ref_model = perturbed_reference_model(cfg)
    trainer = make_trainer(cfg, AcceptEveryOther(), ref_model=ref_model)
    steps = capture_steps(trainer, monkeypatch)
    trainer.train()

    assert [step.epoch for step in steps] == [1, 2] and all(step.grads for step in steps)
    assert steps[1].anchors is not None and bool((steps[1].anchors != 0).any())
    policy = AutoModelForCausalLM.from_pretrained(tiny_model_dir, dtype=torch.float32).eval()
    for step in steps:
        with torch.no_grad():
            for name, parameter in policy.named_parameters():
                parameter.copy_(step.params[name])
        parts, rewards = direct_gradients(policy, ref_model, trainer.projector, trainer.tokenizer, cfg, step)
        assert step.rewards is not None
        torch.testing.assert_close(step.rewards.cpu(), rewards, rtol=0.0, atol=1e-6)
        expected = {name: torch.stack([part[name] for part in parts.values()]).sum(dim=0) for name in step.grads}
        scale = max(float(grad.abs().max()) for grad in expected.values())
        errors = {name: float((step.grads[name] - expected[name]).abs().max()) / scale for name in expected}
        worst = max(errors, key=errors.__getitem__)
        assert errors[worst] < 1e-5, f"epoch {step.epoch}: {worst} differs by {errors[worst]:.2e} of the largest entry"
        for part, grads in parts.items():
            assert max(float(grad.abs().max()) for grad in grads.values()) / scale > 1e-4, part


def test_anchors_are_retrieved_from_the_second_epoch_and_give_the_partial_reward_its_similarity(
    tiny_model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = run_config(tmp_path, tiny_model_dir, "training.epochs=2", "training.max_steps=2")
    assert (cfg.arix.retrieval_start_epoch, cfg.arix.bank_size) == (2, 3)
    trainer = make_trainer(cfg, AcceptEverything())
    retrievals: list[tuple[int, str, Tensor, Tensor, Tensor]] = []
    scored: list[tuple[int, list[str], Tensor, dict[str, Tensor]]] = []
    similarities: dict[int, list[Tensor]] = {}
    retrieve, partial_rewards, signals = ReferenceBank.retrieve, trainer_module.partial_rewards, ARIXTrainer._signals

    def on_retrieve(bank: ReferenceBank, rollout_emb: Tensor, entry_emb: Tensor, *args: Any) -> Tensor:
        anchors = retrieve(bank, rollout_emb, entry_emb, *args)
        uid = next(uid for uid, owned in trainer.banks.banks.items() if owned is bank)
        retrievals.append((trainer.epoch, uid, rollout_emb.clone(), entry_emb.clone(), anchors.clone()))
        return anchors

    def on_partial_rewards(correct: Tensor, anchor_cos: Tensor, *args: Any) -> Tensor:
        similarities.setdefault(trainer.epoch, []).append(anchor_cos.clone())
        return partial_rewards(correct, anchor_cos, *args)

    def on_signals(self: ARIXTrainer, rollouts: StepRollouts, micro_batches: Any, emb: Tensor, epoch: int) -> Any:
        result = signals(self, rollouts, micro_batches, emb, epoch)
        entry_emb = {uid: value.clone() for uid, value in result.entry_emb.items()}
        scored.append((epoch, [ex.uid for ex in rollouts.examples], emb.clone(), entry_emb))
        return result

    monkeypatch.setattr(ReferenceBank, "retrieve", on_retrieve)
    monkeypatch.setattr(trainer_module, "partial_rewards", on_partial_rewards)
    monkeypatch.setattr(ARIXTrainer, "_signals", on_signals)
    trainer.train()

    assert sorted((epoch, uid) for epoch, uid, *_ in retrievals) == [(2, ex.uid) for ex in EXAMPLES]
    assert all(len(entry_emb) == cfg.arix.bank_size for _, _, _, entry_emb, _ in retrievals)
    retrieved = {uid: (rollout_emb, entry_emb, anchors) for _, uid, rollout_emb, entry_emb, anchors in retrievals}
    assert [epoch for epoch, *_ in scored] == [1, 2] and sorted(similarities) == [1, 2]
    moved = 0.0
    for epoch, uids, emb, entry_emb in scored:
        expected = []
        for p, uid in enumerate(uids):
            rows = emb[p * GROUP_SIZE : (p + 1) * GROUP_SIZE]
            anchors = torch.zeros(GROUP_SIZE, dtype=torch.long)
            if epoch == 2:
                rollout_emb, bank_emb, anchors = retrieved[uid]
                assert torch.equal(rollout_emb, rows) and torch.equal(bank_emb, entry_emb[uid])
            cosine = F.cosine_similarity(rows, entry_emb[uid][anchors], dim=-1)
            moved = max(moved, float((cosine - F.cosine_similarity(rows, entry_emb[uid][:1], dim=-1)).abs().max()))
            expected.append(cosine)
        torch.testing.assert_close(torch.cat(similarities[epoch]), torch.cat(expected), rtol=0.0, atol=1e-6)
    assert moved > 1e-3


def test_logged_grpo_loss_is_the_mean_of_l_grpo_over_the_prompts(
    tiny_model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values: list[Tensor] = []

    def offset_values(*args: Any, **kwargs: Any) -> Tensor:
        seq = grpo_sequence_losses(*args, **kwargs)
        seq = seq - torch.arange(1, len(seq) + 1, dtype=seq.dtype)
        values.append(seq.detach().clone())
        return seq

    monkeypatch.setattr(trainer_module, "grpo_sequence_losses", offset_values)
    cfg = run_config(tmp_path, tiny_model_dir)
    make_trainer(cfg, AcceptEveryOther()).train()
    (record,) = read_log(cfg)
    seq = torch.cat(values)
    assert [len(v) for v in values] == [3, 3, 2] and len(seq) == NUM_PROMPTS * GROUP_SIZE
    l_grpo = [-float(seq[p * GROUP_SIZE : (p + 1) * GROUP_SIZE].sum()) / GROUP_SIZE for p in range(NUM_PROMPTS)]
    assert record["grpo_loss"] == pytest.approx(sum(l_grpo) / NUM_PROMPTS, rel=1e-6)
    assert record["grpo_loss"] > 1.0
    assert record["pos_loss"] > 0.0
    assert record["loss"] == pytest.approx(record["grpo_loss"] + cfg.arix.alpha_pos * record["pos_loss"], rel=1e-9)


@pytest.mark.parametrize("ppo_epochs", [1, 2])
def test_signals_come_first_once_per_step_and_the_bank_update_last(
    ppo_epochs: int, tiny_model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[str, Any]] = []
    signals, gradient_pass, train_step = ARIXTrainer._signals, ARIXTrainer._gradient_pass, ARIXTrainer._train_step
    retrieve, update, loss_grads = ReferenceBank.retrieve, ReferenceBank.update, trainer_module.embedding_loss_grads

    def on_signals(self: ARIXTrainer, *args: Any) -> Any:
        events.append(("signals", None))
        return signals(self, *args)

    def on_gradient_pass(self: ARIXTrainer, micro_batches: list[Any]) -> Any:
        events.append(("gradient_pass", [(mb.advantages.clone(), mb.emb_grad.clone()) for mb in micro_batches]))
        return gradient_pass(self, micro_batches)

    def on_retrieve(self: ReferenceBank, *args: Any) -> Any:
        events.append(("retrieve", len(self.entries)))
        return retrieve(self, *args)

    def on_update(self: ReferenceBank, *args: Any) -> Any:
        events.append(("update", None))
        return update(self, *args)

    def on_loss_grads(rollout_emb: Tensor, anchor_emb: Tensor, *args: Any) -> Any:
        events.append(("contrastive", (rollout_emb.clone(), anchor_emb.clone())))
        return loss_grads(rollout_emb, anchor_emb, *args)

    def on_train_step(self: ARIXTrainer, *args: Any) -> Any:
        record = train_step(self, *args)
        events.append(("end of step", None))
        return record

    monkeypatch.setattr(ARIXTrainer, "_signals", on_signals)
    monkeypatch.setattr(ARIXTrainer, "_gradient_pass", on_gradient_pass)
    monkeypatch.setattr(ARIXTrainer, "_train_step", on_train_step)
    monkeypatch.setattr(ReferenceBank, "retrieve", on_retrieve)
    monkeypatch.setattr(ReferenceBank, "update", on_update)
    monkeypatch.setattr(trainer_module, "embedding_loss_grads", on_loss_grads)
    overrides = [f"training.ppo_epochs={ppo_epochs}", "training.epochs=2", "training.max_steps=2"]
    overrides += ["training.learning_rate=1e-3", "arix.bank_size=5", "arix.retrieval_start_epoch=1"]
    cfg = run_config(tmp_path, tiny_model_dir, *overrides)
    make_trainer(cfg, AcceptEverything()).train()

    steps: list[list[tuple[str, Any]]] = [[]]
    for kind, payload in events:
        if kind == "end of step":
            steps.append([])
        else:
            steps[-1].append((kind, payload))
    assert steps.pop() == [] and len(steps) == 2
    for step, bank_size in zip(steps, (1, 5), strict=True):
        kinds = [kind for kind, _ in step]
        passes = [i for i, kind in enumerate(kinds) if kind == "gradient_pass"]
        updates = [i for i, kind in enumerate(kinds) if kind == "update"]
        retrievals = [i for i, kind in enumerate(kinds) if kind == "retrieve"]
        assert kinds[0] == "signals" and kinds.count("signals") == 1
        assert len(passes) == ppo_epochs and len(retrievals) == NUM_PROMPTS and max(retrievals) < passes[0]
        assert len(updates) == NUM_PROMPTS * GROUP_SIZE and min(updates) > passes[-1]
        assert {payload for kind, payload in step if kind == "retrieve"} == {bank_size}
        first, *later = [payload for kind, payload in step if kind == "gradient_pass"]
        for state in later:
            for (advantages, grads), (first_advantages, first_grads) in zip(state, first, strict=True):
                assert torch.equal(advantages, first_advantages)
                assert not torch.equal(grads, first_grads)
        groups = [payload for kind, payload in step if kind == "contrastive"]
        assert len(groups) == NUM_PROMPTS * ppo_epochs
        for p in range(NUM_PROMPTS):
            (first_emb, first_anchors), *repeats = groups[p::NUM_PROMPTS]
            for emb, anchors in repeats:
                assert torch.equal(anchors, first_anchors)
                assert not torch.equal(emb, first_emb)
    first_record, second_record = read_log(cfg)
    assert first_record["pinned_anchor_fraction"] == 1.0 and first_record["bank_inserted"] == NUM_PROMPTS * GROUP_SIZE
    assert second_record["bank_inserted"] == 0


def test_only_the_last_micro_batch_of_a_step_synchronizes_the_gradients(
    tiny_model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[bool | str] = []
    set_gradient_sync = trainer_module._set_gradient_sync

    def on_set_gradient_sync(accelerator: Accelerator, sync: bool) -> None:
        events.append(sync)
        set_gradient_sync(accelerator, sync)

    monkeypatch.setattr(trainer_module, "_set_gradient_sync", on_set_gradient_sync)
    cfg = run_config(tmp_path, tiny_model_dir, f"training.micro_batch_size={GROUP_SIZE}")
    trainer = make_trainer(cfg, AcceptEveryOther())
    no_sync = trainer.accelerator.no_sync

    def on_no_sync(model: nn.Module) -> Any:
        events.append("no_sync")
        return no_sync(model)

    monkeypatch.setattr(trainer.accelerator, "no_sync", on_no_sync)
    trainer.train()
    assert events == [False, "no_sync", True]


class SimulatedProcess(Accelerator):
    def __init__(self, rank: int, world_size: int) -> None:
        self.simulated = (rank, world_size)
        super().__init__(cpu=True)

    @property
    def num_processes(self) -> int:
        return self.simulated[1]

    @property
    def process_index(self) -> int:
        return self.simulated[0]


def numbered_examples(count: int) -> list[Example]:
    prompts = {i: f"Question:\n{i} + 3?\nAnswer reasoning:\n" for i in range(count)}
    return [Example(f"u{i:02d}", prompts[i], f"x = {i} + 3", i + 3) for i in reversed(range(count))]


@pytest.mark.parametrize(
    ("count", "batch_size", "world_size"),
    [(100, 32, 6), (96, 32, 6), (31, 32, 6), (70, 32, 4), (7, 4, 2), (50, 5, 3), (10, 3, 2), (9, 7, 7)],
)
def test_the_shards_are_proportional_to_the_prompts_per_step(count: int, batch_size: int, world_size: int) -> None:
    examples = numbered_examples(count)
    ordered = sorted(ex.uid for ex in examples)
    per_step = [_prompts_per_process(batch_size, world_size, rank) for rank in range(world_size)]
    assert sum(per_step) == batch_size and per_step[0] == math.ceil(batch_size / world_size)
    assert per_step == sorted(per_step, reverse=True) and per_step[0] - per_step[-1] <= 1
    shards = [[ex.uid for ex in _shard(examples, rank, world_size, batch_size)] for rank in range(world_size)]
    assert sorted(uid for shard in shards for uid in shard) == ordered
    blocks = [set(ordered[start : start + batch_size]) for start in range(0, count, batch_size)]
    for shard, n_r in zip(shards, per_step, strict=True):
        assert shard == sorted(shard)
        assert [len(block.intersection(shard)) for block in blocks if len(block) == batch_size] == [n_r] * (
            count // batch_size
        )
        assert len(shard) <= math.ceil(count / batch_size) * n_r
    if batch_size % world_size == 0:
        assert shards == [ordered[rank::world_size] for rank in range(world_size)]


def test_32_prompts_over_6_processes() -> None:
    assert [_prompts_per_process(32, 6, rank) for rank in range(6)] == [6, 6, 5, 5, 5, 5]
    examples = numbered_examples(100)
    assert [len(_shard(examples, rank, 6, 32)) for rank in range(6)] == [19, 19, 16, 16, 15, 15]
    with pytest.raises(ValueError, match="a minibatch of 5 prompts cannot be shared by 6 processes"):
        _shard(examples, 0, 6, 5)
    with pytest.raises(ValueError, match="5 training examples cannot be sharded over 6 processes"):
        _shard(examples[:5], 0, 6, 32)
    with pytest.raises(ValueError, match="unique"):
        _shard([*examples, examples[0]], 0, 6, 32)


def test_the_minibatch_need_not_be_a_multiple_of_the_number_of_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    required = ["model.name_or_path=base", f"output.output_dir={tmp_path / 'run'}"]
    monkeypatch.setenv("WORLD_SIZE", "6")
    cfg = load_config(None, required)
    assert cfg.training.batch_size == 32
    for world_size in (1, 4, 6, 32):
        cfg.validate(world_size=world_size)
    with pytest.raises(
        ValueError, match=r"training\.batch_size \(32\) must be at least the number of processes \(33\)"
    ):
        cfg.validate(world_size=33)
    assert load_config(None, [*required, "training.batch_size=6"]).training.batch_size == 6
    with pytest.raises(ValueError, match=r"training\.batch_size \(5\) must be at least the number of processes \(6\)"):
        load_config(None, [*required, "training.batch_size=5"])


@pytest.mark.parametrize(("count", "batch_size", "world_size"), [(15, 5, 3), (16, 6, 4), (15, 4, 2)])
def test_every_process_takes_the_same_steps_with_its_share_of_the_minibatch(
    count: int, batch_size: int, world_size: int, tiny_model_dir: Path, tmp_path: Path
) -> None:
    cfg = run_config(tmp_path, tiny_model_dir, f"training.batch_size={batch_size}")
    steps = math.ceil(count / batch_size)
    per_step = [_prompts_per_process(batch_size, world_size, rank) for rank in range(world_size)]
    for rank in range(world_size):
        accelerator = SimulatedProcess(rank, world_size)
        trainer = make_trainer(cfg, AcceptEverything(), examples=numbered_examples(count), accelerator=accelerator)
        assert (trainer.rank, trainer.world_size) == (rank, world_size)
        assert trainer.steps_per_epoch == steps and trainer.prompts_per_step == per_step[rank]
        assert trainer.max_prompts_per_step == per_step[0]
        assert trainer.settings == LossSettings.from_config(cfg, batch_size / world_size)
        assert trainer._trainer_state()["prompts_per_process_per_step"] == per_step
        shard = [ex.uid for ex in trainer.examples]
        uids = [ex.uid for ex in trainer._epoch_order(1)]
        assert len(uids) == steps * per_step[rank]
        assert sorted(uids[: len(shard)]) == shard and uids[len(shard) :] == uids[: len(uids) - len(shard)]
    if batch_size % world_size == 0:
        assert steps == math.ceil(math.ceil(count / world_size) / (batch_size // world_size))
    for total in range(1, 100):
        for processes in range(1, 9):
            for minibatch in range(processes, 65, processes):
                assert math.ceil(total / minibatch) == math.ceil(
                    math.ceil(total / processes) / (minibatch // processes)
                )


def test_a_process_steps_through_its_shard_with_its_share_of_the_minibatch(
    tiny_model_dir: Path, tmp_path: Path
) -> None:

    def second_process(cfg: ARIXConfig) -> ARIXTrainer:
        accelerator = SimulatedProcess(1, 2)
        return make_trainer(cfg, AcceptEverything(), examples=numbered_examples(15), accelerator=accelerator)

    cfg = run_config(tmp_path, tiny_model_dir, "training.batch_size=4")
    trainer = second_process(cfg)
    shard = [f"u{i:02d}" for i in range(1, 15, 2)]
    assert (trainer.rank, trainer.world_size) == (1, 2) and [ex.uid for ex in trainer.examples] == shard
    assert trainer.settings.num_prompts == trainer.prompts_per_step == cfg.training.batch_size // trainer.world_size
    assert trainer.settings == LossSettings.from_config(cfg, 2)
    assert trainer.steps_per_epoch == 4

    def epoch_order(process: ARIXTrainer, epoch: int) -> list[str]:
        uids = [ex.uid for ex in process._epoch_order(epoch)]
        assert len(uids) == process.steps_per_epoch * process.prompts_per_step == 8
        assert sorted(uids[: len(shard)]) == shard and uids[len(shard) :] == uids[:1]
        return uids

    first = epoch_order(trainer, 1)
    assert epoch_order(trainer, 1) == first and epoch_order(trainer, 2) != first
    reseeded = second_process(run_config(tmp_path, tiny_model_dir, "training.batch_size=4", "training.seed=2222"))
    assert epoch_order(reseeded, 1) != first


def merged_step(steps: Sequence[CapturedStep]) -> CapturedStep:
    rollouts = [step.rollouts for step in steps]
    anchors = [step.anchors for step in steps if step.anchors is not None]
    rewards = [step.rewards for step in steps if step.rewards is not None]
    assert len(anchors) == len(rewards) == len(steps)
    return CapturedStep(
        rollouts=StepRollouts(
            examples=[example for part in rollouts for example in part.examples],
            responses=[response for part in rollouts for response in part.responses],
            texts=[text for part in rollouts for text in part.texts],
            verdicts=[verdict for part in rollouts for verdict in part.verdicts],
            correct=torch.cat([part.correct for part in rollouts]),
        ),
        epoch=steps[0].epoch,
        entries={uid: entries for step in steps for uid, entries in step.entries.items()},
        anchors=torch.cat(anchors),
        rewards=torch.cat(rewards),
        params=steps[0].params,
    )


@pytest.mark.parametrize("collective_backward", [False, True])
def test_the_averaged_gradient_of_uneven_processes_is_that_of_equation_12(
    collective_backward: bool, tiny_model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world_size, overrides = 2, ["training.batch_size=3"]
    cfg = run_config(tmp_path, tiny_model_dir, *overrides)
    ref_model = perturbed_reference_model(cfg)
    steps: list[CapturedStep] = []
    trainers: list[ARIXTrainer] = []
    for rank in range(world_size):
        trainer = make_trainer(
            run_config(tmp_path / f"rank{rank}", tiny_model_dir, *overrides),
            AcceptEveryOther(),
            examples=numbered_examples(3),
            ref_model=ref_model,
            accelerator=SimulatedProcess(rank, world_size),
        )
        trainer.collective_backward = collective_backward
        with monkeypatch.context() as patch:
            captured = capture_steps(trainer, patch)
            trainer.train()
        steps += captured
        trainers.append(trainer)

    assert [sorted(ex.uid for ex in step.rollouts.examples) for step in steps] == [["u00", "u02"], ["u01"]]
    assert all(torch.equal(steps[1].params[name], value) for name, value in steps[0].params.items())
    step = merged_step(steps)
    policy = AutoModelForCausalLM.from_pretrained(tiny_model_dir, dtype=torch.float32).eval()
    with torch.no_grad():
        for name, parameter in policy.named_parameters():
            parameter.copy_(step.params[name])
    parts, rewards = direct_gradients(policy, ref_model, trainers[0].projector, trainers[0].tokenizer, cfg, step)
    assert step.rewards is not None
    torch.testing.assert_close(step.rewards.cpu(), rewards, rtol=0.0, atol=1e-6)
    averaged = {name: sum(process.grads[name] for process in steps) / world_size for name in steps[0].grads}
    expected = {name: torch.stack([part[name] for part in parts.values()]).sum(dim=0) for name in averaged}
    scale = max(float(grad.abs().max()) for grad in expected.values())
    errors = {name: float((averaged[name] - expected[name]).abs().max()) / scale for name in expected}
    worst = max(errors, key=errors.__getitem__)
    assert errors[worst] < 1e-5, f"{worst} differs by {errors[worst]:.2e} of the largest entry"


@dataclass
class SimulatedStep:
    step: CapturedStep
    forwards: Counter[str]
    generated: list[tuple[list[list[int]], bool, list[list[int]]]]
    generate_calls: int
    syncs: list[bool]
    record: dict[str, Any]


COLLECTIVE_PASSES = ("_no_grad_pass", "_embed_entries", "_subset_pass", "_gradient_pass")


def simulate_first_step(
    cfg: ARIXConfig,
    rank: int,
    world_size: int,
    monkeypatch: pytest.MonkeyPatch,
    *,
    zero_stage: int,
    replay: StepRollouts | None = None,
) -> SimulatedStep:
    examples = numbered_examples(cfg.training.batch_size)
    trainer = make_trainer(cfg, AcceptEveryOther(), examples=examples, accelerator=SimulatedProcess(rank, world_size))
    trainer.zero3, trainer.collective_backward = _collective_flags(zero_stage)
    forwards: Counter[str] = Counter()
    generated: list[tuple[list[list[int]], bool, list[list[int]]]] = []
    generate_calls: list[None] = []
    syncs: list[bool] = []
    within: list[str] = []
    generate_batch, set_gradient_sync = trainer_module.generate_batch, trainer_module._set_gradient_sync
    policy_generate = trainer.policy.generate

    def labelled(name: str, method: Any) -> Any:
        def call(self: ARIXTrainer, *args: Any, **kwargs: Any) -> Any:
            within.append(name)
            try:
                return method(self, *args, **kwargs)
            finally:
                within.pop()

        return call

    def counter(label: str) -> Any:
        def hook(_module: nn.Module, _args: Any) -> None:
            if within:
                forwards[f"{within[-1]} {label}"] += 1

        return hook

    def single_process_generate(*args: Any, synced_gpus: bool = False, **kwargs: Any) -> list[list[int]]:
        responses = generate_batch(*args, **kwargs)
        generated.append(([list(prompt) for prompt in args[2]], synced_gpus, responses))
        return responses

    def counting_generate(*args: Any, **kwargs: Any) -> Any:
        generate_calls.append(None)
        return policy_generate(*args, **kwargs)

    def on_set_gradient_sync(accelerator: Accelerator, sync: bool) -> None:
        syncs.append(sync)
        set_gradient_sync(accelerator, sync)

    def replayed(_self: ARIXTrainer, batch: list[Example]) -> StepRollouts:
        assert replay is not None and [ex.uid for ex in batch] == [ex.uid for ex in replay.examples]
        return replay

    with monkeypatch.context() as patch:
        if replay is not None:
            patch.setattr(ARIXTrainer, "_sample", replayed)
        for name in COLLECTIVE_PASSES:
            patch.setattr(ARIXTrainer, name, labelled(name, getattr(ARIXTrainer, name)))
        patch.setattr(trainer_module, "generate_batch", single_process_generate)
        patch.setattr(trainer_module, "_set_gradient_sync", on_set_gradient_sync)
        patch.setattr(trainer.policy, "generate", counting_generate)
        hooks = [trainer.policy.register_forward_pre_hook(counter("policy"))]
        if trainer.ref_model is not None:
            hooks.append(trainer.ref_model.register_forward_pre_hook(counter("pi_ref")))
        captured = capture_steps(trainer, patch)
        trainer.train()
        for hook in hooks:
            hook.remove()
    (step,) = captured
    (record,) = read_log(cfg)
    record = {key: value for key, value in record.items() if not key.startswith("time_") and key != "elapsed"}
    return SimulatedStep(step, forwards, generated, len(generate_calls), syncs, record)


def assert_same_step(padded: SimulatedStep, plain: SimulatedStep) -> None:
    assert padded.record.keys() == plain.record.keys()
    assert {key: value for key, value in padded.record.items() if value is not None} == pytest.approx(
        {key: value for key, value in plain.record.items() if value is not None}, rel=1e-6, abs=1e-9
    )
    assert padded.step.rewards is not None and plain.step.rewards is not None
    torch.testing.assert_close(padded.step.rewards, plain.step.rewards, rtol=0.0, atol=1e-6)
    scale = max(float(grad.abs().max()) for grad in plain.step.grads.values())
    errors = {
        name: float((grad - plain.step.grads[name]).abs().max()) / scale for name, grad in padded.step.grads.items()
    }
    assert max(errors.values()) < 1e-6, errors


@pytest.mark.parametrize(
    ("zero_stage", "flags"), [(0, (False, False)), (1, (False, False)), (2, (False, True)), (3, (True, True))]
)
def test_the_zero_stage_sets_which_passes_are_collective(zero_stage: int, flags: tuple[bool, bool]) -> None:
    assert _collective_flags(zero_stage) == flags


def test_under_zero3_every_process_runs_the_collective_passes_of_the_busiest_one(
    tiny_model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    overrides = ["training.batch_size=5", "training.generation_batch_size=4", "generation.ds3_gather=false"]
    overrides.append("arix.use_negative=true")
    world_size, group = 3, GROUP_SIZE
    padded = [
        simulate_first_step(
            run_config(tmp_path / f"rank{rank}", tiny_model_dir, *overrides),
            rank,
            world_size,
            monkeypatch,
            zero_stage=3,
        )
        for rank in range(world_size)
    ]
    passes = {"_no_grad_pass policy": 3, "_no_grad_pass pi_ref": 3, "_embed_entries policy": 2}
    passes.update({"_subset_pass policy": 3, "_gradient_pass policy": 3})
    for run, prompts in zip(padded, (2, 2, 1), strict=True):
        assert dict(run.forwards) == passes and run.syncs == [False, False, True] and run.generate_calls == 2
        ((generated, synced, responses),) = run.generated
        assert synced and len(generated) == 2 * group
        assert generated[prompts * group :] == [generated[prompts * group - 1]] * ((2 - prompts) * group)
        assert run.step.rollouts.responses == responses[: prompts * group]

    plain = simulate_first_step(
        run_config(tmp_path / "plain", tiny_model_dir, *overrides),
        2,
        world_size,
        monkeypatch,
        zero_stage=0,
        replay=padded[2].step.rollouts,
    )
    assert dict(plain.forwards) == {
        "_no_grad_pass policy": 2,
        "_no_grad_pass pi_ref": 2,
        "_embed_entries policy": 1,
        "_subset_pass policy": 2,
        "_gradient_pass policy": 2,
    }
    assert plain.syncs == [False, True] and plain.generated == [] and plain.generate_calls == 0
    assert_same_step(padded[2], plain)


def test_under_zero2_only_the_gradient_pass_is_padded(
    tiny_model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    overrides = ["training.batch_size=5", "training.generation_batch_size=4", "arix.use_negative=true"]
    rank, world_size = 2, 3
    padded = simulate_first_step(
        run_config(tmp_path / "zero2", tiny_model_dir, *overrides), rank, world_size, monkeypatch, zero_stage=2
    )
    unpadded = {"_no_grad_pass policy": 2, "_no_grad_pass pi_ref": 2, "_embed_entries policy": 1}
    unpadded["_subset_pass policy"] = 2
    assert dict(padded.forwards) == {**unpadded, "_gradient_pass policy": 3}
    assert padded.syncs == [False, False, True] and padded.generate_calls == 1
    ((generated, synced, responses),) = padded.generated
    assert not synced and len(generated) == GROUP_SIZE and padded.step.rollouts.responses == responses

    plain = simulate_first_step(
        run_config(tmp_path / "plain", tiny_model_dir, *overrides),
        rank,
        world_size,
        monkeypatch,
        zero_stage=0,
        replay=padded.step.rollouts,
    )
    assert dict(plain.forwards) == {**unpadded, "_gradient_pass policy": 2}
    assert plain.syncs == [False, True] and plain.generated == [] and plain.generate_calls == 0
    assert_same_step(padded, plain)


class _ListTokenizer:
    skip_special_tokens: bool | None = None

    def batch_decode(self, sequences: list[list[int]], skip_special_tokens: bool = False) -> list[str]:
        self.skip_special_tokens = skip_special_tokens
        return [" ".join(map(str, ids)) for ids in sequences]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return " ".join(map(str, ids)) + ("" if skip_special_tokens else " <special>")


def test_decode_for_verification_prefers_the_tasks_own_decoding() -> None:
    tokenizer = _ListTokenizer()
    responses = [[5, 6, 1], [7]]
    assert isinstance(AcceptEverything(), Task) and not isinstance(AcceptEverything(), ResponseDecoder)
    assert decode_for_verification(AcceptEverything(), tokenizer, responses) == ["5 6 1", "7"]  # type: ignore[arg-type]
    assert tokenizer.skip_special_tokens is True
    task = DecodingTask()
    assert isinstance(task, Task) and isinstance(task, ResponseDecoder)
    texts = decode_for_verification(task, tokenizer, [tuple(ids) for ids in responses])  # type: ignore[arg-type]
    assert texts == ["[3 tokens] 5 6 1 <special>", "[1 tokens] 7 <special>"] and task.decoded == [responses]

    class Truncating(AcceptEverything):
        def decode_responses(self, tokenizer: Any, responses: list[list[int]]) -> list[str]:
            return ["only one"]

    with pytest.raises(ValueError, match="2 responses"):
        decode_for_verification(Truncating(), tokenizer, responses)  # type: ignore[arg-type]


def test_the_trainer_verifies_the_texts_of_the_tasks_decoding(tiny_model_dir: Path, tmp_path: Path) -> None:
    task = DecodingTask()
    trainer = make_trainer(run_config(tmp_path, tiny_model_dir), task)
    trainer.train()
    (responses,) = task.decoded
    assert len(responses) == NUM_PROMPTS * GROUP_SIZE
    expected = [f"[{len(ids)} tokens] " + trainer.tokenizer.decode(ids, skip_special_tokens=False) for ids in responses]
    assert task.verified == expected
    admitted = [entry for bank in trainer.banks.banks.values() for entry in bank.entries[1:]]
    assert admitted and all(entry.text == expected[responses.index(entry.token_ids)] for entry in admitted)


def test_an_output_dir_holding_a_run_is_refused_unless_overwrite(tiny_model_dir: Path, tmp_path: Path) -> None:
    cfg = run_config(tmp_path, tiny_model_dir)
    run = Path(cfg.output.output_dir)
    run.mkdir(parents=True)
    (run / "data_summary.json").write_text("{}", encoding="utf-8")
    check_output_dir(cfg.output)
    make_trainer(cfg, AcceptEverything()).train()
    assert [record["step"] for record in read_log(cfg)] == [1]
    with pytest.raises(FileExistsError, match=r"train_log\.jsonl, final"):
        make_trainer(cfg, AcceptEverything())
    (run / TRAIN_LOG).unlink()
    with pytest.raises(FileExistsError, match="final"):
        check_output_dir(cfg.output)
    (run / TRAIN_LOG).write_text('{"step": 1}\n{"step": 2}\n', encoding="utf-8")
    cfg.output.overwrite = True
    make_trainer(cfg, AcceptEverything()).train()
    assert [record["step"] for record in read_log(cfg)] == [1]
    assert run_config(tmp_path, tiny_model_dir).output.overwrite is False
    assert run_config(tmp_path, tiny_model_dir, "output.overwrite=true").output.overwrite is True


def test_the_loss_settings_are_read_from_the_configuration() -> None:
    training = {"clip_eps": 0.3, "kl_beta": 0.5, "kl_estimator": "full", "length_normalize": False, "group_size": 6}
    cfg = config_from_dict({"training": training, "generation": {"max_new_tokens": 17}, "arix": {"pooling": "mean"}})
    settings = LossSettings.from_config(cfg, 2.5)
    assert settings == LossSettings(
        clip_eps=0.3,
        kl_beta=0.5,
        kl_estimator="full",
        length_normalize=False,
        max_new_tokens=17,
        pooling="mean",
        group_size=6,
        num_prompts=2.5,
    )
    defaults = LossSettings.from_config(ARIXConfig(), 2.5)
    copied = [f.name for f in fields(LossSettings) if f.name != "num_prompts"]
    assert all(getattr(settings, name) != getattr(defaults, name) for name in copied)


def test_each_seed_draws_its_own_projector_and_zero_shot_evaluation_rebuilds_it(
    tiny_model_dir: Path, tmp_path: Path
) -> None:
    weights = {}
    for seed in (1111, 2222):
        cfg = run_config(tmp_path / str(seed), tiny_model_dir, f"training.seed={seed}")
        trainer = make_trainer(cfg, AcceptEverything())
        final = trainer.save_checkpoint("final")
        assert read_yaml_mapping(final / "config.yaml")["arix"]["proj_seed"] == seed
        saved = RationaleProjector.load(final / "projector.pt")
        weights[seed] = saved.linear.weight
        assert saved.seed == seed and torch.equal(trainer.projector.linear.weight.cpu(), saved.linear.weight)
        training_config = tmp_path / f"zero_shot_{seed}.yaml"
        training_config.write_text(yaml.safe_dump({"training": {"seed": seed}}), encoding="utf-8")
        zero_shot = read_run_config(str(tiny_model_dir), str(training_config))
        rebuilt = load_projector(str(tiny_model_dir), None, zero_shot.arix, saved.hidden_size)
        assert torch.equal(rebuilt.linear.weight, saved.linear.weight)
    assert not torch.equal(weights[1111], weights[2222])
    explicit = run_config(tmp_path / "explicit", tiny_model_dir, "training.seed=2222", "arix.proj_seed=1111")
    assert torch.equal(make_trainer(explicit, AcceptEverything()).projector.linear.weight.cpu(), weights[1111])


def test_the_evaluation_section_loads_round_trips_and_is_validated(tmp_path: Path) -> None:
    assert ARIXConfig().evaluation == EvaluationConfig(batch_size=None, embed_batch_size=None)
    path = tmp_path / "config.yaml"
    raw = {"model": {"name_or_path": "base"}, "output": {"output_dir": "run"}, "evaluation": {"batch_size": 4}}
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(path, ["evaluation.embed_batch_size=2"])
    assert cfg.evaluation == EvaluationConfig(batch_size=4, embed_batch_size=2)
    assert load_config(path, ["evaluation.batch_size=null"]).evaluation.batch_size is None
    save_config(cfg, tmp_path / "saved.yaml")
    saved = read_yaml_mapping(tmp_path / "saved.yaml")
    assert saved["evaluation"] == {"batch_size": 4, "embed_batch_size": 2} and config_from_dict(saved) == cfg
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(yaml.safe_dump({"evaluation": {"embed_batch_size": 1}}), encoding="utf-8")
    merged = load_run_config([path, overlay], ["output.output_dir=elsewhere"]).evaluation
    assert merged == EvaluationConfig(batch_size=4, embed_batch_size=1)
    for key in ("batch_size", "embed_batch_size"):
        with pytest.raises(ValueError, match=f"evaluation.{key} must be >= 1"):
            load_config(path, [f"evaluation.{key}=0"])
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(path, ["evaluation.max_new_tokens=8"])


@pytest.mark.parametrize(
    "override", ["generation.temperature=0.6", "generation.temperature=0", "generation.top_p=0.9", "generation.top_k=5"]
)
def test_training_samples_from_the_untempered_policy_only(override: str, tmp_path: Path) -> None:
    required = ["model.name_or_path=base", f"output.output_dir={tmp_path / 'run'}"]
    sampling = load_config(None, required).generation
    assert (sampling.temperature, sampling.top_p, sampling.top_k) == (1.0, 1.0, 0)
    key, _, value = override.partition("=")
    with pytest.raises(ValueError, match=rf"{re.escape(key)} must be \d for training: rollouts must be drawn"):
        load_config(None, [*required, override])
    evaluation = tmp_path / "config.yaml"
    generation = {key.partition(".")[2]: yaml.safe_load(value)}
    evaluation.write_text(yaml.safe_dump({"generation": generation}), encoding="utf-8")
    assert read_run_config("base", str(evaluation)) is not None
