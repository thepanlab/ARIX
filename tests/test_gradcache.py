from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from tiny_architectures import FAMILIES, tiny_causal_lm
from torch import Tensor
from transformers import PreTrainedModel

import arix.modeling as modeling
import arix.trainer as trainer_module
from arix.embedding import RationaleProjector, lcs_mask, masked_subset_embedding, rationale_embedding
from arix.losses import (
    embedding_loss_grads,
    group_advantages,
    grpo_sequence_losses,
    info_nce_negative,
    info_nce_positive,
    partial_rewards,
)
from arix.modeling import (
    SequenceBatch,
    build_sequence_batch,
    disable_dropout,
    enable_gradient_checkpointing,
)
from arix.trainer import (
    LossSettings,
    MicroBatch,
    _set_gradient_sync,
    lcs_subset_masks,
    micro_batch_loss,
    score_micro_batch,
    subset_embeddings,
    subset_gradients,
)

VOCAB, PROJ_DIM = 64, 16
GROUP_SIZE, MAX_NEW_TOKENS, PAD_ID = 6, 9, 0
LOGPROB_CHUNK = 5
CLIP_EPS, KL_BETA = 0.2, 0.3
TAU, ALPHA_POS, ALPHA_NEG = 0.2, 1.0, 0.5
CORRECT = ([True, False, True, False, False, True], [False, True, False, False, True, False])
ANCHORS = ([0, 1, 2, 0, 1, 2], [2, 0, 1, 1, 0, 2])


@dataclass
class Group:
    prompt: list[int]
    responses: list[list[int]]
    correct: list[bool]
    entries: list[list[int]]
    anchors: list[int]
    old_shift: list[Tensor]


@dataclass
class Case:
    kl_estimator: str
    pooling: str
    length_normalize: bool
    use_negative: bool
    old_noise: float
    micro_batch_size: int


CASES = {
    "manuscript": Case("k3", "softmax", True, False, 0.0, 4),
    "clipped_ratio": Case("k3", "softmax", True, False, 0.15, 5),
    "full_kl_mean_pooling_dr_grpo": Case("full", "mean", False, False, 0.0, 4),
    "negative_loss": Case("k3", "softmax", True, True, 0.0, 4),
}


def tiny_model(family: str, seed: int) -> PreTrainedModel:
    model = tiny_causal_lm(family, VOCAB, seed=seed, dtype=torch.float64)
    disable_dropout(model)
    return model


def reference_forward(model: PreTrainedModel, batch: SequenceBatch) -> tuple[Tensor, Tensor, Tensor]:
    out = model(
        input_ids=batch.input_ids, attention_mask=batch.attention_mask, output_hidden_states=True, use_cache=False
    )
    logp = torch.log_softmax(out.logits[:, :-1].float(), dim=-1).gather(-1, batch.input_ids[:, 1:, None]).squeeze(-1)
    return F.pad(logp, (1, 0)).masked_fill(~batch.response_mask, 0.0), out.hidden_states[-1], out.logits


def reference_kl(policy_logits: Tensor, ref_logits: Tensor, mask: Tensor) -> Tensor:
    log_p = F.log_softmax(policy_logits[:, :-1].float(), dim=-1)
    log_q = F.log_softmax(ref_logits[:, :-1].float(), dim=-1)
    kl = (log_p.exp() * (log_p - log_q)).sum(dim=-1)
    return F.pad(kl, (1, 0)).masked_fill(~mask, 0.0)


def make_groups(old_noise: float) -> list[Group]:
    generator = torch.Generator().manual_seed(1234)

    def tokens(low: int, high: int) -> list[int]:
        length = int(torch.randint(low, high + 1, (1,), generator=generator))
        return torch.randint(1, 12, (length,), generator=generator).tolist()

    groups = []
    for correct, anchors in zip(CORRECT, ANCHORS, strict=True):
        entries = [tokens(3, MAX_NEW_TOKENS) for _ in range(3)]
        responses = [tokens(2, MAX_NEW_TOKENS) for _ in range(GROUP_SIZE)]
        shifts = [old_noise * torch.randn(len(r), generator=generator) for r in responses]
        groups.append(Group(tokens(3, 5), responses, list(correct), entries, list(anchors), shifts))
    groups[0].responses[3] = list(groups[0].entries[0])
    groups[0].old_shift[3] = old_noise * torch.randn(len(groups[0].responses[3]), generator=generator)
    return groups


def place(batch: SequenceBatch, values: list[Tensor]) -> Tensor:
    out = torch.zeros(batch.input_ids.shape, dtype=torch.float32)
    for row, value in enumerate(values):
        start = batch.prompt_lens[row]
        out[row, start : start + len(value)] = value
    return out


def settings_for(case: Case, num_prompts: int) -> LossSettings:
    return LossSettings(
        clip_eps=CLIP_EPS,
        kl_beta=KL_BETA,
        kl_estimator=case.kl_estimator,
        length_normalize=case.length_normalize,
        max_new_tokens=MAX_NEW_TOKENS,
        pooling=case.pooling,
        group_size=GROUP_SIZE,
        num_prompts=num_prompts,
    )


def anchor_embeddings(
    policy: PreTrainedModel, projector: RationaleProjector, groups: list[Group], pooling: str
) -> Tensor:
    rows = []
    for group in groups:
        batch = build_sequence_batch([group.prompt] * len(group.entries), group.entries, PAD_ID)
        _, _, entry_emb = score_micro_batch(policy, None, batch, projector, pooling, need_emb=True)
        rows.append(entry_emb[torch.tensor(group.anchors)])
    return torch.cat(rows)


def parameter_grads(model: torch.nn.Module) -> dict[str, Tensor]:
    return {name: p.grad.detach().clone() for name, p in model.named_parameters()}


def two_pass_gradients(
    policy: PreTrainedModel,
    ref: PreTrainedModel,
    projector: RationaleProjector,
    groups: list[Group],
    case: Case,
    drop: str | None = None,
) -> tuple[dict[str, Tensor], Tensor, Tensor]:
    settings = settings_for(case, len(groups))
    flat = [(p, j) for p in range(len(groups)) for j in range(GROUP_SIZE)]
    micro_batches = []
    for start in range(0, len(flat), case.micro_batch_size):
        index = list(range(start, min(start + case.micro_batch_size, len(flat))))
        prompts = [groups[flat[i][0]].prompt for i in index]
        batch = build_sequence_batch(prompts, [groups[flat[i][0]].responses[flat[i][1]] for i in index], PAD_ID)
        micro_batches.append(MicroBatch(batch=batch, index=index))

    policy.eval()
    ref_for_k3 = ref if case.kl_estimator == "k3" else None
    emb_rows = []
    for mb in micro_batches:
        old_logp, mb.ref_logp, emb = score_micro_batch(policy, ref_for_k3, mb.batch, projector, case.pooling, True)
        mb.old_logp = old_logp - place(mb.batch, [groups[flat[i][0]].old_shift[flat[i][1]] for i in mb.index])
        emb_rows.append(emb)
    emb = torch.cat(emb_rows)
    anchor_emb = anchor_embeddings(policy, projector, groups, case.pooling)
    correct = torch.tensor([c for group in groups for c in group.correct])
    subset_emb = None
    if case.use_negative:
        for mb in micro_batches:
            members = [flat[i] for i in mb.index]
            anchors = [None if groups[p].correct[j] else groups[p].entries[groups[p].anchors[j]] for p, j in members]
            responses = [groups[p].responses[j] for p, j in members]
            mb.subset_masks = lcs_subset_masks(mb.batch, responses, anchors)
        parts = [subset_embeddings(policy, mb.batch, mb.subset_masks, projector, case.pooling) for mb in micro_batches]
        subset_emb = (torch.cat([lcs for lcs, _ in parts]), torch.cat([exc for _, exc in parts]))

    advantages = torch.zeros(len(flat))
    grads = torch.zeros(3, len(flat), emb.shape[1])
    for p in range(len(groups)):
        rows = slice(p * GROUP_SIZE, (p + 1) * GROUP_SIZE)
        cosine = (emb[rows] * anchor_emb[rows]).sum(dim=-1)
        rewards = partial_rewards(correct[rows], cosine, True, 0.1, 0.2)
        advantages[rows] = group_advantages(rewards, 1e-4, True)
        neg = None
        if subset_emb is not None:
            neg = {
                "lcs_emb": subset_emb[0][rows],
                "exc_emb": subset_emb[1][rows],
                "incorrect": ~correct[rows],
                "alpha_neg": ALPHA_NEG,
            }
        _, g = embedding_loss_grads(emb[rows], anchor_emb[rows], correct[rows], TAU, ALPHA_POS, neg)
        if neg is None:
            grads[0, rows] = g
        else:
            grads[:, rows] = g
    for mb in micro_batches:
        mb.advantages = advantages[mb.index]
        if drop != "positive":
            mb.emb_grad = grads[0, mb.index]
        if mb.subset_masks is not None and drop != "negative":
            mb.subset_grads = subset_gradients(mb.subset_masks, grads[1, mb.index], grads[2, mb.index])

    policy.train()
    policy.zero_grad(set_to_none=True)
    for mb in micro_batches:
        loss, _ = micro_batch_loss(policy, ref, mb, projector, settings)
        loss.backward()
    return parameter_grads(policy), advantages, anchor_emb


def direct_gradients(
    policy: PreTrainedModel,
    ref: PreTrainedModel,
    projector: RationaleProjector,
    groups: list[Group],
    case: Case,
    advantages: Tensor,
    anchor_emb: Tensor,
) -> dict[str, Tensor]:
    policy.eval()
    policy.zero_grad(set_to_none=True)
    full = case.kl_estimator == "full"
    total = torch.zeros((), dtype=torch.float32)
    for p, group in enumerate(groups):
        rows = slice(p * GROUP_SIZE, (p + 1) * GROUP_SIZE)
        batch = build_sequence_batch([group.prompt] * GROUP_SIZE, group.responses, PAD_ID)
        mask = batch.response_mask
        logp, hidden, logits = reference_forward(policy, batch)
        with torch.no_grad():
            ref_logp, _, ref_logits = reference_forward(ref, batch)
        kl_full = reference_kl(logits, ref_logits, mask) if full else None
        old_logp = logp.detach() - place(batch, group.old_shift)
        seq = grpo_sequence_losses(
            logp,
            old_logp,
            None if full else ref_logp,
            advantages[rows],
            mask,
            clip_eps=CLIP_EPS,
            beta=KL_BETA,
            kl_estimator=case.kl_estimator,
            length_normalize=case.length_normalize,
            max_new_tokens=MAX_NEW_TOKENS,
            kl_full=kl_full,
        )
        correct = torch.tensor(group.correct)
        emb = rationale_embedding(hidden, logp, mask, projector, case.pooling)
        loss = -seq.mean() + ALPHA_POS * info_nce_positive(emb, anchor_emb[rows], correct, TAU)
        if case.use_negative:
            lcs = torch.zeros_like(mask)
            for j, (response, anchor) in enumerate(zip(group.responses, group.anchors, strict=True)):
                if not group.correct[j]:
                    start = batch.prompt_lens[j]
                    lcs[j, start : start + len(response)] = torch.tensor(lcs_mask(response, group.entries[anchor]))
            exc = mask & ~lcs & ~correct.unsqueeze(1)
            lcs_emb = masked_subset_embedding(hidden, logp, mask, lcs, projector, case.pooling)
            exc_emb = masked_subset_embedding(hidden, logp, mask, exc, projector, case.pooling)
            loss = loss + ALPHA_NEG * info_nce_negative(lcs_emb, exc_emb, anchor_emb[rows], ~correct, TAU)
        total = total + loss
    (total / len(groups)).backward()
    return parameter_grads(policy)


@pytest.fixture(scope="module", params=FAMILIES)
def models(request: pytest.FixtureRequest) -> tuple[PreTrainedModel, PreTrainedModel, RationaleProjector]:
    policy = tiny_model(request.param, seed=0)
    enable_gradient_checkpointing(policy)
    ref = tiny_model(request.param, seed=1).requires_grad_(False).eval()
    return policy, ref, RationaleProjector(policy.config.hidden_size, PROJ_DIM, seed=0)


@pytest.fixture(autouse=True)
def small_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(modeling, "LOGPROB_CHUNK", LOGPROB_CHUNK)
    monkeypatch.setattr(trainer_module, "FULL_KL_CHUNK", LOGPROB_CHUNK + 2)


def max_relative_error(actual: dict[str, Tensor], expected: dict[str, Tensor]) -> float:
    scale = max(float(g.abs().max()) for g in expected.values())
    return max(float((actual[name] - expected[name]).abs().max()) for name in expected) / scale


@pytest.mark.parametrize("case", list(CASES.values()), ids=list(CASES))
def test_two_pass_scheme_matches_direct_autograd(models, case: Case) -> None:
    policy, ref, projector = models
    groups = make_groups(case.old_noise)
    scheme, advantages, anchor_emb = two_pass_gradients(policy, ref, projector, groups, case)
    direct = direct_gradients(policy, ref, projector, groups, case, advantages, anchor_emb)
    assert all(float(g.abs().max()) > 0 for g in direct.values())
    for name, expected in direct.items():
        torch.testing.assert_close(scheme[name], expected, rtol=1e-5, atol=1e-8, msg=lambda m, n=name: f"{n}: {m}")
    assert max_relative_error(scheme, direct) < 1e-6


@pytest.mark.parametrize("drop", ["positive", "negative"])
def test_comparison_resolves_the_contrastive_terms(models, drop: str) -> None:
    policy, ref, projector = models
    case = CASES["negative_loss"]
    groups = make_groups(case.old_noise)
    without, advantages, anchor_emb = two_pass_gradients(policy, ref, projector, groups, case, drop=drop)
    direct = direct_gradients(policy, ref, projector, groups, case, advantages, anchor_emb)
    assert max_relative_error(without, direct) > 1e-3


def test_clipped_case_exercises_the_clip_range() -> None:
    groups = make_groups(CASES["clipped_ratio"].old_noise)
    ratios = torch.cat([torch.exp(shift) for group in groups for shift in group.old_shift])
    assert bool((ratios < 0.8).any()) and bool((ratios > 1.2).any())


def test_only_the_last_micro_batch_ends_the_deepspeed_step() -> None:
    events: list[tuple[str, bool]] = []
    state = SimpleNamespace(_set_sync_gradients=lambda sync: events.append(("accelerate", sync)))
    engine = SimpleNamespace(set_gradient_accumulation_boundary=lambda boundary: events.append(("engine", boundary)))
    deepspeed = SimpleNamespace(gradient_state=state, deepspeed_engine_wrapped=SimpleNamespace(engine=engine))
    for sync in (False, False, True):
        _set_gradient_sync(deepspeed, sync)
    assert events == [("accelerate", False), ("engine", False)] * 2 + [("accelerate", True), ("engine", True)]
    events.clear()
    _set_gradient_sync(SimpleNamespace(gradient_state=state, deepspeed_engine_wrapped=None), True)
    assert events == [("accelerate", True)]
