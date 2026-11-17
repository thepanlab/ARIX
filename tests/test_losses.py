from __future__ import annotations

import math
import statistics

import pytest
import torch
import torch.nn.functional as F

from arix.losses import (
    embedding_loss_grads,
    group_advantages,
    grpo_sequence_losses,
    info_nce_negative,
    info_nce_positive,
    partial_rewards,
)

TAU = 0.2


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True)) / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


def _group(seed: int, size: int = 6, dim: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    rollouts = F.normalize(torch.randn(size, dim, generator=generator), dim=-1)
    anchors = F.normalize(torch.randn(size, dim, generator=generator), dim=-1)
    return rollouts, anchors


def test_partial_rewards_lie_between_point_one_and_point_three() -> None:
    anchor_cos = torch.linspace(-1.0, 1.0, 101)
    correct = torch.zeros(101, dtype=torch.bool)
    correct[::7] = True
    rewards = partial_rewards(correct, anchor_cos, True, 0.1, 0.2)
    assert torch.all(rewards[correct] == 1.0)
    incorrect = rewards[~correct]
    assert torch.allclose(incorrect, 0.1 * anchor_cos[~correct] + 0.2)
    assert float(incorrect.min()) >= 0.1 - 1e-7 and float(incorrect.max()) <= 0.3 + 1e-7
    assert partial_rewards(torch.tensor([False]), torch.tensor([-1.0]), True, 0.1, 0.2).item() == pytest.approx(0.1)
    assert partial_rewards(torch.tensor([False]), torch.tensor([1.0]), True, 0.1, 0.2).item() == pytest.approx(0.3)


def test_binary_rewards_when_partial_reward_is_disabled() -> None:
    correct = torch.tensor([True, False, True, False])
    rewards = partial_rewards(correct, torch.tensor([0.9, 0.9, -0.5, 0.1]), False, 0.1, 0.2)
    assert rewards.tolist() == [1.0, 0.0, 1.0, 0.0]


def test_rewards_carry_no_gradient() -> None:
    anchor_cos = torch.tensor([0.5, 0.2], requires_grad=True)
    assert not partial_rewards(torch.tensor([False, True]), anchor_cos, True, 0.1, 0.2).requires_grad


def test_group_advantages_standardize_with_the_sample_std() -> None:
    rewards = [1.0, 0.25, 0.3, 1.0, 0.28, 0.21]
    advantages = group_advantages(torch.tensor(rewards), 1e-4, std_normalize=True)
    mean, std = statistics.fmean(rewards), statistics.stdev(rewards)
    assert advantages.tolist() == pytest.approx([(r - mean) / (std + 1e-4) for r in rewards], rel=1e-5)
    assert float(advantages.sum()) == pytest.approx(0.0, abs=1e-5)


def test_dr_grpo_advantages_only_center_the_rewards() -> None:
    rewards = [1.0, 0.25, 0.3, 1.0]
    advantages = group_advantages(torch.tensor(rewards), 1e-4, std_normalize=False)
    assert advantages.tolist() == pytest.approx([r - statistics.fmean(rewards) for r in rewards])


def test_identical_rewards_give_zero_advantages() -> None:
    assert torch.all(group_advantages(torch.ones(8), 1e-4, True) == 0)
    assert torch.all(group_advantages(torch.ones(1), 1e-4, True) == 0)


LOGP = [[-1.0, -0.5, -2.0, -0.1, -0.7], [-0.3, -1.2, -0.7, -0.2, -0.4]]
OLD = [[-1.2, -0.8, -1.5, -0.1, -0.9], [-0.3, -0.9, -1.0, -0.6, -0.4]]
REF = [[-1.1, -0.6, -1.8, -0.2, -0.5], [-0.4, -1.1, -0.8, -0.1, -0.3]]
MASK = [[False, True, True, True, True], [False, True, True, True, False]]
ADV = [0.8, -1.3]


def _hand_sequence_values(beta: float, length_normalize: bool, max_new_tokens: int) -> list[float]:
    values = []
    for logp, old, ref, mask, adv in zip(LOGP, OLD, REF, MASK, ADV, strict=True):
        total = 0.0
        for p, p_old, q, keep in zip(logp, old, ref, mask, strict=True):
            if not keep:
                continue
            rho = math.exp(p - p_old)
            surrogate = min(rho * adv, min(max(rho, 0.8), 1.2) * adv)
            kl = math.exp(q - p) - (q - p) - 1.0
            total += surrogate - beta * kl
        values.append(total / (sum(mask) if length_normalize else max_new_tokens))
    return values


@pytest.mark.parametrize("length_normalize", [True, False])
@pytest.mark.parametrize("beta", [0.0, 0.04])
def test_grpo_sequence_losses_match_a_hand_computation(beta: float, length_normalize: bool) -> None:
    values = grpo_sequence_losses(
        torch.tensor(LOGP),
        torch.tensor(OLD),
        torch.tensor(REF),
        torch.tensor(ADV),
        torch.tensor(MASK),
        clip_eps=0.2,
        beta=beta,
        kl_estimator="k3",
        length_normalize=length_normalize,
        max_new_tokens=16,
    )
    assert values.tolist() == pytest.approx(_hand_sequence_values(beta, length_normalize, 16), rel=1e-6)


def test_clipped_tokens_receive_no_policy_gradient() -> None:
    logp = torch.tensor(LOGP, requires_grad=True)
    values = grpo_sequence_losses(
        logp,
        torch.tensor(OLD),
        None,
        torch.tensor(ADV),
        torch.tensor(MASK),
        clip_eps=0.2,
        beta=0.0,
        kl_estimator="k3",
        length_normalize=True,
        max_new_tokens=16,
    )
    values.sum().backward()
    ratio = torch.exp(torch.tensor(LOGP) - torch.tensor(OLD))
    adv = torch.tensor(ADV)[:, None]
    clipped = ((adv > 0) & (ratio > 1.2)) | ((adv < 0) & (ratio < 0.8))
    mask = torch.tensor(MASK)
    assert torch.any(clipped & mask) and torch.any(~clipped & mask)
    assert torch.all(logp.grad[clipped | ~mask] == 0)
    assert torch.all(logp.grad[~clipped & mask] != 0)


def test_full_kl_estimator_uses_the_given_per_token_kl() -> None:
    kl_full = torch.tensor([[9.0, 0.1, 0.2, 0.3, 0.4], [9.0, 0.5, 0.6, 0.7, 9.0]])
    args = (torch.tensor(LOGP), torch.tensor(OLD), None, torch.tensor(ADV), torch.tensor(MASK))
    options = {"clip_eps": 0.2, "kl_estimator": "full", "length_normalize": True, "max_new_tokens": 16}
    without_kl = grpo_sequence_losses(*args, beta=0.0, **options)
    with_kl = grpo_sequence_losses(*args, beta=0.5, kl_full=kl_full, **options)
    expected = without_kl - 0.5 * torch.tensor([(0.1 + 0.2 + 0.3 + 0.4) / 4, (0.5 + 0.6 + 0.7) / 3])
    assert torch.allclose(with_kl, expected)
    with pytest.raises(ValueError):
        grpo_sequence_losses(*args, beta=0.5, **options)
    with pytest.raises(ValueError):
        grpo_sequence_losses(*args, beta=0.5, **{**options, "kl_estimator": "k2"})


def test_hyperparameters_are_keyword_only() -> None:
    args = (torch.tensor(LOGP), torch.tensor(OLD), torch.tensor(REF), torch.tensor(ADV), torch.tensor(MASK))
    with pytest.raises(TypeError):
        grpo_sequence_losses(*args, 0.2, 0.04, "k3", True, 16)  # type: ignore[misc]


def test_values_outside_the_response_mask_are_ignored() -> None:
    logp, old, ref = torch.tensor(LOGP), torch.tensor(OLD), torch.tensor(REF)
    mask = torch.tensor(MASK)
    options = {"clip_eps": 0.2, "beta": 0.04, "kl_estimator": "k3", "length_normalize": True, "max_new_tokens": 16}
    clean = grpo_sequence_losses(logp, old, ref, torch.tensor(ADV), mask, **options)
    polluted_logp = logp.masked_fill(~mask, float("nan")).requires_grad_(True)
    polluted_old, polluted_ref = old.masked_fill(~mask, 50.0), ref.masked_fill(~mask, -math.inf)
    polluted = grpo_sequence_losses(polluted_logp, polluted_old, polluted_ref, torch.tensor(ADV), mask, **options)
    assert torch.allclose(polluted, clean)
    polluted.sum().backward()
    assert torch.all(torch.isfinite(polluted_logp.grad)) and torch.all(polluted_logp.grad[~mask] == 0)


def _info_nce_by_formula(rollouts: torch.Tensor, anchors: torch.Tensor, correct: list[bool]) -> float:
    e, a = rollouts.tolist(), anchors.tolist()
    terms = []
    for j, is_correct in enumerate(correct):
        if is_correct:
            numerator = math.exp(_cos(a[j], e[j]) / TAU)
            denominator = sum(math.exp(_cos(a[j], e[m]) / TAU) for m in range(len(e)))
            terms.append(-math.log(numerator / denominator))
    return sum(terms) / len(terms)


def test_info_nce_matches_equation_10_with_per_rollout_anchors() -> None:
    rollouts, anchors = _group(0)
    correct = [True, False, True, True, False, False]
    loss = info_nce_positive(rollouts, anchors, torch.tensor(correct), TAU)
    assert loss.item() == pytest.approx(_info_nce_by_formula(rollouts, anchors, correct), rel=1e-5)


def test_info_nce_with_the_pinned_anchor_matches_arix_without_bank() -> None:
    rollouts, anchors = _group(1)
    shared = anchors[:1].expand_as(anchors)
    correct = [False, True, True, False, True, True]
    loss = info_nce_positive(rollouts, shared, torch.tensor(correct), TAU)
    assert loss.item() == pytest.approx(_info_nce_by_formula(rollouts, shared, correct), rel=1e-5)


def test_info_nce_normalizes_its_inputs() -> None:
    rollouts, anchors = _group(2)
    correct = torch.tensor([True, True, False, False, True, False])
    scaled = info_nce_positive(3.0 * rollouts, 0.5 * anchors, correct, TAU)
    assert scaled.item() == pytest.approx(info_nce_positive(rollouts, anchors, correct, TAU).item(), rel=1e-5)


def test_info_nce_is_zero_without_correct_rollouts() -> None:
    rollouts, anchors = _group(3)
    loss = info_nce_positive(rollouts, anchors, torch.zeros(6, dtype=torch.bool), TAU)
    assert loss.item() == 0.0


def test_negative_loss_matches_equation_s1() -> None:
    generator = torch.Generator().manual_seed(4)
    lcs, exc, anchors = (F.normalize(torch.randn(6, 5, generator=generator), dim=-1) for _ in range(3))
    incorrect = [True, True, False, True, False, True]
    loss = info_nce_negative(lcs, exc, anchors, torch.tensor(incorrect), TAU)
    negatives = [j for j, flag in enumerate(incorrect) if flag]
    a, lcs_rows, x = anchors.tolist(), lcs.tolist(), exc.tolist()
    expected = sum(
        -math.log(math.exp(_cos(a[j], lcs_rows[j]) / TAU) / sum(math.exp(_cos(a[j], x[m]) / TAU) for m in negatives))
        for j in negatives
    ) / len(negatives)
    assert loss.item() == pytest.approx(expected, rel=1e-5)
    assert info_nce_negative(lcs, exc, anchors, torch.zeros(6, dtype=torch.bool), TAU).item() == 0.0


def test_embedding_loss_grads_match_autograd() -> None:
    rollouts, anchors = _group(5)
    correct = torch.tensor([True, False, True, True, False, False])
    leaf = rollouts.clone().requires_grad_(True)
    objective = 1e-3 * info_nce_positive(leaf, anchors, correct, TAU)
    (expected_grad,) = torch.autograd.grad(objective, leaf)
    value, grad = embedding_loss_grads(rollouts, anchors, correct, TAU, alpha_pos=1e-3)
    assert value.item() == pytest.approx(objective.item(), rel=1e-6)
    assert grad.shape == rollouts.shape and torch.allclose(grad, expected_grad)


def test_embedding_gradients_reproduce_the_parameter_gradient() -> None:
    generator = torch.Generator().manual_seed(6)
    raw = torch.randn(6, 5, generator=generator, requires_grad=True)
    anchors = F.normalize(torch.randn(6, 5, generator=generator), dim=-1)
    correct = torch.tensor([False, True, True, False, True, False])
    (direct,) = torch.autograd.grad(1e-3 * info_nce_positive(F.normalize(raw, dim=-1), anchors, correct, TAU), raw)
    with torch.no_grad():
        _, grad = embedding_loss_grads(F.normalize(raw, dim=-1), anchors, correct, TAU, alpha_pos=1e-3)
    (two_pass,) = torch.autograd.grad((F.normalize(raw, dim=-1) * grad).sum(), raw)
    assert torch.allclose(two_pass, direct, atol=1e-9)


def test_embedding_loss_grads_with_the_negative_term() -> None:
    rollouts, anchors = _group(7)
    generator = torch.Generator().manual_seed(8)
    lcs, exc = (F.normalize(torch.randn(6, 5, generator=generator), dim=-1) for _ in range(2))
    correct = torch.tensor([True, False, True, False, False, True])
    leaves = [x.clone().requires_grad_(True) for x in (rollouts, lcs, exc)]
    objective = 1e-3 * info_nce_positive(leaves[0], anchors, correct, TAU) + 5e-4 * info_nce_negative(
        leaves[1], leaves[2], anchors, ~correct, TAU
    )
    expected = torch.autograd.grad(objective, leaves)
    neg = {"lcs_emb": lcs, "exc_emb": exc, "incorrect": ~correct, "alpha_neg": 5e-4}
    value, grads = embedding_loss_grads(rollouts, anchors, correct, TAU, alpha_pos=1e-3, neg=neg)
    assert value.item() == pytest.approx(objective.item(), rel=1e-6)
    assert grads.shape == (3, 6, 5)
    for got, want in zip(grads, expected, strict=True):
        assert torch.allclose(got, want)


def test_embedding_loss_grads_are_zero_without_contrastive_terms() -> None:
    rollouts, anchors = _group(9)
    value, grad = embedding_loss_grads(rollouts, anchors, torch.zeros(6, dtype=torch.bool), TAU, alpha_pos=1e-3)
    assert value.item() == 0.0 and torch.all(grad == 0) and grad.shape == rollouts.shape
    neg = {"lcs_emb": rollouts, "exc_emb": rollouts, "incorrect": torch.zeros(6, dtype=torch.bool), "alpha_neg": 5e-4}
    value, grads = embedding_loss_grads(rollouts, anchors, torch.ones(6, dtype=torch.bool), TAU, 0.0, neg=neg)
    assert value.item() == 0.0 and torch.all(grads == 0) and grads.shape == (3, 6, 5)


def test_anchors_are_stop_gradient_targets() -> None:
    rollouts, anchors = _group(10)
    anchors = anchors.clone().requires_grad_(True)
    value, grad = embedding_loss_grads(rollouts, anchors, torch.ones(6, dtype=torch.bool), TAU, alpha_pos=1.0)
    assert not value.requires_grad and not grad.requires_grad
    assert anchors.grad is None
