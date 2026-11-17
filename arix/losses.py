from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from arix.config import KL_ESTIMATORS


def _cosine_matrix(a: Tensor, b: Tensor) -> Tensor:
    return F.normalize(a.float(), dim=-1) @ F.normalize(b.float(), dim=-1).T


def _cosine_rows(a: Tensor, b: Tensor) -> Tensor:
    return (F.normalize(a.float(), dim=-1) * F.normalize(b.float(), dim=-1)).sum(dim=-1)


def partial_rewards(correct: Tensor, anchor_cos: Tensor, enabled: bool, scale: float, bias: float) -> Tensor:
    correct = correct.to(device=anchor_cos.device, dtype=torch.bool)
    if not enabled:
        return correct.float()
    partial = scale * anchor_cos.detach().float() + bias
    return torch.where(correct, torch.ones_like(partial), partial)


def group_advantages(rewards: Tensor, eps: float, std_normalize: bool) -> Tensor:
    rewards = rewards.detach().float()
    centered = rewards - rewards.mean()
    if not std_normalize:
        return centered
    std = rewards.std() if rewards.numel() > 1 else rewards.new_zeros(())
    return centered / (std + eps)


def grpo_sequence_losses(
    logp: Tensor,
    old_logp: Tensor,
    ref_logp: Tensor | None,
    advantages: Tensor,
    mask: Tensor,
    *,
    clip_eps: float,
    beta: float,
    kl_estimator: str,
    length_normalize: bool,
    max_new_tokens: int,
    kl_full: Tensor | None = None,
) -> Tensor:
    if kl_estimator not in KL_ESTIMATORS:
        raise ValueError(f"unknown KL estimator {kl_estimator!r}; expected one of {KL_ESTIMATORS}")
    mask = mask.bool()
    logp = logp.float()
    zero = logp.new_zeros(())
    ratio = torch.exp(torch.where(mask, logp - old_logp.detach().float(), zero))
    adv = advantages.detach().float().unsqueeze(-1)
    per_token = torch.minimum(ratio * adv, torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv)
    if beta != 0.0:
        if kl_estimator == "k3":
            if ref_logp is None:
                raise ValueError("kl_estimator='k3' with beta > 0 requires ref_logp")
            log_ratio_ref = torch.where(mask, ref_logp.detach().float() - logp, zero)
            kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0
        else:
            if kl_full is None:
                raise ValueError("kl_estimator='full' with beta > 0 requires kl_full")
            kl = torch.where(mask, kl_full.float(), zero)
        per_token = per_token - beta * kl
    total = torch.where(mask, per_token, zero).sum(dim=-1)
    if length_normalize:
        return total / mask.sum(dim=-1).clamp_min(1).float()
    return total / float(max_new_tokens)


def info_nce_positive(rollout_emb: Tensor, anchor_emb: Tensor, correct: Tensor, tau: float) -> Tensor:
    correct = correct.to(device=rollout_emb.device, dtype=torch.bool)
    if not bool(correct.any()):
        return rollout_emb.new_zeros((), dtype=torch.float32)
    logits = _cosine_matrix(anchor_emb, rollout_emb) / tau
    log_prob = logits.diagonal() - torch.logsumexp(logits, dim=1)
    return -log_prob[correct].mean()


def info_nce_negative(lcs_emb: Tensor, exc_emb: Tensor, anchor_emb: Tensor, incorrect: Tensor, tau: float) -> Tensor:
    incorrect = incorrect.to(device=lcs_emb.device, dtype=torch.bool)
    if not bool(incorrect.any()):
        return lcs_emb.new_zeros((), dtype=torch.float32)
    anchors = anchor_emb[incorrect]
    numerator = _cosine_rows(anchors, lcs_emb[incorrect]) / tau
    log_denominator = torch.logsumexp(_cosine_matrix(anchors, exc_emb[incorrect]) / tau, dim=1)
    return (log_denominator - numerator).mean()


def embedding_loss_grads(
    rollout_emb: Tensor,
    anchor_emb: Tensor,
    correct: Tensor,
    tau: float,
    alpha_pos: float,
    neg: dict | None = None,
) -> tuple[Tensor, Tensor]:
    with torch.enable_grad():
        anchors = anchor_emb.detach().float()
        inputs = [rollout_emb.detach().float().requires_grad_(True)]
        value = alpha_pos * info_nce_positive(inputs[0], anchors, correct, tau)
        if neg is not None:
            inputs += [neg[key].detach().float().requires_grad_(True) for key in ("lcs_emb", "exc_emb")]
            value = value + neg["alpha_neg"] * info_nce_negative(inputs[1], inputs[2], anchors, neg["incorrect"], tau)
        raw = torch.autograd.grad(value, inputs, allow_unused=True) if value.requires_grad else (None,) * len(inputs)
    grads = [torch.zeros_like(x) if g is None else g for x, g in zip(inputs, raw, strict=True)]
    return value.detach(), grads[0] if neg is None else torch.stack(grads)
