from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from arix.config import POOLING_MODES


class RationaleProjector(nn.Module):
    def __init__(self, hidden_size: int, proj_dim: int = 64, seed: int = 0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.proj_dim = proj_dim
        self.seed = seed
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            self.linear = nn.Linear(hidden_size, proj_dim, device="cpu", dtype=torch.float32)
        self.linear.requires_grad_(False)

    def forward(self, h: Tensor) -> Tensor:
        return self.linear(h.to(self.linear.weight.dtype))

    def save(self, path: str | os.PathLike[str]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "hidden_size": self.hidden_size,
                "proj_dim": self.proj_dim,
                "seed": self.seed,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> RationaleProjector:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        projector = cls(payload["hidden_size"], payload["proj_dim"], payload["seed"])
        projector.load_state_dict(payload["state_dict"])
        return projector


def pooling_weights(token_logprobs: Tensor, mask: Tensor, pooling: str = "softmax") -> Tensor:
    mask = mask.bool()
    if pooling == "mean":
        weights = mask.float()
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
    if pooling != "softmax":
        raise ValueError(f"unknown pooling {pooling!r}; expected one of {POOLING_MODES}")
    scores = token_logprobs.float().masked_fill(~mask, torch.finfo(torch.float32).min)
    return torch.softmax(scores, dim=-1).masked_fill(~mask, 0.0)


def rationale_embedding(
    hidden: Tensor,
    token_logprobs: Tensor,
    mask: Tensor,
    projector: RationaleProjector,
    pooling: str = "softmax",
) -> Tensor:
    weights = pooling_weights(token_logprobs, mask, pooling)
    pooled = torch.einsum("bt,btd->bd", weights, projector(hidden))
    return F.normalize(pooled, p=2.0, dim=-1)


def masked_subset_embedding(
    hidden: Tensor,
    token_logprobs: Tensor,
    mask: Tensor,
    subset_mask: Tensor,
    projector: RationaleProjector,
    pooling: str = "softmax",
) -> Tensor:
    return rationale_embedding(hidden, token_logprobs, mask.bool() & subset_mask.bool(), projector, pooling)


def lcs_mask(tokens: Sequence[int], anchor_tokens: Sequence[int]) -> list[bool]:
    rollout = np.asarray(tokens, dtype=np.int64)
    anchor = np.asarray(anchor_tokens, dtype=np.int64)
    n, m = len(rollout), len(anchor)
    table = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(1, n + 1):
        previous = table[i - 1]
        best = np.maximum(previous[1:], previous[:-1] + (anchor == rollout[i - 1]))
        np.maximum.accumulate(best, out=table[i, 1:])
    selected = [False] * n
    rollout_list, anchor_list = rollout.tolist(), anchor.tolist()
    i, j = n, m
    while i > 0 and j > 0:
        if rollout_list[i - 1] == anchor_list[j - 1]:
            selected[i - 1] = True
            i -= 1
            j -= 1
        elif table[i - 1, j] > table[i, j - 1]:
            i -= 1
        else:
            j -= 1
    return selected
