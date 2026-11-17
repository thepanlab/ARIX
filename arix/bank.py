from __future__ import annotations

import dataclasses
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor

from arix.config import BANK_UPDATE_RULES, RETRIEVAL_MODES
from arix.task import Example

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

UPDATE_ACTIONS = ("inserted", "replaced", "rejected")
BANK_FILE_TEMPLATE = "bank_rank{rank}.json"
EVICTION_TIE_TOLERANCE = 1e-6


@dataclass
class BankEntry:
    token_ids: list[int]
    text: str
    pinned: bool
    source: str
    added_step: int
    insertion_order: int


def _unit(x: Tensor) -> Tensor:
    return F.normalize(x.detach().float(), p=2.0, dim=-1)


class ReferenceBank:
    def __init__(self, capacity: int, pinned: BankEntry) -> None:
        if capacity < 1:
            raise ValueError(f"bank capacity must be at least 1, got {capacity}")
        if not pinned.pinned:
            raise ValueError("entry 0 of a reference bank must be the pinned original reference")
        self.capacity = capacity
        self.entries: list[BankEntry] = [pinned]
        self.redundancy: list[float] = [-math.inf]
        self.neighbor: list[int] = [-1]
        self.action_counts: dict[str, int] = dict.fromkeys(UPDATE_ACTIONS, 0)

    def __len__(self) -> int:
        return len(self.entries)

    def _check_entry_emb(self, entry_emb: Tensor) -> None:
        if entry_emb.dim() != 2 or entry_emb.shape[0] != len(self.entries):
            raise ValueError(f"entry embeddings must have shape [{len(self.entries)}, D], got {tuple(entry_emb.shape)}")

    def recompute_scores(self, entry_emb: Tensor) -> None:
        self._check_entry_emb(entry_emb)
        if len(self.entries) == 1:
            self.redundancy, self.neighbor = [-math.inf], [-1]
            return
        unit = _unit(entry_emb)
        similarity = unit @ unit.T
        similarity.fill_diagonal_(-math.inf)
        scores, neighbors = similarity.max(dim=1)
        self.redundancy = scores.tolist()
        self.neighbor = neighbors.tolist()

    def retrieve(self, rollout_emb: Tensor, entry_emb: Tensor, mode: str, generator: torch.Generator) -> Tensor:
        self._check_entry_emb(entry_emb)
        if mode == "similarity":
            return (_unit(rollout_emb) @ _unit(entry_emb).to(rollout_emb.device).T).argmax(dim=1)
        if mode == "random":
            size = (rollout_emb.shape[0],)
            draws = torch.randint(len(self.entries), size, generator=generator, device=generator.device)
            return draws.to(rollout_emb.device)
        raise ValueError(f"unknown retrieval mode {mode!r}; expected one of {RETRIEVAL_MODES}")

    def update(
        self,
        cand: BankEntry,
        cand_emb: Tensor,
        entry_emb: Tensor,
        rule: str,
        generator: torch.Generator,
    ) -> tuple[str, Tensor]:
        if rule not in BANK_UPDATE_RULES:
            raise ValueError(f"unknown bank update rule {rule!r}; expected one of {BANK_UPDATE_RULES}")
        if cand.pinned:
            raise ValueError("only rollout rationales can be offered to a bank")
        entry_emb = entry_emb.detach()
        self.recompute_scores(entry_emb)
        cand_row = cand_emb.detach().reshape(1, -1).to(device=entry_emb.device, dtype=entry_emb.dtype)
        admitted = dataclasses.replace(cand, insertion_order=max(e.insertion_order for e in self.entries) + 1)
        if len(self.entries) < self.capacity:
            action = "inserted"
            self.entries.append(admitted)
            new_emb = torch.cat([entry_emb, cand_row], dim=0)
        else:
            slot = self._eviction_slot(cand_row, entry_emb, rule, generator)
            if slot is None:
                action, new_emb = "rejected", entry_emb
            else:
                action = "replaced"
                self.entries[slot] = admitted
                new_emb = entry_emb.clone()
                new_emb[slot] = cand_row[0]
        self.recompute_scores(new_emb)
        self.action_counts[action] += 1
        return action, new_emb

    def _eviction_slot(self, cand_row: Tensor, entry_emb: Tensor, rule: str, generator: torch.Generator) -> int | None:
        if rule == "redundancy":
            similarity = (_unit(entry_emb) @ _unit(cand_row)[0]).cpu()
            highest = float(similarity.max())
            tied = [k for k, value in enumerate(similarity.tolist()) if value >= highest - EVICTION_TIE_TOLERANCE]
            evictable_nearest = [k for k in tied if not self.entries[k].pinned]
            if not evictable_nearest:
                return None
            nearest = evictable_nearest[0]
            second_highest = float(torch.sort(similarity, descending=True).values[1])
            return nearest if second_highest < self.redundancy[nearest] else None
        evictable = [k for k, entry in enumerate(self.entries) if not entry.pinned]
        if not evictable:
            return None
        if rule == "fifo":
            return min(evictable, key=lambda k: self.entries[k].insertion_order)
        draw = torch.randint(len(evictable), (1,), generator=generator, device=generator.device)
        return evictable[int(draw)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "entries": [dataclasses.asdict(entry) for entry in self.entries],
            "redundancy": [None if math.isinf(score) else score for score in self.redundancy],
            "neighbor": list(self.neighbor),
            "action_counts": dict(self.action_counts),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ReferenceBank:
        entries = [BankEntry(**entry) for entry in d["entries"]]
        if len(entries) > d["capacity"] or any(entry.pinned for entry in entries[1:]):
            raise ValueError("a bank holds at most `capacity` entries and only entry 0 is pinned")
        if not len(entries) == len(d["redundancy"]) == len(d["neighbor"]):
            raise ValueError("redundancy scores and neighbor indices must have one value per entry")
        bank = cls(d["capacity"], entries[0])
        bank.entries = entries
        bank.redundancy = [-math.inf if score is None else float(score) for score in d["redundancy"]]
        bank.neighbor = [int(n) for n in d["neighbor"]]
        bank.action_counts = {action: int(d["action_counts"][action]) for action in UPDATE_ACTIONS}
        return bank


def _process_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", "0"))


class BankStore:
    def __init__(self, rank: int | None = None) -> None:
        self.rank = _process_rank() if rank is None else rank
        self.banks: dict[str, ReferenceBank] = {}

    def get_or_create(self, ex: Example, tokenizer: PreTrainedTokenizerBase, capacity: int) -> ReferenceBank:
        bank = self.banks.get(ex.uid)
        if bank is None:
            token_ids = list(tokenizer.encode(ex.reference, add_special_tokens=False))
            if tokenizer.eos_token_id is not None:
                token_ids.append(tokenizer.eos_token_id)
            pinned = BankEntry(
                token_ids=token_ids,
                text=ex.reference,
                pinned=True,
                source="reference",
                added_step=0,
                insertion_order=0,
            )
            bank = self.banks[ex.uid] = ReferenceBank(capacity, pinned)
        elif bank.capacity != capacity or bank.entries[0].text != ex.reference:
            raise ValueError(f"the stored bank of {ex.uid!r} has another capacity or pinned reference")
        return bank

    def file_path(self, path: str | os.PathLike[str]) -> Path:
        path = Path(path)
        return path if path.suffix == ".json" else path / BANK_FILE_TEMPLATE.format(rank=self.rank)

    def save(self, path: str | os.PathLike[str]) -> Path:
        file = self.file_path(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        payload = {"rank": self.rank, "banks": {uid: self.banks[uid].to_dict() for uid in sorted(self.banks)}}
        tmp = file.with_name(file.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        tmp.replace(file)
        return file

    def load(self, path: str | os.PathLike[str]) -> None:
        with self.file_path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        self.banks = {uid: ReferenceBank.from_dict(d) for uid, d in payload["banks"].items()}

    def stats(self) -> dict[str, float]:
        sizes = [len(bank) for bank in self.banks.values()]
        stats: dict[str, float] = {
            "num_banks": len(sizes),
            "mean_size": sum(sizes) / len(sizes) if sizes else 0.0,
            "max_size": max(sizes, default=0),
            "full_fraction": (
                sum(len(bank) == bank.capacity for bank in self.banks.values()) / len(sizes) if sizes else 0.0
            ),
        }
        for action in UPDATE_ACTIONS:
            stats[action] = sum(bank.action_counts[action] for bank in self.banks.values())
        return stats
