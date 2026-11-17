from __future__ import annotations

import json
import math
from collections import Counter

import pytest
import torch
import torch.nn.functional as F

from arix.bank import EVICTION_TIE_TOLERANCE, BankEntry, BankStore, ReferenceBank
from arix.task import Example


def _pinned() -> BankEntry:
    return BankEntry(
        token_ids=[1, 2], text="reference", pinned=True, source="reference", added_step=0, insertion_order=0
    )


def _rollout(tag: int, step: int = 0) -> BankEntry:
    return BankEntry(
        token_ids=[tag], text=f"rollout {tag}", pinned=False, source="rollout", added_step=step, insertion_order=-1
    )


def _angles(*degrees: float) -> torch.Tensor:
    radians = torch.tensor(degrees, dtype=torch.float64) * math.pi / 180
    return torch.stack([radians.cos(), radians.sin()], dim=-1).float()


def _scores(emb: torch.Tensor) -> tuple[list[float], list[int]]:
    unit = F.normalize(emb, dim=-1)
    scores, neighbors = [], []
    for k in range(len(unit)):
        others = [(float(unit[k] @ unit[m]), m) for m in range(len(unit)) if m != k]
        best = max(others, key=lambda pair: (pair[0], -pair[1]))
        scores.append(best[0])
        neighbors.append(best[1])
    return scores, neighbors


def _max_pairwise(emb: torch.Tensor) -> float:
    unit = F.normalize(emb, dim=-1)
    similarity = unit @ unit.T
    similarity.fill_diagonal_(-math.inf)
    return float(similarity.max())


def _full_bank(degrees: tuple[float, ...]) -> tuple[ReferenceBank, torch.Tensor]:
    bank = ReferenceBank(len(degrees), _pinned())
    emb = _angles(degrees[0])
    generator = torch.Generator().manual_seed(0)
    for tag, vector in enumerate(_angles(*degrees[1:]), start=1):
        action, emb = bank.update(_rollout(tag), vector, emb, "redundancy", generator)
        assert action == "inserted"
    return bank, emb


def test_new_bank_holds_only_the_pinned_reference() -> None:
    bank = ReferenceBank(5, _pinned())
    assert len(bank) == 1 and bank.entries[0].pinned
    assert bank.redundancy == [-math.inf] and bank.neighbor == [-1]
    with pytest.raises(ValueError):
        ReferenceBank(5, _rollout(1))
    with pytest.raises(ValueError):
        ReferenceBank(0, _pinned())


@pytest.mark.parametrize("rule", ["redundancy", "fifo", "random"])
def test_bank_fills_directly_without_redundancy_check(rule: str) -> None:
    bank = ReferenceBank(4, _pinned())
    emb = _angles(0.0)
    generator = torch.Generator().manual_seed(0)
    for tag in range(1, 4):
        action, emb = bank.update(_rollout(tag), _angles(0.0)[0], emb, rule, generator)
        assert action == "inserted"
        assert len(bank) == tag + 1 and emb.shape == (tag + 1, 2)
        assert bank.entries[tag].token_ids == [tag] and bank.entries[tag].insertion_order == tag
    assert bank.redundancy == pytest.approx([1.0] * 4)


def test_scores_are_recomputed_after_every_change() -> None:
    generator = torch.Generator().manual_seed(1)
    bank = ReferenceBank(4, _pinned())
    emb = F.normalize(torch.randn(1, 3, generator=generator), dim=-1)
    for tag in range(1, 40):
        candidate = F.normalize(torch.randn(3, generator=generator), dim=0)
        _, emb = bank.update(_rollout(tag), candidate, emb, "redundancy", generator)
        if len(bank) > 1:
            scores, neighbors = _scores(emb)
            assert bank.redundancy == pytest.approx(scores, abs=1e-6)
            assert bank.neighbor == neighbors


def test_recompute_scores_matches_brute_force_and_rejects_wrong_shapes() -> None:
    bank, emb = _full_bank((0.0, 90.0, 100.0, 200.0))
    bank.recompute_scores(emb)
    scores, neighbors = _scores(emb)
    assert bank.redundancy == pytest.approx(scores, abs=1e-6) and bank.neighbor == neighbors
    with pytest.raises(ValueError):
        bank.recompute_scores(emb[:2])


def test_exchange_is_made_when_candidate_is_less_redundant() -> None:
    bank, emb = _full_bank((0.0, 90.0, 100.0))
    action, new_emb = bank.update(_rollout(9), _angles(200.0)[0], emb, "redundancy", torch.Generator())
    assert action == "replaced"
    assert [entry.token_ids for entry in bank.entries] == [[1, 2], [1], [9]]
    assert bank.entries[2].insertion_order == 3
    assert torch.allclose(new_emb, _angles(0.0, 90.0, 200.0))
    assert torch.equal(new_emb[:2], emb[:2])
    assert _max_pairwise(new_emb) < _max_pairwise(emb)


def test_exchange_uses_the_second_highest_similarity_of_the_candidate() -> None:
    bank, emb = _full_bank((0.0, 90.0, 100.0))
    action, new_emb = bank.update(_rollout(9), _angles(105.0)[0], emb, "redundancy", torch.Generator())
    assert action == "replaced" and bank.entries[2].token_ids == [9]
    assert _max_pairwise(new_emb) < _max_pairwise(emb)


def test_exact_duplicate_is_refused_once_the_bank_is_full() -> None:
    emb = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.6, 0.8]])
    bank = ReferenceBank(3, _pinned())
    generator = torch.Generator()
    current = emb[:1]
    for tag in (1, 2):
        _, current = bank.update(_rollout(tag), emb[tag], current, "redundancy", generator)
    action, _ = bank.update(_rollout(3), emb[2], current, "redundancy", generator)
    assert action == "rejected"


def test_exchange_is_refused_when_candidate_is_not_less_redundant() -> None:
    bank, emb = _full_bank((0.0, 90.0, 100.0))
    entries_before = list(bank.entries)
    action, new_emb = bank.update(_rollout(9), _angles(96.0)[0], emb, "redundancy", torch.Generator())
    assert action == "rejected"
    assert bank.entries == entries_before and torch.equal(new_emb, emb)


def test_score_of_nearest_incumbent_is_taken_against_the_current_bank() -> None:
    bank, _ = _full_bank((0.0, 90.0, 100.0))
    moved = _angles(0.0, 30.0, 100.0)
    action, _ = bank.update(_rollout(9), _angles(90.0)[0], moved, "redundancy", torch.Generator())
    assert action == "rejected"
    assert bank.redundancy[2] == pytest.approx(math.cos(math.radians(70.0)), abs=1e-6)
    action, _ = bank.update(_rollout(10), _angles(175.0)[0], moved, "redundancy", torch.Generator())
    assert action == "replaced"


def test_pinned_reference_is_never_evicted_by_the_redundancy_rule() -> None:
    bank, emb = _full_bank((0.0, 90.0, 100.0))
    action, new_emb = bank.update(_rollout(9), _angles(-5.0)[0], emb, "redundancy", torch.Generator())
    assert action == "rejected"
    assert bank.entries[0].pinned and bank.entries[0].token_ids == [1, 2]
    assert torch.equal(new_emb, emb)


def _bank_with_a_copy_of_the_pinned_reference(copy: torch.Tensor) -> tuple[ReferenceBank, torch.Tensor]:
    bank = ReferenceBank(3, _pinned())
    emb, generator = _angles(0.0), torch.Generator()
    for tag, vector in ((1, copy), (2, _angles(90.0)[0])):
        action, emb = bank.update(_rollout(tag), vector, emb, "redundancy", generator)
        assert action == "inserted"
    return bank, emb


def test_a_copy_of_the_pinned_reference_is_evicted_by_a_candidate_near_both() -> None:
    bank, emb = _bank_with_a_copy_of_the_pinned_reference(_angles(0.0)[0])
    assert torch.equal(emb[0], emb[1]) and bank.redundancy[1] == pytest.approx(1.0)
    generator = torch.Generator()
    action, emb = bank.update(_rollout(3), _angles(3.0)[0], emb, "redundancy", generator)
    assert action == "replaced" and [entry.token_ids for entry in bank.entries] == [[1, 2], [3], [2]]
    assert bank.entries[0].pinned and torch.allclose(emb, _angles(0.0, 3.0, 90.0))
    for tag, degrees in enumerate((6.0, -2.0, 4.0, 1.0, -7.0, 8.0, 2.5, -4.0, 9.0, 5.0, -1.0), start=4):
        _, emb = bank.update(_rollout(tag), _angles(degrees)[0], emb, "redundancy", generator)
    assert bank.entries[0].token_ids == [1, 2] and all(entry.token_ids != [1] for entry in bank.entries)
    assert all(float(emb[0] @ emb[k]) < 1.0 - 1e-6 for k in (1, 2))


@pytest.mark.parametrize("offset", [-5e-6, 5e-6])
def test_near_ties_with_the_pinned_reference_go_to_the_copy(offset: float) -> None:
    copy = torch.tensor([math.cos(offset), math.sin(offset)], dtype=torch.float32)
    bank, emb = _bank_with_a_copy_of_the_pinned_reference(copy)
    candidate = _angles(5.0)[0]
    similarity = F.normalize(emb, dim=-1) @ candidate
    assert 1e-8 < abs(float(similarity[0] - similarity[1])) < EVICTION_TIE_TOLERANCE
    action, _ = bank.update(_rollout(3), candidate, emb, "redundancy", torch.Generator())
    assert action == "replaced" and bank.entries[1].token_ids == [3]


def test_a_candidate_clearly_nearest_to_the_pinned_reference_is_rejected() -> None:
    offset = -2e-4
    copy = torch.tensor([math.cos(offset), math.sin(offset)], dtype=torch.float32)
    bank, emb = _bank_with_a_copy_of_the_pinned_reference(copy)
    candidate = _angles(5.0)[0]
    similarity = F.normalize(emb, dim=-1) @ candidate
    assert float(similarity[0] - similarity[1]) > 10 * EVICTION_TIE_TOLERANCE
    action, new_emb = bank.update(_rollout(3), candidate, emb, "redundancy", torch.Generator())
    assert action == "rejected" and bank.entries[1].token_ids == [1] and torch.equal(new_emb, emb)


def test_near_ties_between_unpinned_entries_go_to_the_lower_index() -> None:
    bank, emb = _full_bank((45.0, 60.0, 120.0, 200.0))
    candidate = _angles(90.0 + math.degrees(5e-7))[0]
    similarity = F.normalize(emb, dim=-1) @ candidate
    assert 0.0 < float(similarity[2] - similarity[1]) < EVICTION_TIE_TOLERANCE
    action, new_emb = bank.update(_rollout(9), candidate, emb, "redundancy", torch.Generator())
    assert action == "replaced" and bank.entries[1].token_ids == [9]
    assert _max_pairwise(new_emb) < _max_pairwise(emb)


def test_fifo_evicts_the_oldest_non_pinned_entry() -> None:
    bank, emb = _full_bank((0.0, 90.0, 180.0))
    generator = torch.Generator()
    for tag, expected_slot in [(10, 1), (11, 2), (12, 1), (13, 2)]:
        action, emb = bank.update(_rollout(tag), _angles(float(tag))[0], emb, "fifo", generator)
        assert action == "replaced"
        assert bank.entries[expected_slot].token_ids == [tag]
        assert bank.entries[0].pinned
        assert torch.allclose(emb[expected_slot], _angles(float(tag))[0])
    assert [entry.insertion_order for entry in bank.entries] == [0, 5, 6]


def _random_eviction_slots(seed: int, updates: int) -> list[int]:
    bank, emb = _full_bank((0.0, 90.0, 180.0, 270.0))
    generator = torch.Generator().manual_seed(seed)
    slots = []
    for tag in range(100, 100 + updates):
        action, emb = bank.update(_rollout(tag), _angles(float(tag))[0], emb, "random", generator)
        assert action == "replaced" and bank.entries[0].pinned
        slots.append(next(k for k, entry in enumerate(bank.entries) if entry.token_ids == [tag]))
    return slots


def test_random_eviction_is_uniform_over_non_pinned_entries_and_reproducible() -> None:
    slots = _random_eviction_slots(seed=0, updates=600)
    counts = Counter(slots)
    assert set(counts) == {1, 2, 3}
    assert all(150 < count < 250 for count in counts.values())
    assert slots == _random_eviction_slots(seed=0, updates=600)
    assert slots != _random_eviction_slots(seed=1, updates=600)


@pytest.mark.parametrize("rule", ["redundancy", "fifo", "random"])
def test_bank_of_capacity_one_keeps_only_the_pinned_reference(rule: str) -> None:
    bank = ReferenceBank(1, _pinned())
    emb = _angles(0.0)
    action, new_emb = bank.update(_rollout(1), _angles(120.0)[0], emb, rule, torch.Generator())
    assert action == "rejected" and len(bank) == 1 and torch.equal(new_emb, emb)


def test_update_validates_its_inputs() -> None:
    bank, emb = _full_bank((0.0, 90.0))
    with pytest.raises(ValueError):
        bank.update(_rollout(5), _angles(10.0)[0], emb[:1], "redundancy", torch.Generator())
    with pytest.raises(ValueError):
        bank.update(_rollout(5), _angles(10.0)[0], emb, "lru", torch.Generator())
    with pytest.raises(ValueError):
        bank.update(_pinned(), _angles(10.0)[0], emb, "redundancy", torch.Generator())


def test_similarity_retrieval_returns_the_most_similar_entry() -> None:
    bank, emb = _full_bank((0.0, 90.0, 180.0))
    rollouts = _angles(10.0, 80.0, 170.0, 200.0, 45.0)
    anchors = bank.retrieve(rollouts, emb, "similarity", torch.Generator())
    assert anchors.dtype == torch.long
    assert anchors.tolist() == [0, 1, 2, 2, 0]


def test_random_retrieval_is_uniform_over_the_bank_and_reproducible() -> None:
    bank, emb = _full_bank((0.0, 90.0, 180.0))
    rollouts = _angles(*([10.0] * 3000))
    anchors = bank.retrieve(rollouts, emb, "random", torch.Generator().manual_seed(3))
    counts = Counter(anchors.tolist())
    assert set(counts) == {0, 1, 2} and all(900 < count < 1100 for count in counts.values())
    again = bank.retrieve(rollouts, emb, "random", torch.Generator().manual_seed(3))
    assert torch.equal(anchors, again)
    with pytest.raises(ValueError):
        bank.retrieve(rollouts, emb, "nearest", torch.Generator())


@pytest.mark.parametrize("seed", range(5))
def test_accepted_exchange_never_increases_the_maximum_pairwise_similarity(seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    accepted = strictly_lower = 0
    for _ in range(30):
        dim = int(torch.randint(2, 6, (1,), generator=generator))
        capacity = int(torch.randint(2, 7, (1,), generator=generator))
        bank = ReferenceBank(capacity, _pinned())
        emb = F.normalize(torch.randn(1, dim, generator=generator), dim=-1)
        for tag in range(1, 40):
            candidate = F.normalize(torch.randn(dim, generator=generator), dim=0)
            full = len(bank) == capacity
            before = _max_pairwise(emb) if full else None
            entries_before = list(bank.entries)
            action, new_emb = bank.update(_rollout(tag), candidate, emb, "redundancy", generator)
            if action == "replaced":
                assert before is not None and len(bank) == capacity and bank.entries[0].pinned
                after = _max_pairwise(new_emb)
                assert after <= before + 1e-6
                accepted += 1
                slot = next(k for k, entry in enumerate(bank.entries) if entry is not entries_before[k])
                unit = F.normalize(emb, dim=-1)
                pairs = [(a, b) for a in range(capacity) for b in range(a + 1, capacity)]
                top = [pair for pair in pairs if float(unit[pair[0]] @ unit[pair[1]]) >= before - 1e-6]
                if all(slot in pair for pair in top):
                    assert after < before
                    strictly_lower += 1
            elif full:
                assert action == "rejected" and bank.entries == entries_before
            emb = new_emb
    assert accepted > 0 and strictly_lower > 0


def test_bank_survives_a_json_roundtrip() -> None:
    bank, emb = _full_bank((0.0, 90.0, 100.0))
    bank.update(_rollout(9), _angles(200.0)[0], emb, "redundancy", torch.Generator())
    restored = ReferenceBank.from_dict(json.loads(json.dumps(bank.to_dict())))
    assert restored.capacity == bank.capacity and restored.entries == bank.entries
    assert restored.redundancy == pytest.approx(bank.redundancy)
    assert restored.neighbor == bank.neighbor and restored.action_counts == bank.action_counts
    single = ReferenceBank.from_dict(json.loads(json.dumps(ReferenceBank(3, _pinned()).to_dict())))
    assert single.redundancy == [-math.inf] and single.neighbor == [-1]


class _CharTokenizer:
    eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [ord(char) for char in text]
        return [0, *ids] if add_special_tokens else ids


def _example(uid: str, reference: str = "x = 2") -> Example:
    return Example(uid=uid, prompt="Question:\nq\nAnswer reasoning:\n", reference=reference, target=2)


def test_store_creates_each_bank_once_with_the_reference_pinned() -> None:
    store = BankStore(rank=0)
    bank = store.get_or_create(_example("a"), _CharTokenizer(), capacity=5)
    pinned = bank.entries[0]
    assert pinned.pinned and pinned.source == "reference" and pinned.text == "x = 2"
    assert pinned.token_ids == [ord(char) for char in "x = 2"] + [1]
    assert store.get_or_create(_example("a"), _CharTokenizer(), capacity=5) is bank
    with pytest.raises(ValueError):
        store.get_or_create(_example("a"), _CharTokenizer(), capacity=3)
    with pytest.raises(ValueError):
        store.get_or_create(_example("a", reference="x = 3"), _CharTokenizer(), capacity=5)


def test_store_saves_and_loads_per_rank_files(tmp_path) -> None:
    store = BankStore(rank=2)
    generator = torch.Generator().manual_seed(0)
    for uid in ("b", "a"):
        bank = store.get_or_create(_example(uid), _CharTokenizer(), capacity=3)
        emb = _angles(0.0)
        for tag, degrees in enumerate((90.0, 100.0, 200.0), start=1):
            _, emb = bank.update(_rollout(tag, step=7), _angles(degrees)[0], emb, "redundancy", generator)
    written = store.save(tmp_path / "step_7")
    assert written == tmp_path / "step_7" / "bank_rank2.json"
    restored = BankStore(rank=2)
    restored.load(tmp_path / "step_7")
    assert sorted(restored.banks) == ["a", "b"]
    for uid in ("a", "b"):
        assert restored.banks[uid].to_dict() == store.banks[uid].to_dict()
    explicit = tmp_path / "final" / "banks.json"
    assert store.save(explicit) == explicit
    other = BankStore(rank=0)
    other.load(explicit)
    assert other.banks["a"].to_dict() == store.banks["a"].to_dict()


def test_store_stats_report_sizes_and_cumulative_actions() -> None:
    store = BankStore(rank=0)
    assert store.stats()["num_banks"] == 0 and store.stats()["mean_size"] == 0.0
    generator = torch.Generator()
    full = store.get_or_create(_example("full"), _CharTokenizer(), capacity=3)
    emb = _angles(0.0)
    for tag, degrees in enumerate((90.0, 100.0, 200.0, 5.0), start=1):
        _, emb = full.update(_rollout(tag), _angles(degrees)[0], emb, "redundancy", generator)
    store.get_or_create(_example("empty"), _CharTokenizer(), capacity=3)
    stats = store.stats()
    assert stats["num_banks"] == 2 and stats["mean_size"] == 2.0 and stats["max_size"] == 3
    assert stats["full_fraction"] == 0.5
    assert (stats["inserted"], stats["replaced"], stats["rejected"]) == (2, 1, 1)
