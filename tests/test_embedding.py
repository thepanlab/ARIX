from __future__ import annotations

import random

import pytest
import torch
import torch.nn.functional as F

from arix.config import (
    ARIXConfig,
    ARIXMethodConfig,
    TrainingConfig,
    config_from_dict,
    load_run_config,
    read_yaml_mapping,
    save_config,
)
from arix.embedding import (
    RationaleProjector,
    lcs_mask,
    masked_subset_embedding,
    pooling_weights,
    rationale_embedding,
)

PROMPTS = [[5, 6, 7], [8, 9], [11, 12, 13, 14]]
RESPONSES = [[10, 11, 12, 2], [13, 14], [15, 16, 17, 18, 19, 2]]
PAD_ID = 0


@pytest.fixture(scope="module")
def tiny_lm() -> torch.nn.Module:
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        vocab_size=37,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = transformers.LlamaForCausalLM(config)
    return model.eval()


def _definition_embedding(model: torch.nn.Module, prompt: list[int], response: list[int], projector) -> torch.Tensor:
    log_probs, hidden = [], []
    for i, token in enumerate(response):
        out = model(input_ids=torch.tensor([prompt + response[: i + 1]]), output_hidden_states=True)
        log_probs.append(torch.log_softmax(out.logits[0, -2].float(), dim=-1)[token])
        hidden.append(out.hidden_states[-1][0, -1])
    weights = torch.softmax(torch.stack(log_probs), dim=0)
    return F.normalize((weights[:, None] * projector(torch.stack(hidden))).sum(dim=0), dim=0)


def test_softmax_weights_cover_rationale_positions_only() -> None:
    torch.manual_seed(0)
    log_probs = -3.0 * torch.rand(3, 7)
    mask = torch.tensor([[0, 0, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1], [0, 1, 0, 1, 0, 1, 0]], dtype=torch.bool)
    weights = pooling_weights(log_probs, mask, "softmax")
    assert weights.dtype == torch.float32
    assert torch.all(weights[~mask] == 0)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3))
    for row in range(3):
        positions = mask[row].nonzero().squeeze(-1)
        assert torch.allclose(weights[row, positions], torch.softmax(log_probs[row, positions], dim=0))
    polluted = log_probs.masked_fill(~mask, 50.0)
    assert torch.allclose(pooling_weights(polluted, mask, "softmax"), weights)


def test_mean_pooling_is_uniform_over_rationale_positions() -> None:
    log_probs = torch.tensor([[-0.1, -5.0, -2.0, -0.3]])
    mask = torch.tensor([[False, True, True, True]])
    weights = pooling_weights(log_probs, mask, "mean")
    assert torch.allclose(weights, torch.tensor([[0.0, 1 / 3, 1 / 3, 1 / 3]]))


def test_unknown_pooling_is_rejected() -> None:
    with pytest.raises(ValueError):
        pooling_weights(torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool), "max")


def test_empty_rationale_gives_zero_weights_and_zero_embedding() -> None:
    projector = RationaleProjector(4, proj_dim=3)
    mask = torch.zeros(1, 5, dtype=torch.bool)
    assert torch.all(pooling_weights(torch.randn(1, 5), mask) == 0)
    embedding = rationale_embedding(torch.randn(1, 5, 4), torch.randn(1, 5), mask, projector)
    assert torch.all(embedding == 0)


@pytest.mark.parametrize("pooling", ["softmax", "mean"])
def test_embedding_matches_equation_4_and_has_unit_norm(pooling: str) -> None:
    torch.manual_seed(1)
    hidden = torch.randn(2, 6, 16, dtype=torch.bfloat16)
    log_probs = -4.0 * torch.rand(2, 6)
    mask = torch.tensor([[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 0, 0]], dtype=torch.bool)
    projector = RationaleProjector(16, proj_dim=8, seed=3)
    embedding = rationale_embedding(hidden, log_probs, mask, projector, pooling)
    assert embedding.shape == (2, 8) and embedding.dtype == torch.float32
    assert torch.allclose(embedding.norm(dim=-1), torch.ones(2))
    weight, bias = projector.linear.weight, projector.linear.bias
    for row in range(2):
        positions = mask[row].nonzero().squeeze(-1)
        u = log_probs[row, positions]
        w = torch.softmax(u, dim=0) if pooling == "softmax" else torch.full_like(u, 1 / len(u))
        projected = hidden[row, positions].float() @ weight.T + bias
        expected = F.normalize((w[:, None] * projected).sum(dim=0), dim=0)
        assert torch.allclose(embedding[row], expected, atol=1e-6)


def test_weight_of_position_t_multiplies_hidden_state_t() -> None:
    torch.manual_seed(2)
    hidden = torch.randn(1, 5, 8)
    log_probs = torch.full((1, 5), -60.0)
    log_probs[0, 3] = 0.0
    projector = RationaleProjector(8, proj_dim=4)
    embedding = rationale_embedding(hidden, log_probs, torch.ones(1, 5, dtype=torch.bool), projector)
    assert torch.allclose(embedding[0], F.normalize(projector(hidden[0, 3]), dim=0), atol=1e-6)


def test_projector_is_frozen_float32_with_seeded_default_initialization() -> None:
    projector = RationaleProjector(32, proj_dim=64, seed=0)
    assert all(not p.requires_grad for p in projector.parameters())
    assert projector.linear.weight.dtype == torch.float32 and projector.linear.weight.shape == (64, 32)
    assert torch.equal(projector.linear.weight, RationaleProjector(32, 64, seed=0).linear.weight)
    assert not torch.equal(projector.linear.weight, RationaleProjector(32, 64, seed=1).linear.weight)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        default = torch.nn.Linear(32, 64)
    assert torch.equal(projector.linear.weight, default.weight)
    assert torch.equal(projector.linear.bias, default.bias)
    assert projector(torch.randn(3, 32, dtype=torch.bfloat16)).dtype == torch.float32


def test_projector_construction_leaves_global_rng_untouched() -> None:
    state = torch.get_rng_state()
    RationaleProjector(8, proj_dim=4, seed=5)
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize("seed", [1111, 2222, 5555])
def test_projector_seed_is_the_run_seed_unless_set(code_root, tmp_path, seed: int) -> None:
    configs = code_root / "math_reasoning" / "configs"
    base = configs / "svamp_qwen25_7b.yaml"
    for overlays in ([], ["arix_wo_rb"], ["proj32"]):
        paths = [base, *(configs / "ablations" / f"{name}.yaml" for name in overlays)]
        cfg = load_run_config(paths, [f"training.seed={seed}"])
        assert cfg.arix.proj_seed == seed
    assert load_run_config([base], [f"training.seed={seed}", "arix.proj_seed=7"]).arix.proj_seed == 7
    assert load_run_config([base], [f"training.seed={seed}", "arix.proj_seed=null"]).arix.proj_seed == seed
    save_config(cfg, tmp_path / "config.yaml")
    saved = read_yaml_mapping(tmp_path / "config.yaml")
    assert saved["arix"]["proj_seed"] == seed and config_from_dict(saved).arix.proj_seed == seed
    assert config_from_dict({**saved, "training": {**saved["training"], "seed": 1}}).arix.proj_seed == seed
    assert ARIXConfig().arix.proj_seed == TrainingConfig().seed
    assert ARIXConfig(training=TrainingConfig(seed=seed)).arix.proj_seed == seed
    assert ARIXConfig(arix=ARIXMethodConfig(proj_seed=3), training=TrainingConfig(seed=seed)).arix.proj_seed == 3


def test_projector_save_load_roundtrip(tmp_path) -> None:
    projector = RationaleProjector(16, proj_dim=32, seed=7)
    path = tmp_path / "step_1" / "projector.pt"
    projector.save(path)
    loaded = RationaleProjector.load(path)
    assert (loaded.hidden_size, loaded.proj_dim, loaded.seed) == (16, 32, 7)
    assert torch.equal(loaded.linear.weight, projector.linear.weight)
    assert torch.equal(loaded.linear.bias, projector.linear.bias)
    assert all(not p.requires_grad for p in loaded.parameters())


def test_gradients_reach_hidden_states_and_log_probs_but_not_the_projector() -> None:
    torch.manual_seed(3)
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    log_probs = (-2.0 * torch.rand(2, 5)).requires_grad_(True)
    mask = torch.tensor([[0, 1, 1, 1, 0], [0, 0, 1, 1, 1]], dtype=torch.bool)
    projector = RationaleProjector(8, proj_dim=4)
    embedding = rationale_embedding(hidden, log_probs, mask, projector)
    (embedding * torch.randn(2, 4)).sum().backward()
    assert hidden.grad is not None and log_probs.grad is not None
    assert torch.any(hidden.grad[mask] != 0) and torch.any(log_probs.grad[mask] != 0)
    assert torch.all(hidden.grad[~mask] == 0) and torch.all(log_probs.grad[~mask] == 0)
    assert projector.linear.weight.grad is None and projector.linear.bias.grad is None


def test_embedding_follows_next_token_alignment_on_a_causal_lm(tiny_lm) -> None:
    projector = RationaleProjector(16, proj_dim=8)
    lengths = [len(p) + len(r) for p, r in zip(PROMPTS, RESPONSES, strict=True)]
    input_ids = torch.full((len(PROMPTS), max(lengths)), PAD_ID)
    attention_mask = torch.zeros_like(input_ids)
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for row, (prompt, response) in enumerate(zip(PROMPTS, RESPONSES, strict=True)):
        input_ids[row, : lengths[row]] = torch.tensor(prompt + response)
        attention_mask[row, : lengths[row]] = 1
        mask[row, len(prompt) : lengths[row]] = True
    with torch.no_grad():
        out = tiny_lm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        token_logprobs = torch.zeros(input_ids.shape)
        token_logprobs[:, 1:] = (
            torch.log_softmax(out.logits[:, :-1].float(), dim=-1).gather(-1, input_ids[:, 1:, None]).squeeze(-1)
        )
        embedding = rationale_embedding(out.hidden_states[-1], token_logprobs, mask, projector)
        for row, (prompt, response) in enumerate(zip(PROMPTS, RESPONSES, strict=True)):
            expected = _definition_embedding(tiny_lm, prompt, response, projector)
            assert torch.allclose(embedding[row], expected, atol=1e-5)


def test_modeling_forward_feeds_the_embedding_with_aligned_positions(tiny_lm) -> None:
    modeling = pytest.importorskip("arix.modeling")
    projector = RationaleProjector(16, proj_dim=8)
    batch = modeling.build_sequence_batch(PROMPTS, RESPONSES, PAD_ID)
    with torch.no_grad():
        token_logprobs, hidden = modeling.forward_logprobs_hidden(tiny_lm, batch, need_hidden=True)
        embedding = rationale_embedding(hidden, token_logprobs, batch.response_mask, projector)
        for row, (prompt, response) in enumerate(zip(PROMPTS, RESPONSES, strict=True)):
            expected = _definition_embedding(tiny_lm, prompt, response, projector)
            assert torch.allclose(embedding[row], expected, atol=1e-5)


def test_subset_embedding_pools_over_the_subset_only() -> None:
    torch.manual_seed(4)
    hidden = torch.randn(1, 6, 8)
    log_probs = -2.0 * torch.rand(1, 6)
    mask = torch.tensor([[False, True, True, True, True, True]])
    subset = torch.tensor([[True, True, False, True, False, False]])
    projector = RationaleProjector(8, proj_dim=4)
    lcs = masked_subset_embedding(hidden, log_probs, mask, subset, projector)
    exc = masked_subset_embedding(hidden, log_probs, mask, ~subset, projector)
    assert torch.allclose(lcs, rationale_embedding(hidden, log_probs, mask & subset, projector))
    assert torch.allclose(exc, rationale_embedding(hidden, log_probs, mask & ~subset, projector))
    only_first = log_probs.clone()
    only_first[0, 1] = 0.0
    only_first[0, 3] = -80.0
    assert torch.allclose(
        masked_subset_embedding(hidden, only_first, mask, subset, projector)[0],
        F.normalize(projector(hidden[0, 1]), dim=0),
        atol=1e-6,
    )


def _lcs_length(a: list[int], b: list[int]) -> int:
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    return table[-1][-1]


def _is_subsequence(seq: list[int], of: list[int]) -> bool:
    remaining = iter(of)
    return all(any(token == other for other in remaining) for token in seq)


def test_lcs_mask_examples() -> None:
    assert lcs_mask([1, 2, 3, 4], [2, 4]) == [False, True, False, True]
    assert lcs_mask([7, 8], [9]) == [False, False]
    assert lcs_mask([], [1, 2]) == []
    assert lcs_mask([1, 2], []) == [False, False]
    assert lcs_mask([3, 1, 3], [3, 1, 3]) == [True, True, True]


def test_lcs_mask_selects_a_longest_common_subsequence() -> None:
    rng = random.Random(0)
    for _ in range(300):
        rollout = [rng.randrange(4) for _ in range(rng.randrange(0, 15))]
        anchor = [rng.randrange(4) for _ in range(rng.randrange(0, 15))]
        selected = lcs_mask(rollout, anchor)
        assert len(selected) == len(rollout)
        chosen = [token for token, keep in zip(rollout, selected, strict=True) if keep]
        assert len(chosen) == _lcs_length(rollout, anchor)
        assert _is_subsequence(chosen, anchor)


def test_lcs_mask_on_long_sequences() -> None:
    rng = random.Random(1)
    rollout = [rng.randrange(20) for _ in range(400)]
    anchor = [rng.randrange(20) for _ in range(300)]
    selected = lcs_mask(rollout, anchor)
    chosen = [token for token, keep in zip(rollout, selected, strict=True) if keep]
    assert len(chosen) == _lcs_length(rollout, anchor)
    assert _is_subsequence(chosen, anchor)
