from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from accelerate import Accelerator
from transformers import PreTrainedTokenizerBase

from arix.bank import BankStore, ReferenceBank
from arix.config import ARIXConfig, load_config
from arix.embedding import RationaleProjector
from arix.modeling import load_policy_and_tokenizer, load_reference_model
from arix.task import Example, Task
from arix.trainer import ARIXTrainer
from clinical_trials.data import ClinicalDataConfig, load_training_pairs
from clinical_trials.task import ClinicalTask
from clinical_trials.task import build_examples as build_clinical_examples
from math_reasoning.task import MathTask, task_from_config
from math_reasoning.task import build_examples as build_math_examples

NUM_PROMPTS = 2
GROUP_SIZE = 4
BANK_SIZE = 3
ROLLOUTS_PER_STEP = NUM_PROMPTS * GROUP_SIZE
FINITE_KEYS = ("loss", "grpo_loss", "mean_reward", "fraction_correct", "kl", "grad_norm")
CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "projector.pt",
    "config.yaml",
    "trainer_state.json",
)

ExampleBuilder = Callable[[PreTrainedTokenizerBase], list[Example]]


class AcceptEverything:
    name = "accept_everything"

    def verify_batch(self, examples: list[Example], responses: list[str]) -> list[bool]:
        return [True] * len(responses)

    def extract_batch(self, examples: list[Example], responses: list[str]) -> list[Any]:
        return [None] * len(responses)


@pytest.fixture(scope="module")
def math_data(math_fixtures_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    records = json.loads((math_fixtures_dir / "gsm8k_python_sdp_sample.json").read_text(encoding="utf-8"))
    train_file = tmp_path_factory.mktemp("math_data") / "gsm8k_python_sdp.json"
    train_file.write_text(json.dumps(records[:NUM_PROMPTS]), encoding="utf-8")
    return {
        "dataset": "gsm8k",
        "engine": "python",
        "train_file": str(train_file),
        "exec_timeout": 2.0,
        "exec_workers": 2,
    }


@pytest.fixture(scope="module")
def math_task(math_data: dict[str, Any]) -> Iterator[MathTask]:
    task = task_from_config(math_data)
    yield task
    task.close()


@pytest.fixture(scope="module")
def clinical_data(clinical_fixtures_dir: Path) -> dict[str, Any]:
    matching = clinical_fixtures_dir / "matching_results_trec_2021_SYNTHETIC.json"
    return {
        "dataset_dir": str(clinical_fixtures_dir),
        "train_corpora": ["trec_2021"],
        "test_corpora": [],
        "matching_results": {"trec_2021": str(matching)},
    }


def write_config(directory: Path, model_dir: Path, data: dict[str, Any], method: str = "arix") -> Path:
    config = {
        "method": method,
        "model": {"name_or_path": str(model_dir), "torch_dtype": "float32", "gradient_checkpointing": False},
        "data": data,
        "training": {
            "epochs": 2,
            "batch_size": NUM_PROMPTS,
            "group_size": GROUP_SIZE,
            "micro_batch_size": 4,
            "generation_batch_size": 4,
            "max_steps": 3,
        },
        "generation": {"max_new_tokens": 8},
        "arix": {"bank_size": BANK_SIZE, "retrieval_start_epoch": 2},
        "output": {"output_dir": str(directory / "run"), "save_every_steps": 1},
    }
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def math_examples(cfg: ARIXConfig) -> ExampleBuilder:
    return lambda tokenizer: build_math_examples(cfg.data, tokenizer, split="train")


def clinical_examples(cfg: ARIXConfig) -> ExampleBuilder:

    def build(tokenizer: PreTrainedTokenizerBase) -> list[Example]:
        pairs, summary = load_training_pairs(ClinicalDataConfig.from_dict(cfg.data))
        assert summary["references"]["kept"] == len(pairs) > NUM_PROMPTS
        examples = build_clinical_examples(pairs, tokenizer)
        return sorted(examples, key=lambda example: len(example.prompt))[:NUM_PROMPTS]

    return build


def run_training(cfg: ARIXConfig, task: Task, build_examples: ExampleBuilder) -> tuple[Path, list[Example]]:
    accelerator = Accelerator(cpu=True)
    model, tokenizer = load_policy_and_tokenizer(cfg.model)
    ref_model = load_reference_model(cfg.model)
    examples = build_examples(tokenizer)
    assert len(examples) == NUM_PROMPTS
    final = ARIXTrainer(cfg, task, examples, model, tokenizer, ref_model, accelerator).train()
    return final, examples


def read_log(cfg: ARIXConfig) -> list[dict[str, Any]]:
    path = Path(cfg.output.output_dir) / "train_log.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_records(records: Sequence[dict[str, Any]], cfg: ARIXConfig, steps: int) -> None:
    assert [record["step"] for record in records] == list(range(1, steps + 1))
    for record in records:
        for key in FINITE_KEYS:
            assert math.isfinite(record[key]), f"step {record['step']}: {key} = {record[key]}"
        if cfg.arix.use_contrastive:
            assert math.isfinite(record["pos_loss"]) and record["pos_loss"] >= 0.0
        if cfg.needs_embeddings:
            assert -1.0 - 1e-5 <= record["mean_anchor_cos"] <= 1.0 + 1e-5
        if cfg.arix.use_bank:
            correct = round(record["fraction_correct"] * ROLLOUTS_PER_STEP)
            assert record["bank_inserted"] + record["bank_replaced"] + record["bank_rejected"] == correct
            assert 1 <= record["bank_mean_size"] <= BANK_SIZE
        else:
            assert "bank_inserted" not in record


def load_banks(checkpoint: Path) -> dict[str, ReferenceBank]:
    store = BankStore(rank=0)
    store.load(checkpoint)
    return store.banks


def check_checkpoint(checkpoint: Path, cfg: ARIXConfig, examples: Sequence[Example]) -> dict[str, ReferenceBank]:
    for name in CHECKPOINT_FILES:
        assert (checkpoint / name).is_file(), f"{checkpoint / name} is missing"
    hidden_size = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))["hidden_size"]
    projector = RationaleProjector.load(checkpoint / "projector.pt")
    saved = yaml.safe_load((checkpoint / "config.yaml").read_text(encoding="utf-8"))
    assert cfg.arix.proj_seed == cfg.training.seed == saved["arix"]["proj_seed"] == projector.seed
    seeded = RationaleProjector(hidden_size, cfg.arix.proj_dim, cfg.training.seed)
    torch.testing.assert_close(projector.linear.weight, seeded.linear.weight, rtol=0.0, atol=0.0)
    assert not any(parameter.requires_grad for parameter in projector.parameters())
    if not cfg.needs_embeddings:
        return {}
    banks = load_banks(checkpoint)
    assert sorted(banks) == sorted(example.uid for example in examples)
    for example in examples:
        bank = banks[example.uid]
        pinned = bank.entries[0]
        assert pinned.pinned and pinned.source == "reference" and pinned.text == example.reference
        assert 1 <= len(bank) <= BANK_SIZE
        assert all(not entry.pinned and entry.source == "rollout" for entry in bank.entries[1:])
    return banks


def test_arix_trains_on_math(
    tiny_model_dir: Path, math_data: dict[str, Any], math_task: MathTask, tmp_path: Path
) -> None:
    cfg = load_config(write_config(tmp_path, tiny_model_dir, math_data))
    final, examples = run_training(cfg, math_task, math_examples(cfg))
    records = read_log(cfg)
    check_records(records, cfg, steps=2)
    assert [record["epoch"] for record in records] == [1, 2]
    assert records[0]["pinned_anchor_fraction"] == 1.0
    check_checkpoint(final, cfg, examples)
    for step in (1, 2):
        assert (Path(cfg.output.output_dir) / f"step_{step}" / "projector.pt").is_file()
    state = json.loads((final / "trainer_state.json").read_text(encoding="utf-8"))
    assert state["global_step"] == 2 and state["epoch"] == 2 and state["method"] == "arix"


def test_arix_trains_on_clinical_pairs(tiny_model_dir: Path, clinical_data: dict[str, Any], tmp_path: Path) -> None:
    cfg = load_config(write_config(tmp_path, tiny_model_dir, clinical_data))
    final, examples = run_training(cfg, ClinicalTask(), clinical_examples(cfg))
    assert all(not example.meta["add_special_tokens"] for example in examples)
    assert all(example.reference.rstrip().endswith(f"Final label: {example.target}") for example in examples)
    records = read_log(cfg)
    check_records(records, cfg, steps=2)
    assert records[0]["pinned_anchor_fraction"] == 1.0
    check_checkpoint(final, cfg, examples)


def test_correct_rollouts_fill_the_banks_then_compete_for_their_slots(
    tiny_model_dir: Path, math_data: dict[str, Any], tmp_path: Path
) -> None:
    cfg = load_config(write_config(tmp_path, tiny_model_dir, math_data))
    final, examples = run_training(cfg, AcceptEverything(), math_examples(cfg))
    first, second = read_log(cfg)
    check_records([first, second], cfg, steps=2)
    assert first["fraction_correct"] == second["fraction_correct"] == 1.0
    assert first["mean_reward"] == 1.0
    assert first["bank_inserted"] == NUM_PROMPTS * (BANK_SIZE - 1)
    assert first["bank_replaced"] + first["bank_rejected"] == NUM_PROMPTS * (GROUP_SIZE - BANK_SIZE + 1)
    assert first["bank_mean_size"] == BANK_SIZE
    assert first["pinned_anchor_fraction"] == 1.0
    assert second["bank_inserted"] == 0
    assert second["bank_replaced"] + second["bank_rejected"] == ROLLOUTS_PER_STEP
    banks = check_checkpoint(final, cfg, examples)
    for bank in banks.values():
        assert len(bank) == BANK_SIZE
        assert all(math.isfinite(score) for score in bank.redundancy)
        assert all(0 <= neighbor < BANK_SIZE and neighbor != k for k, neighbor in enumerate(bank.neighbor))
        assert all(entry.added_step in (1, 2) for entry in bank.entries[1:])


@pytest.mark.parametrize("method", ["arix_wo_rb", "grpo", "dr_grpo"])
def test_method_presets_run_one_step(
    method: str, tiny_model_dir: Path, math_data: dict[str, Any], math_task: MathTask, tmp_path: Path
) -> None:
    cfg = load_config(write_config(tmp_path, tiny_model_dir, math_data, method=method), ["training.max_steps=1"])
    a, t = cfg.arix, cfg.training
    assert not a.use_bank
    assert a.use_contrastive == a.use_partial_reward == (method == "arix_wo_rb")
    assert t.std_normalize == t.length_normalize == (method != "dr_grpo")
    final, examples = run_training(cfg, math_task, math_examples(cfg))
    (record,) = read_log(cfg)
    check_records([record], cfg, steps=1)
    if method == "arix_wo_rb":
        assert record["pinned_anchor_fraction"] == 1.0
    else:
        assert record["pos_loss"] is None and record["mean_anchor_cos"] is None
        assert record["mean_reward"] == record["fraction_correct"]
    check_checkpoint(final, cfg, examples)
