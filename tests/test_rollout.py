from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
from accelerate import Accelerator
from tiny_architectures import FAMILIES, GPT_OSS_SLIDING_WINDOW, tiny_causal_lm
from torch import nn

import arix.trainer as trainer_module
from arix.config import GenerationConfig, load_config
from arix.modeling import load_policy_and_tokenizer, load_reference_model
from arix.rollout import generate_batch, zero3_generation
from arix.task import Example
from arix.trainer import ARIXTrainer

VOCAB, PAD_ID, EOS_ID = 97, 0, 1
TOKENIZER: Any = SimpleNamespace(pad_token_id=PAD_ID, eos_token_id=EOS_ID)


def prompts() -> list[list[int]]:
    generator = torch.Generator().manual_seed(1)
    return [torch.randint(3, VOCAB, (length,), generator=generator).tolist() for length in (37, 5, 22, 30)]


@pytest.mark.parametrize("family", FAMILIES)
def test_chunked_prefill_gives_the_same_greedy_responses(family: str) -> None:
    model = tiny_causal_lm(family, VOCAB, seed=0)
    widths: list[int] = []
    model.register_forward_pre_hook(
        lambda _module, _args, kwargs: widths.append(kwargs["input_ids"].shape[1]), with_kwargs=True
    )
    greedy = GenerationConfig(max_new_tokens=10, temperature=0.0)
    expected = generate_batch(model, TOKENIZER, prompts(), greedy, generation_batch_size=4)
    assert widths[0] == 37 and all(width == 1 for width in widths[1:])
    assert any(len(response) > 1 for response in expected)
    for chunk in (1, 3, GPT_OSS_SLIDING_WINDOW, 16, 64):
        widths.clear()
        chunked = GenerationConfig(max_new_tokens=10, temperature=0.0, prefill_chunk_size=chunk)
        assert generate_batch(model, TOKENIZER, prompts(), chunked, generation_batch_size=4) == expected, chunk
        prefill = -(-37 // chunk)
        assert widths[:prefill] == [min(chunk, 37 - start) for start in range(0, 37, chunk)], chunk
    with pytest.raises(ValueError, match="prefill_chunk_size"):
        generate_batch(model, TOKENIZER, prompts(), GenerationConfig(prefill_chunk_size=0), generation_batch_size=4)


SYNCED_PROMPT_LENGTHS = {0: (10, 6, 20, 3), 1: (30, 25, 9, 12)}
SYNCED_WORKER = """
import json, os, sys
from datetime import timedelta
from types import SimpleNamespace
sys.path[:0] = [{code_root!r}, {tests_dir!r}]
import torch
import torch.distributed as dist
from tiny_architectures import tiny_causal_lm
from arix.config import GenerationConfig
from arix.rollout import generate_batch

dist.init_process_group("gloo", timeout=timedelta(seconds=60))
rank = dist.get_rank()
tokenizer = SimpleNamespace(pad_token_id={pad_id}, eos_token_id={eos_id})
model = tiny_causal_lm("qwen2", {vocab}, seed=0)
generator = torch.Generator().manual_seed(10 + rank)
prompts = [torch.randint(3, {vocab}, (length,), generator=generator).tolist() for length in {lengths}[rank]]

collectives, widths, hooks = [], [], []
all_reduce = dist.all_reduce

def logged_all_reduce(tensor, *args, **kwargs):
    collectives.append("all_reduce")
    return all_reduce(tensor, *args, **kwargs)

def gather_parameters(module, _args):
    collectives.append("all_gather " + type(module).__name__)
    dist.all_gather([torch.zeros(4) for _ in range(dist.get_world_size())], torch.zeros(4))

dist.all_reduce = logged_all_reduce
hooks += [module.register_forward_pre_hook(gather_parameters) for module in model.modules()
          if any(True for _ in module.parameters(recurse=False))]
hooks.append(model.register_forward_pre_hook(
    lambda _module, _args, kwargs: widths.append(kwargs["input_ids"].shape[1]), with_kwargs=True))
chunked = GenerationConfig(max_new_tokens=4, temperature=0.0, prefill_chunk_size=8)
responses = generate_batch(model, tokenizer, prompts, chunked, generation_batch_size=2, synced_gpus=True)
for hook in hooks:
    hook.remove()
dist.all_reduce = all_reduce
unchunked = GenerationConfig(max_new_tokens=4, temperature=0.0)
expected = generate_batch(model, tokenizer, prompts, unchunked, generation_batch_size=2)
with open(os.path.join({out_dir!r}, f"rank{{rank}}.json"), "w") as handle:
    json.dump({{"collectives": collectives, "widths": widths, "responses": responses, "expected": expected}}, handle)
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="launches two local processes with gloo")
def test_synced_processes_run_the_same_prefill_passes(tmp_path: Path) -> None:
    worker = tmp_path / "worker.py"
    worker.write_text(
        SYNCED_WORKER.format(
            code_root=str(Path(__file__).resolve().parent.parent),
            tests_dir=str(Path(__file__).resolve().parent),
            pad_id=PAD_ID,
            eos_id=EOS_ID,
            vocab=VOCAB,
            lengths=SYNCED_PROMPT_LENGTHS,
            out_dir=str(tmp_path),
        ),
        encoding="utf-8",
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    launch = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", "--master_addr", "127.0.0.1"]
    launch += ["--master_port", str(port), str(worker)]
    result = subprocess.run(
        launch,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-3000:]

    first, second = (json.loads((tmp_path / f"rank{rank}.json").read_text(encoding="utf-8")) for rank in (0, 1))
    assert first["collectives"] == second["collectives"]
    prefill = [width for width in first["widths"] if width > 1]
    assert prefill == [8, 8, 8, 6, 8, 8, 4]
    assert first["widths"] == second["widths"]
    for rank, outputs in enumerate((first, second)):
        assert len(outputs["responses"]) == len(SYNCED_PROMPT_LENGTHS[rank])
        assert outputs["responses"] == outputs["expected"], rank


class FakeHook:
    def __init__(self, events: list[str], name: str) -> None:
        self.events, self.name = events, name

    def remove(self) -> None:
        self.events.append(f"remove {self.name}")


def fake_zero3(events: list[str], version: str) -> tuple[ModuleType, nn.Module, SimpleNamespace]:

    @contextlib.contextmanager
    def gathered_parameters(params: list[nn.Parameter], **_kwargs: Any) -> Iterator[None]:
        events.append(f"gather {len(params)}")
        try:
            yield
        finally:
            events.append("partition")

    deepspeed = ModuleType("deepspeed")
    deepspeed.__version__ = version  # type: ignore[attr-defined]
    deepspeed.zero = SimpleNamespace(GatheredParameters=gathered_parameters)  # type: ignore[attr-defined]
    module = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    module.ds_external_parameters = lambda: iter([])  # type: ignore[assignment]
    for parameter in module.parameters():
        parameter.ds_active_sub_modules = {1, 2}  # type: ignore[attr-defined]
    coordinator = SimpleNamespace(
        is_invalid_trace=lambda: False, _invalidate_trace=lambda: events.append("invalidate trace")
    )
    offload = SimpleNamespace(
        module=module,
        forward_hooks=[FakeHook(events, "forward 0"), FakeHook(events, "forward 1")],
        backward_hooks=[FakeHook(events, "backward 0")],
        param_coordinator=coordinator,
        _register_deepspeed_module=lambda root: events.append("register " + ("root" if root is module else "?")),
        _register_hooks_recursively=lambda root: events.append(
            "register recursively " + ("root" if root is module else "?")
        ),
    )
    engine = nn.Module()
    engine.module = module
    engine.optimizer = SimpleNamespace(parameter_offload=offload)  # type: ignore[assignment]
    return deepspeed, engine, offload


@pytest.mark.parametrize(
    ("version", "register"), [("0.20.0", "register root"), ("0.16.3", "register recursively root")]
)
def test_zero3_generation_gathers_once_and_restores_the_hooks(
    version: str, register: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    deepspeed, engine, offload = fake_zero3(events, version)
    monkeypatch.setitem(sys.modules, "deepspeed", deepspeed)
    with zero3_generation(engine, gather=True) as gathered:
        assert gathered
        assert offload.forward_hooks == [] and offload.backward_hooks == []
        assert all(not p.ds_active_sub_modules for p in engine.parameters())  # type: ignore[attr-defined]
        events.append("generate")
    assert events == [
        "gather 4",
        "remove forward 0",
        "remove forward 1",
        "remove backward 0",
        "generate",
        "invalidate trace",
        register,
        "partition",
    ]

    events.clear()
    deepspeed, engine, offload = fake_zero3(events, version)
    monkeypatch.setitem(sys.modules, "deepspeed", deepspeed)
    with pytest.raises(RuntimeError, match="out of memory"), zero3_generation(engine, gather=True):
        raise RuntimeError("CUDA out of memory")
    assert events[-3:] == ["invalidate trace", register, "partition"]

    events.clear()
    with zero3_generation(engine, gather=False) as gathered:
        assert not gathered
    assert events == []


def test_the_trainer_generates_a_step_inside_one_zero3_gather(
    tiny_model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = load_config(
        None,
        [
            f"model.name_or_path={tiny_model_dir}",
            "model.torch_dtype=float32",
            "model.gradient_checkpointing=false",
            "training.batch_size=2",
            "training.group_size=2",
            "training.micro_batch_size=2",
            "training.generation_batch_size=2",
            "generation.max_new_tokens=3",
            f"output.output_dir={tmp_path / 'run'}",
        ],
    )
    examples = [
        Example(uid=f"u{i}", prompt=f"Question:\n{i} + 1?\n", reference=str(i + 1), target=i + 1) for i in range(2)
    ]
    model, tokenizer = load_policy_and_tokenizer(cfg.model)
    task = SimpleNamespace(
        name="stub",
        verify_batch=lambda _examples, texts: [True] * len(texts),
        extract_batch=lambda _examples, texts: [None] * len(texts),
    )
    trainer = ARIXTrainer(cfg, task, examples, model, tokenizer, load_reference_model(cfg.model), Accelerator(cpu=True))
    events: list[tuple[str, Any]] = []

    @contextlib.contextmanager
    def recording_context(engine: nn.Module, gather: bool) -> Iterator[bool]:
        assert engine is trainer.model
        events.append(("enter", gather))
        yield gather
        events.append(("exit", gather))

    def recording_generate(*args: Any, synced_gpus: bool = False, **kwargs: Any) -> list[list[int]]:
        events.append(("generate", synced_gpus))
        return generate_batch(*args, **kwargs)

    monkeypatch.setattr(trainer_module, "zero3_generation", recording_context)
    monkeypatch.setattr(trainer_module, "generate_batch", recording_generate)
    trainer.zero3 = True
    rollouts = trainer._sample(examples)
    assert len(rollouts.responses) == 4
    assert events == [("enter", True), ("generate", False), ("exit", True)]

    events.clear()
    trainer.cfg.generation.ds3_gather = False
    trainer._sample(examples)
    assert events == [("enter", False), ("generate", True), ("exit", False)]

    events.clear()
    trainer.zero3 = False
    trainer.cfg.generation.ds3_gather = True
    trainer._sample(examples)
    assert events == [("enter", False), ("generate", False), ("exit", False)]
