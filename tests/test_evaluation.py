from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch

import arix.evaluation
from arix.config import GenerationConfig, config_from_dict
from arix.evaluation import (
    PROCESS_GROUP_TIMEOUT,
    EvaluationSettings,
    add_evaluation_arguments,
    check_evaluation_arguments,
    evaluation_generation,
    evaluation_settings,
    metrics_file,
    shard_file,
    start_evaluation,
)
from arix.utils import read_json, write_json

TESTS_DIR = Path(__file__).resolve().parent
NL_FILE = TESTS_DIR / "fixtures" / "math" / "gsm8k_nl_sample.json"
WRONG = "Therefore, the answer is: 7"


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_evaluation_arguments(parser, "problem", 64)
    args = parser.parse_args(["--model", "m", *argv])
    check_evaluation_arguments(parser, args)
    return args


def _fixed_responses(text: str):

    def generate(_model, tokenizer, prompts, *_settings):
        return [tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id] for _ in prompts]

    return generate


def test_the_evaluation_waits_hours_for_its_processes_and_seeds_each_one(monkeypatch: pytest.MonkeyPatch) -> None:
    accelerator = start_evaluation(seed=1234)
    assert accelerator.init_handler is not None and accelerator.init_handler.timeout == PROCESS_GROUP_TIMEOUT
    assert PROCESS_GROUP_TIMEOUT >= timedelta(hours=12)
    assert torch.initial_seed() == 1234 + accelerator.process_index

    class SecondProcess(arix.evaluation.Accelerator):
        @property
        def process_index(self) -> int:
            return 1

    monkeypatch.setattr(arix.evaluation, "Accelerator", SecondProcess)
    start_evaluation(seed=1234)
    assert torch.initial_seed() == 1235


def test_batch_sizes_come_from_the_command_line_then_the_config_then_the_default() -> None:
    configured = config_from_dict({"evaluation": {"batch_size": 6, "embed_batch_size": 2}})
    assert evaluation_settings(_parse([]), None, 64) == EvaluationSettings(batch_size=64, embed_batch_size=4)
    assert evaluation_settings(_parse([]), config_from_dict({}), 16) == EvaluationSettings(batch_size=16)
    settings = evaluation_settings(_parse([]), configured, 64)
    assert (settings.batch_size, settings.embed_batch_size) == (6, 2)
    settings = evaluation_settings(_parse(["--batch_size", "3", "--embed_batch_size", "1"]), configured, 64)
    assert (settings.batch_size, settings.embed_batch_size) == (3, 1)
    settings = evaluation_settings(_parse([]), config_from_dict({"evaluation": {"batch_size": 5}}), 64)
    assert (settings.batch_size, settings.embed_batch_size) == (5, 4)
    sampled = evaluation_settings(_parse(["--num_samples", "4", "--temperature", "1", "--diversity"]), None, 64)
    assert (sampled.num_samples, sampled.diversity, sampled.self_bleu_n, sampled.self_bleu_tokenizer) == (
        4,
        True,
        4,
        "regex",
    )
    with pytest.raises(ValueError, match="batch sizes must be >= 1"):
        evaluation_settings(_parse([]), config_from_dict({"evaluation": {"embed_batch_size": 0}}), 64)


def test_the_decoding_is_the_runs_with_the_command_lines_sampling() -> None:
    cfg = config_from_dict(
        {"generation": {"max_new_tokens": 300, "temperature": 1.0, "top_p": 0.9, "top_k": 50, "prefill_chunk_size": 64}}
    )
    gen = evaluation_generation(_parse([]), cfg)
    assert (gen.max_new_tokens, gen.temperature, gen.top_p, gen.top_k, gen.prefill_chunk_size) == (300, 0.0, 1.0, 0, 64)
    gen = evaluation_generation(_parse(["--max_new_tokens", "7", "--temperature", "0.7", "--top_p", "0.95"]), cfg)
    assert (gen.max_new_tokens, gen.temperature, gen.top_p, gen.top_k, gen.prefill_chunk_size) == (7, 0.7, 0.95, 0, 64)
    without_config = evaluation_generation(_parse([]), None, 1024)
    assert without_config == GenerationConfig(max_new_tokens=1024, temperature=0.0, top_p=1.0, top_k=0)
    with pytest.raises(ValueError, match="max_new_tokens"):
        evaluation_generation(_parse([]), None)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--num_samples", "0"], "--num_samples must be >= 1"),
        (["--num_samples", "2"], "needs sampling"),
        (["--diversity"], "--diversity needs --num_samples >= 2"),
        (["--batch_size", "0"], "--batch_size must be >= 1"),
        (["--embed_batch_size", "0"], "--embed_batch_size must be >= 1"),
        (["--max_new_tokens", "0"], "--max_new_tokens must be >= 1"),
        (["--self_bleu_n", "0"], "--self_bleu_n must be >= 1"),
        (["--self_bleu_tokenizer", "whitespace"], "invalid choice"),
    ],
)
def test_inconsistent_options_are_rejected(argv: list[str], message: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        _parse(argv)
    assert message in capsys.readouterr().err


def test_the_nltk_self_bleu_tokenizer_is_checked_before_any_generation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    nltk = pytest.importorskip("nltk")

    def without_punkt(_text: str) -> list[str]:
        raise LookupError("Resource 'punkt_tab' not found.")

    monkeypatch.setattr(nltk, "word_tokenize", without_punkt)
    diversity = ["--num_samples", "2", "--temperature", "1", "--diversity", "--self_bleu_tokenizer", "nltk"]
    with pytest.raises(SystemExit):
        _parse(diversity)
    assert "python -m nltk.downloader punkt_tab punkt" in capsys.readouterr().err
    assert _parse(diversity[:-2]).self_bleu_tokenizer == "regex"
    assert _parse(["--self_bleu_tokenizer", "nltk"]).self_bleu_tokenizer == "nltk"


def test_records_reach_the_shard_as_they_are_produced_and_survive_a_failure(
    tiny_model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from math_reasoning.evaluate import main

    answer = _fixed_responses(WRONG)
    calls = []

    def fails_on_the_third_call(*arguments):
        calls.append(len(shard_file(output, 0).read_text(encoding="utf-8").splitlines()))
        if len(calls) == 3:
            raise RuntimeError("CUDA out of memory (simulated)")
        return answer(*arguments)

    monkeypatch.setattr(arix.evaluation, "generate_batch", fails_on_the_third_call)
    output = tmp_path / "eval" / "gsm8k.jsonl"
    argv = ["--model", str(tiny_model_dir), "--dataset", "gsm8k", "--engine", "nl", "--eval_file", str(NL_FILE)]
    argv += ["--torch_dtype", "float32", "--batch_size", "4", "--output_file", str(output)]
    with pytest.raises(RuntimeError, match="simulated"):
        main(argv)
    assert calls == [0, 4, 8]
    shard = shard_file(output, 0)
    assert shard == tmp_path / "eval" / "gsm8k.rank0.jsonl"
    kept = [json.loads(line) for line in shard.read_text(encoding="utf-8").splitlines()]
    assert [record["index"] for record in kept] == list(range(8))
    assert all(record["responses"] == [WRONG] and record["correct"] == [False] for record in kept)
    assert not output.exists() and not metrics_file(output).exists()

    monkeypatch.setattr(arix.evaluation, "generate_batch", answer)
    summary = main(argv)
    assert summary is not None and (summary["total"], summary["correct"], summary["batch_size"]) == (16, 0, 4)
    assert [json.loads(line)["index"] for line in output.read_text(encoding="utf-8").splitlines()] == list(range(16))
    assert read_json(metrics_file(output)) == summary
    assert sorted(path.name for path in output.parent.iterdir()) == ["gsm8k.jsonl", "gsm8k.metrics.json"]


WORKER = """
import json, os, sys
sys.path[:0] = [{code_root!r}, {tests_dir!r}]
import arix.evaluation
from math_reasoning.data import load_problems
from math_reasoning.evaluate import main

problems = load_problems({nl_file!r}, "gsm8k")

def scripted(_model, tokenizer, prompts, *_settings):
    responses = []
    for prompt in prompts:
        text = tokenizer.decode(prompt)
        index = next(i for i, problem in enumerate(problems) if problem.question.strip() in text)
        text = f"Therefore, the answer is: {{problems[index].answer}}" if index % 2 == 0 else {wrong!r}
        responses.append(tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id])
    return responses

arix.evaluation.generate_batch = scripted
import torch

summary = main(sys.argv[1:])
with open(os.path.join({out_dir!r}, "rank" + os.environ["RANK"] + ".json"), "w") as handle:
    json.dump({{"summary": summary, "seed": torch.initial_seed()}}, handle)
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="launches two local processes with gloo")
def test_two_processes_merge_their_records_on_the_main_process(tiny_model_dir: Path, tmp_path: Path) -> None:
    worker = tmp_path / "worker.py"
    code_root = str(TESTS_DIR.parent)
    worker.write_text(
        WORKER.format(
            code_root=code_root, tests_dir=str(TESTS_DIR), nl_file=str(NL_FILE), wrong=WRONG, out_dir=str(tmp_path)
        ),
        encoding="utf-8",
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    output = tmp_path / "eval" / "gsm8k.jsonl"
    argv = ["--model", str(tiny_model_dir), "--dataset", "gsm8k", "--engine", "nl", "--eval_file", str(NL_FILE)]
    argv += ["--torch_dtype", "float32", "--batch_size", "3", "--output_file", str(output)]
    launch = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", "--master_addr", "127.0.0.1"]
    launch += ["--master_port", str(port), str(worker), *argv]
    environment = {**os.environ, "ACCELERATE_USE_CPU": "true", "OMP_NUM_THREADS": "1"}
    result = subprocess.run(
        launch, cwd=code_root, env=environment, capture_output=True, text=True, timeout=300, check=False
    )
    assert result.returncode == 0, result.stderr[-3000:]

    main_process, second = (json.loads((tmp_path / f"rank{r}.json").read_text(encoding="utf-8")) for r in (0, 1))
    summary = main_process["summary"]
    assert second["summary"] is None
    assert (main_process["seed"], second["seed"]) == (1111, 1112)
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [record["index"] for record in records] == list(range(16))
    assert [record["num_correct"] for record in records] == [1, 0] * 8
    assert (summary["correct"], summary["total"], summary["accuracy"]) == (8, 16, 50.0)
    assert read_json(metrics_file(output)) == summary
    assert sorted(path.name for path in output.parent.iterdir()) == ["gsm8k.jsonl", "gsm8k.metrics.json"]


def test_write_json_replaces_a_file_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "metrics.json"
    write_json({"positive_labels": [2], "f1": 21.94}, path)
    with pytest.raises(TypeError):
        write_json({"positive_labels": [1, 2], "f1": 46.38, "bad": object()}, path)
    assert read_json(path) == {"positive_labels": [2], "f1": 21.94}
    assert [p.name for p in tmp_path.iterdir()] == ["metrics.json"]

    replaced = []
    real_replace = os.replace

    def recording_replace(source, target):
        replaced.append((Path(source), Path(target), json.loads(Path(source).read_text(encoding="utf-8"))))
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", recording_replace)
    write_json({"f1": 46.38}, path)
    [(source, target, content)] = replaced
    assert target == path and source.parent == path.parent and source.name.startswith(".metrics.json.")
    assert content == {"f1": 46.38} == read_json(path)
    write_json([1], tmp_path / "new" / "dir" / "list.json")
    assert read_json(tmp_path / "new" / "dir" / "list.json") == [1]
