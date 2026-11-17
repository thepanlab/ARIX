from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from math_reasoning.verifier import last_boxed_content

DATASETS = ("svamp", "gsm8k", "math", "aime")
REFT_DATASETS = ("svamp", "gsm8k")


@dataclass(frozen=True)
class MathProblem:
    uid: str
    question: str
    answer: str
    solution: str | None = None


def load_problems(path: str | os.PathLike[str], dataset: str) -> list[MathProblem]:
    if dataset in REFT_DATASETS:
        return load_reft(path)
    if dataset == "math":
        return load_math(path)
    if dataset == "aime":
        return load_aime(path)
    raise ValueError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")


def load_reft(path: str | os.PathLike[str]) -> list[MathProblem]:
    records = _read_records(Path(path))
    problems = []
    for position, record in enumerate(records):
        missing = {"item_id", "question", "answer_value"} - set(record)
        if missing:
            raise ValueError(f"{path}: record {position} lacks {sorted(missing)}")
        problems.append(
            MathProblem(
                uid=str(record["item_id"]),
                question=record["question"].strip(),
                answer=str(record["answer_value"]).strip(),
                solution=record.get("answer_cot") or None,
            )
        )
    _check_unique(problems, path)
    return problems


def load_math(path: str | os.PathLike[str]) -> list[MathProblem]:
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*/*.json"))
        if not files:
            raise ValueError(f"{path}: no <subject>/<id>.json files")
        records = [_read_json(file) for file in files]
    else:
        records = _read_records(path)
    return _competition_problems(records, "math", path, answer_from_solution=True)


def load_aime(path: str | os.PathLike[str]) -> list[MathProblem]:
    return _competition_problems(_read_records(Path(path)), "aime", Path(path), answer_from_solution=False)


def problem_uid(dataset: str, text: str) -> str:
    return f"{dataset}-{hashlib.sha1(text.strip().encode('utf-8')).hexdigest()[:16]}"


def _competition_problems(
    records: Iterable[Mapping[str, Any]], dataset: str, path: Path, answer_from_solution: bool
) -> list[MathProblem]:
    problems = []
    seen: dict[str, int] = {}
    for position, record in enumerate(records):
        question = record.get("problem", record.get("question"))
        if question is None:
            raise ValueError(f"{path}: record {position} has neither 'problem' nor 'question'")
        solution = record.get("solution") or None
        boxed = last_boxed_content(solution) if answer_from_solution and solution is not None else None
        answer = boxed if boxed is not None else record.get("answer")
        if answer is None:
            raise ValueError(f"{path}: record {position} has no 'answer' and no \\boxed{{}} answer in its solution")
        if "item_id" in record:
            uid = str(record["item_id"])
        else:
            uid = problem_uid(dataset, question)
            repeats = seen.get(uid, 0)
            seen[uid] = repeats + 1
            uid = f"{uid}-{repeats}" if repeats else uid
        problems.append(
            MathProblem(
                uid=uid,
                question=question.strip(),
                answer=str(answer).strip(),
                solution=solution,
            )
        )
    _check_unique(problems, path)
    return problems


def _check_unique(problems: list[MathProblem], path: str | os.PathLike[str]) -> None:
    uids = [problem.uid for problem in problems]
    if len(set(uids)) != len(uids):
        raise ValueError(f"{path}: duplicate item ids")


def _read_records(path: Path) -> list[Mapping[str, Any]]:
    if path.suffix == ".json":
        records = _read_json(path)
        if not isinstance(records, list):
            raise ValueError(f"{path}: expected a JSON list of records")
        return records
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
