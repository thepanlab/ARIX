from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

from arix.task import Example
from math_reasoning.data import DATASETS, REFT_DATASETS, MathProblem, load_problems
from math_reasoning.prompts import (
    ENGINES,
    build_prompt,
    math_reference,
    nl_reference,
    program_header,
    python_reference,
)
from math_reasoning.verifier import (
    DEFAULT_NUM_WORKERS,
    DEFAULT_TIMEOUT,
    ProgramExecutor,
    extract_math_answer,
    extract_nl_answer,
    math_equal,
    numbers_match,
    parse_number,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

SPLITS = ("train", "test")


def check_dataset_engine(dataset: str, engine: str) -> None:
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}; expected one of {ENGINES}")
    if engine == "python" and dataset not in REFT_DATASETS:
        raise ValueError(f"the python engine needs program-format data ({REFT_DATASETS}), not {dataset!r}")


@dataclass(frozen=True)
class MathDataConfig:
    dataset: str
    engine: str = "nl"
    train_file: str | None = None
    test_file: str | None = None
    exec_timeout: float = DEFAULT_TIMEOUT
    exec_workers: int = DEFAULT_NUM_WORKERS

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> MathDataConfig:
        unknown = set(data) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown keys in the math data config: {sorted(unknown)}")
        if "dataset" not in data:
            raise ValueError("the math data config needs 'dataset'")
        config = cls(
            dataset=str(data["dataset"]),
            engine=str(data.get("engine", "nl")),
            train_file=None if data.get("train_file") is None else str(data["train_file"]),
            test_file=None if data.get("test_file") is None else str(data["test_file"]),
            exec_timeout=float(data.get("exec_timeout", DEFAULT_TIMEOUT)),
            exec_workers=int(data.get("exec_workers", DEFAULT_NUM_WORKERS)),
        )
        check_dataset_engine(config.dataset, config.engine)
        return config


class MathTask:
    def __init__(
        self,
        engine: str,
        dataset_name: str,
        timeout: float = DEFAULT_TIMEOUT,
        num_workers: int = DEFAULT_NUM_WORKERS,
    ) -> None:
        check_dataset_engine(dataset_name, engine)
        self.engine = engine
        self.dataset_name = dataset_name
        self.name = f"{dataset_name}_{engine}"
        self.executor = ProgramExecutor(num_workers, timeout) if engine == "python" else None

    def score_batch(self, examples: Sequence[Example], responses: Sequence[str]) -> tuple[list[Any], list[bool]]:
        if len(examples) != len(responses):
            raise ValueError(f"{len(examples)} examples but {len(responses)} responses")
        if self.dataset_name not in REFT_DATASETS:
            answers = [extract_math_answer(response) for response in responses]
            return answers, [math_equal(answer, ex.target) for answer, ex in zip(answers, examples, strict=True)]
        if self.executor is None:
            values = [extract_nl_answer(response) for response in responses]
        else:
            programs = [
                (program_header(ex.meta["question"]) + response).strip()
                for ex, response in zip(examples, responses, strict=True)
            ]
            values = [result.value for result in self.executor.run(programs)]
        return values, [numbers_match(value, ex.target) for value, ex in zip(values, examples, strict=True)]

    def verify_batch(self, examples: list[Example], responses: list[str]) -> list[bool]:
        return self.score_batch(examples, responses)[1]

    def extract_batch(self, examples: list[Example], responses: list[str]) -> list[Any]:
        return self.score_batch(examples, responses)[0]

    def close(self) -> None:
        if self.executor is not None:
            self.executor.close()


def task_from_config(data: Mapping[str, Any]) -> MathTask:
    config = MathDataConfig.from_mapping(data)
    return MathTask(config.engine, config.dataset, timeout=config.exec_timeout, num_workers=config.exec_workers)


def make_example(problem: MathProblem, dataset: str, engine: str, with_reference: bool) -> Example:
    check_dataset_engine(dataset, engine)
    if dataset in REFT_DATASETS and parse_number(problem.answer) is None:
        raise ValueError(f"{problem.uid}: the gold answer {problem.answer!r} is not a number")
    reference = ""
    if with_reference:
        if problem.solution is None:
            raise ValueError(f"{problem.uid}: a training problem needs a reference solution")
        if dataset not in REFT_DATASETS:
            reference = math_reference(problem.solution)
        elif engine == "python":
            reference = python_reference(problem.solution, problem.question)
        else:
            reference = nl_reference(problem.solution, problem.answer)
    return Example(
        uid=problem.uid,
        prompt=build_prompt(problem.question, engine),
        reference=reference,
        target=problem.answer,
        meta={"add_special_tokens": True, "question": problem.question},
    )


def build_examples(
    data: Mapping[str, Any], tokenizer: PreTrainedTokenizerBase | None = None, split: str = "train"
) -> list[Example]:
    del tokenizer
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    config = MathDataConfig.from_mapping(data)
    path = config.train_file if split == "train" else config.test_file
    if path is None:
        raise ValueError(f"data.{split}_file is not set")
    problems = load_problems(path, config.dataset)
    return [
        make_example(problem, config.dataset, config.engine, with_reference=split == "train") for problem in problems
    ]
