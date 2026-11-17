from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

TESTS_DIR = Path(__file__).resolve().parent
CODE_ROOT = TESTS_DIR.parent
FIXTURES_DIR = TESTS_DIR / "fixtures"

for _path in (TESTS_DIR, CODE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def fixture_texts(directory: Path = FIXTURES_DIR) -> list[str]:
    texts: list[str] = []
    for path in sorted(directory.rglob("*")):
        if path.suffix == ".json":
            texts.extend(_strings(json.loads(path.read_text(encoding="utf-8"))))
        elif path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    texts.extend(_strings(json.loads(line)))
    return texts


@pytest.fixture(scope="session")
def code_root() -> Path:
    return CODE_ROOT


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def math_fixtures_dir() -> Path:
    return FIXTURES_DIR / "math"


@pytest.fixture(scope="session")
def clinical_fixtures_dir() -> Path:
    return FIXTURES_DIR / "clinical"


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from tiny_model import build_tiny_model

    return build_tiny_model(tmp_path_factory.mktemp("tiny_model"), fixture_texts())
