from __future__ import annotations

import json
import logging
import os
import random
import secrets
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from accelerate import Accelerator


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(level: int = logging.INFO, main_process: bool = True) -> None:
    logging.basicConfig(
        level=level if main_process else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(obj: Any, path: str | os.PathLike[str], indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        with open(temporary, "x", encoding="utf-8") as handle:
            json.dump(obj, handle, ensure_ascii=False, indent=indent)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_jsonl(path: str | os.PathLike[str]) -> list[Any]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(rows: Iterable[Any], path: str | os.PathLike[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(row: Any, path: str | os.PathLike[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def all_reduce_sum(accelerator: Accelerator, values: Mapping[str, float]) -> dict[str, float]:
    keys = list(values)
    if accelerator.num_processes == 1:
        return {key: float(values[key]) for key in keys}
    packed = torch.tensor([float(values[key]) for key in keys], dtype=torch.float64, device=accelerator.device)
    summed = accelerator.reduce(packed, reduction="sum")
    return dict(zip(keys, summed.tolist(), strict=True))
