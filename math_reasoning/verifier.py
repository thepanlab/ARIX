from __future__ import annotations

import contextlib
import json
import math
import os
import re
import selectors
import signal
import subprocess
import sys
import time
import weakref
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import IO, cast

from math_reasoning.prompts import ANSWER_TRIGGER

NUMERIC_TOLERANCE = 1e-2
DEFAULT_TIMEOUT = 5.0
DEFAULT_NUM_WORKERS = 8
WORKER_SCRIPT = Path(__file__).with_name("program_worker.py")
WORKER_STARTUP_TIMEOUT = 60.0

_NUMBER = re.compile(r"(?<![\w./])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?|(?<![\w./])-?\.\d+")


def parse_number(text: str) -> float | None:
    try:
        value = float(text.replace(",", "").strip())
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def extract_nl_answer(response: str) -> float | None:
    if ANSWER_TRIGGER in response:
        answer = response.rsplit(ANSWER_TRIGGER, 1)[1]
        value = parse_number(answer)
        if value is not None:
            return value
        first = _NUMBER.search(answer)
        return parse_number(first.group()) if first else None
    numbers = _NUMBER.findall(response)
    return parse_number(numbers[-1]) if numbers else None


def numbers_match(prediction: float | None, gold: str) -> bool:
    target = parse_number(gold)
    return prediction is not None and target is not None and abs(prediction - target) <= NUMERIC_TOLERANCE


def process_code(code_gen: str, import_sympy: bool) -> str:
    code_gen = code_gen.replace("sys.exit", "print")
    lines = ["import math", "import sympy"] if import_sympy else ["import math"]
    for line in code_gen.split("\n"):
        if line.startswith("def solution"):
            lines.append(line)
            continue
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines)


@dataclass(frozen=True)
class ExecutionResult:
    value: float | None
    error: str | None = None


class _Worker:
    def __init__(self) -> None:
        self.process = subprocess.Popen(
            [sys.executable, "-I", str(WORKER_SCRIPT), str(os.getpid())],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            start_new_session=True,
        )
        self.stdin = cast(IO[str], self.process.stdin)
        self.stdout = cast(IO[str], self.process.stdout)
        self.sympy = False

    def wait_ready(self, timeout: float) -> None:
        with selectors.DefaultSelector() as selector:
            selector.register(self.stdout, selectors.EVENT_READ)
            line = self.stdout.readline() if selector.select(timeout) else ""
        if not line:
            self.kill()
            raise RuntimeError(f"the python-engine worker {WORKER_SCRIPT} did not start")
        self.sympy = bool(json.loads(line)["sympy"])

    def send(self, code: str) -> None:
        self.stdin.write(json.dumps({"code": code}) + "\n")
        self.stdin.flush()

    def receive(self) -> ExecutionResult | None:
        line = self.stdout.readline()
        if not line:
            return None
        payload = json.loads(line)
        return ExecutionResult(payload["value"], payload["error"])

    def kill(self) -> None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(self.process.pid, signal.SIGKILL)
        self.process.kill()
        self.process.wait()
        for stream in (self.stdin, self.stdout):
            with contextlib.suppress(OSError):
                stream.close()


def _kill_all(workers: list[_Worker]) -> None:
    for worker in workers:
        worker.kill()
    workers.clear()


class ProgramExecutor:
    def __init__(self, num_workers: int = DEFAULT_NUM_WORKERS, timeout: float = DEFAULT_TIMEOUT) -> None:
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")
        self.num_workers = num_workers
        self.timeout = timeout
        self._workers: list[_Worker] = []
        self._finalizer = weakref.finalize(self, _kill_all, self._workers)

    @property
    def sympy_available(self) -> bool:
        self._start_workers(1)
        return self._workers[0].sympy

    def run(self, programs: Sequence[str]) -> list[ExecutionResult]:
        if not programs:
            return []
        self.close()
        try:
            self._start_workers(min(self.num_workers, len(programs)))
            return self._execute([process_code(program, self._workers[0].sympy) for program in programs])
        finally:
            self.close()

    def _execute(self, codes: list[str]) -> list[ExecutionResult]:
        results = [ExecutionResult(None, "not executed")] * len(codes)
        queue = deque(range(len(codes)))
        idle = list(self._workers)
        busy: dict[_Worker, tuple[int, float]] = {}
        with selectors.DefaultSelector() as selector:
            while queue or busy:
                while queue and idle:
                    index = queue.popleft()
                    worker = self._dispatch(idle.pop(), codes[index])
                    busy[worker] = (index, time.monotonic() + self.timeout)
                    selector.register(worker.stdout, selectors.EVENT_READ, worker)
                next_deadline = min(deadline for _, deadline in busy.values())
                for key, _ in selector.select(max(0.0, next_deadline - time.monotonic())):
                    worker = key.data
                    index, _ = busy.pop(worker)
                    selector.unregister(worker.stdout)
                    result = worker.receive()
                    if result is None:
                        result, worker = ExecutionResult(None, "the worker died"), self._replace(worker)
                    results[index] = result
                    idle.append(worker)
                now = time.monotonic()
                expired = [worker for worker, (_, deadline) in busy.items() if deadline <= now]
                answered = {key.data for key, _ in selector.select(0)} if expired else set()
                for worker in expired:
                    if worker in answered:
                        continue
                    index, _ = busy.pop(worker)
                    selector.unregister(worker.stdout)
                    results[index] = ExecutionResult(None, f"timed out after {self.timeout:g} s")
                    idle.append(self._replace(worker))
        return results

    def close(self) -> None:
        _kill_all(self._workers)

    def _start_workers(self, count: int) -> None:
        new = [_Worker() for _ in range(count - len(self._workers))]
        self._workers.extend(new)
        for worker in new:
            worker.wait_ready(WORKER_STARTUP_TIMEOUT)

    def _replace(self, worker: _Worker) -> _Worker:
        worker.kill()
        fresh = _Worker()
        self._workers[self._workers.index(worker)] = fresh
        fresh.wait_ready(WORKER_STARTUP_TIMEOUT)
        return fresh

    def _dispatch(self, worker: _Worker, code: str) -> _Worker:
        try:
            worker.send(code)
        except OSError:
            worker = self._replace(worker)
            worker.send(code)
        return worker


def last_boxed_content(text: str) -> str | None:
    command = "\\boxed"
    index = text.rfind(command)
    if index < 0:
        command = "\\fbox"
        index = text.rfind(command)
        if index < 0:
            return None
    after = text[index + len(command) :]
    body = after.lstrip()
    if not body.startswith("{"):
        if not after[:1].isspace():
            return None
        return body.split("$")[0].strip() or None
    start = len(text) - len(body)
    depth = 0
    for position in range(start, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : position]
    return None


def extract_math_answer(response: str) -> str | None:
    boxed = last_boxed_content(response)
    if boxed is not None:
        return boxed
    if ANSWER_TRIGGER in response:
        answer = response.rsplit(ANSWER_TRIGGER, 1)[1].strip().rstrip(".").strip().strip("$").strip()
        return answer or None
    return None


class _NotNormalizable(ValueError):
    pass


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    for substr in substrs[1:]:
        new_str += "\\frac"
        if not substr:
            raise _NotNormalizable(string)
        if substr[0] == "{":
            new_str += substr
            continue
        if len(substr) < 2:
            return string
        a, b, rest = substr[0], substr[1], substr[2:]
        new_str += "{" + a + "}{" + b + "}" + rest if b != "{" else "{" + a + "}" + b + rest
    return new_str


def _fix_a_slash_b(string: str) -> str:
    parts = string.split("/")
    if len(parts) != 2:
        return string
    try:
        a, b = int(parts[0]), int(parts[1])
    except ValueError:
        return string
    if string != f"{a}/{b}":
        return string
    return "\\frac{" + str(a) + "}{" + str(b) + "}"


def _remove_right_units(string: str) -> str:
    if "\\text{ " not in string:
        return string
    splits = string.split("\\text{ ")
    if len(splits) != 2:
        raise _NotNormalizable(string)
    return splits[0]


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if not split:
            raise _NotNormalizable(string)
        new_string += "\\sqrt{" + split[0] + "}" + split[1:] if split[0] != "{" else "\\sqrt" + split
    return new_string


def strip_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if not string:
        return string
    if string[0] == ".":
        string = "0" + string
    sides = string.split("=")
    if len(sides) == 2 and len(sides[0]) <= 2:
        string = sides[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    return _fix_a_slash_b(string)


def is_equiv(prediction: str | None, gold: str | None) -> bool:
    if prediction is None or gold is None:
        return False
    try:
        return strip_string(prediction) == strip_string(gold)
    except ValueError:
        return prediction == gold


_DECIMAL = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_FRACTION = re.compile(r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}")


def as_exact_number(string: str) -> Fraction | None:
    if _DECIMAL.fullmatch(string):
        return Fraction(string.replace(",", ""))
    match = _FRACTION.fullmatch(string)
    if match is None or int(match.group(3)) == 0:
        return None
    value = Fraction(int(match.group(2)), int(match.group(3)))
    return -value if match.group(1) else value


def math_equal(prediction: str | None, gold: str | None) -> bool:
    if prediction is None or gold is None:
        return False
    if is_equiv(prediction, gold):
        return True
    try:
        left, right = as_exact_number(strip_string(prediction)), as_exact_number(strip_string(gold))
    except ValueError:
        return False
    return left is not None and right is not None and left == right
