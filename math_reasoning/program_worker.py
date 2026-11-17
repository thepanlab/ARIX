from __future__ import annotations

import json
import math
import os
import signal
import sys
import threading
import time
from types import ModuleType
from typing import Any

MEMORY_LIMIT_BYTES = 4 * 1024**3
_PARENT_POLL_SECONDS = 1.0
_MAX_ERROR_CHARS = 300
_PR_SET_PDEATHSIG = 1


def restricted_modules() -> dict[str, ModuleType]:
    modules: dict[str, ModuleType] = {"math": math}
    try:
        import sympy
    except ImportError:
        return modules
    modules["sympy"] = sympy
    return modules


def execute(code: str, modules: dict[str, ModuleType]) -> dict[str, Any]:
    global_vars: dict[str, Any] = dict(modules)
    local_vars: dict[str, Any] = {}
    try:
        exec(code, global_vars, local_vars)
        value = float(eval("solution()", global_vars, local_vars))
    except BaseException as exc:
        return {"value": None, "error": f"{type(exc).__name__}: {exc}"[:_MAX_ERROR_CHARS]}
    if not math.isfinite(value):
        return {"value": None, "error": f"non-finite result {value}"}
    return {"value": value, "error": None}


def _limit_memory() -> None:
    try:
        import resource
    except ImportError:
        return
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = MEMORY_LIMIT_BYTES if hard == resource.RLIM_INFINITY else min(MEMORY_LIMIT_BYTES, hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))


def _set_parent_death_signal() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        arguments = (ctypes.c_ulong(signal.SIGKILL), ctypes.c_ulong(0), ctypes.c_ulong(0), ctypes.c_ulong(0))
        return bool(libc.prctl(_PR_SET_PDEATHSIG, *arguments) == 0)
    except (AttributeError, OSError):
        return False


def _watch_parent(parent: int) -> None:
    while os.getppid() == parent:
        time.sleep(_PARENT_POLL_SECONDS)
    os._exit(0)


def _exit_with_parent(parent: int) -> None:
    if not _set_parent_death_signal():
        threading.Thread(target=_watch_parent, args=(parent,), daemon=True).start()
    if os.getppid() != parent:
        os._exit(0)


def main() -> None:
    _exit_with_parent(int(sys.argv[1]) if len(sys.argv) > 1 else os.getppid())
    protocol_in = os.fdopen(os.dup(0), "r", encoding="utf-8")
    protocol_out = os.fdopen(os.dup(1), "w", encoding="utf-8")
    modules = restricted_modules()
    _limit_memory()
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(devnull, fd)
    protocol_out.write(json.dumps({"sympy": "sympy" in modules}) + "\n")
    protocol_out.flush()
    for line in protocol_in:
        protocol_out.write(json.dumps(execute(json.loads(line)["code"], modules)) + "\n")
        protocol_out.flush()


if __name__ == "__main__":
    main()
