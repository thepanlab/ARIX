from __future__ import annotations

import re

INSTRUCTION = "Question:\n"
COT_TRIGGER = "\nAnswer reasoning:\n"
ANSWER_TRIGGER = "\nTherefore, the answer is: "
ENGINES = ("nl", "python")

_PROGRAM_HEADER = re.compile(r'def solution\(\):[ \t]*\n(?:[ \t]*""".*?"""[ \t]*\n)?', re.DOTALL)


def program_header(question: str) -> str:
    return f'def solution():\n    """{question.strip()}"""\n'


def build_prompt(question: str, engine: str = "nl") -> str:
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}; expected one of {ENGINES}")
    prompt = f"{INSTRUCTION}{question.strip()}{COT_TRIGGER}"
    if engine == "python":
        prompt += program_header(question)
    return prompt


def nl_reference(answer_cot: str, answer_value: str) -> str:
    if answer_cot.lstrip().startswith("def solution"):
        raise ValueError("this reference is a python program; use engine 'python' for program-format data")
    return f"{answer_cot.strip()}{ANSWER_TRIGGER}{answer_value.strip()}"


def python_reference(answer_cot: str, question: str) -> str:
    program = answer_cot.strip()
    header = program_header(question)
    if program.startswith(header):
        return program[len(header) :]
    match = _PROGRAM_HEADER.match(program)
    if match is None:
        raise ValueError("a python-engine reference must be a 'def solution():' program")
    return program[match.end() :]


def math_reference(solution: str) -> str:
    return solution.strip()
