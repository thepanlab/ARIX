from __future__ import annotations

import re
from collections.abc import Collection, Sequence

from clinical_trials.data import LABELS

FINAL_LABEL_PATTERN = re.compile(r"final\s+label\s*\**\s*:\s*\**\s*([0-2])(?!\d)", re.IGNORECASE)

PLAIN, THINK, HARMONY = "plain", "think", "harmony"
RESPONSE_FORMATS = (PLAIN, THINK, HARMONY)
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
HARMONY_FINAL_HEADER = "<|channel|>final<|message|>"
HARMONY_MESSAGE_ENDS = ("<|end|>", "<|return|>", "<|call|>", "<|start|>")


def check_response_format(response_format: str) -> str:
    if response_format not in RESPONSE_FORMATS:
        raise ValueError(f"response_format must be one of {RESPONSE_FORMATS}, got {response_format!r}")
    return response_format


def final_answer(response: str, response_format: str = PLAIN) -> str | None:
    check_response_format(response_format)
    if response_format == PLAIN:
        return response
    if response_format == THINK:
        _, closed, answer = response.rpartition(THINK_CLOSE)
        return answer if closed else None
    _, opened, message = response.rpartition(HARMONY_FINAL_HEADER)
    if not opened:
        return None
    ends = [index for marker in HARMONY_MESSAGE_ENDS if (index := message.find(marker)) >= 0]
    return message[: min(ends)] if ends else message


def parse_final_label(response: str, response_format: str = PLAIN) -> int | None:
    answer = final_answer(response, response_format)
    if answer is None:
        return None
    matches = FINAL_LABEL_PATTERN.findall(answer)
    return int(matches[-1]) if matches else None


def is_correct(response: str, gold: int, response_format: str = PLAIN) -> bool:
    label = parse_final_label(response, response_format)
    return label is not None and label == gold


def confusion_matrix(gold: Sequence[int], predicted: Sequence[int]) -> list[list[int]]:
    if len(gold) != len(predicted):
        raise ValueError(f"{len(gold)} labels but {len(predicted)} predictions")
    matrix = [[0] * len(LABELS) for _ in LABELS]
    for g, p in zip(gold, predicted, strict=True):
        if g not in LABELS or p not in LABELS:
            raise ValueError(f"labels must be in {LABELS}, got {g} and {p}")
        matrix[g][p] += 1
    return matrix


def binary_counts(matrix: Sequence[Sequence[int]], positive_labels: Collection[int]) -> dict[str, int]:
    positive = set(positive_labels)
    if not positive or not positive <= set(LABELS):
        raise ValueError(f"positive_labels must be a non-empty subset of {LABELS}, got {sorted(positive)}")
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for g in LABELS:
        for p in LABELS:
            key = ("t" if (g in positive) == (p in positive) else "f") + ("p" if p in positive else "n")
            counts[key] += matrix[g][p]
    return counts


def binary_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:

    def percent(numerator: int, denominator: int) -> float:
        return 100.0 * numerator / denominator if denominator else 0.0

    return {
        "precision": percent(tp, tp + fp),
        "recall": percent(tp, tp + fn),
        "f1": percent(2 * tp, 2 * tp + fp + fn),
        "accuracy": percent(tp + tn, tp + fp + fn + tn),
    }
