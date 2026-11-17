from __future__ import annotations

import argparse
import json
from collections.abc import Collection, Iterable, Sequence
from pathlib import Path
from typing import Any

from clinical_trials.data import LABELS, Qrel, load_qrels
from clinical_trials.verifier import binary_metrics


def _percent(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def _baseline(gold: Sequence[bool], predicted: Sequence[bool]) -> dict[str, Any]:
    counts = {
        "tp": sum(g and p for g, p in zip(gold, predicted, strict=True)),
        "fp": sum(not g and p for g, p in zip(gold, predicted, strict=True)),
        "fn": sum(g and not p for g, p in zip(gold, predicted, strict=True)),
        "tn": sum(not g and not p for g, p in zip(gold, predicted, strict=True)),
    }
    return {**counts, **binary_metrics(**counts)}


def trial_overlap(train: Iterable[Qrel], test: Sequence[Qrel], positive_labels: Collection[int]) -> dict[str, Any]:
    positive = set(positive_labels)
    if not positive or not positive <= set(LABELS):
        raise ValueError(f"positive_labels must be a non-empty subset of {LABELS}, got {sorted(positive)}")
    train = list(train)
    train_trials = {q.trial_id for q in train}
    train_patients = {q.patient_id for q in train}
    positive_train_trials = {q.trial_id for q in train if q.label in positive}
    test_trials = {q.trial_id for q in test}
    overlapping = [q for q in test if q.trial_id in train_trials]
    overlapping_trials = {q.trial_id for q in overlapping}
    test_positive = sum(q.label in positive for q in test)
    gold = [q.label in positive for q in overlapping]
    return {
        "positive_labels": sorted(positive),
        "test_patients": len({q.patient_id for q in test}),
        "test_patients_in_train": len({q.patient_id for q in test} & train_patients),
        "test_trials": len(test_trials),
        "overlapping_trials": len(overlapping_trials),
        "overlapping_trials_percent": _percent(len(overlapping_trials), len(test_trials)),
        "test_pairs": len(test),
        "overlapping_pairs": len(overlapping),
        "overlapping_pairs_percent": _percent(len(overlapping), len(test)),
        "test_positive_pairs": test_positive,
        "overlapping_positive_pairs": sum(gold),
        "overlapping_positive_pairs_percent": _percent(sum(gold), test_positive),
        "overlapping_positive_rate_percent": _percent(sum(gold), len(overlapping)),
        "trial_identity_baseline": _baseline(gold, [q.trial_id in positive_train_trials for q in overlapping]),
        "all_positive_baseline": _baseline(gold, [True] * len(overlapping)),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train_qrels", nargs="+", required=True, help="qrels files of the training corpora")
    parser.add_argument("--test_qrels", nargs="+", required=True, help="qrels files of the test corpora")
    parser.add_argument("--positive_labels", type=int, nargs="+", choices=LABELS, default=[2])
    parser.add_argument("--output", help="also write the report to this JSON file")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    train = [qrel for path in args.train_qrels for qrel in load_qrels(path)]
    test = [qrel for path in args.test_qrels for qrel in load_qrels(path)]
    report = trial_overlap(train, test, args.positive_labels)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
