from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

LABELS = (0, 1, 2)
SECTIONS = ("inclusion", "exclusion")
INCLUSION_LABELS = ("not applicable", "not enough information", "included", "not included")
EXCLUSION_LABELS = ("not applicable", "not enough information", "excluded", "not excluded")
CONSENT_SENTENCE = (
    "The patient will provide informed consent, and will comply with the trial protocol without any practical issues."
)


_SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*(?=\s)")
_ABBREVIATIONS = frozenset({"dr", "mr", "mrs", "ms", "prof", "st", "vs", "e.g", "i.e", "approx"})
_INITIAL = re.compile(r"[^\W\d_]\.")
_NEXT_CHARACTER = re.compile(r"\s*(\S)")


def regex_split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        word = text[start : match.end()].split()[-1].lstrip("\"'([").rstrip("\"')]")
        if word.endswith(".") and word[:-1].lower() in _ABBREVIATIONS:
            continue
        following = _NEXT_CHARACTER.match(text, match.end())
        if _INITIAL.fullmatch(word) and following is not None and following.group(1).islower():
            continue
        sentence = text[start : match.end()].strip()
        if sentence:
            sentences.append(sentence)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


@functools.cache
def _punkt_sentence_tokenizer() -> Callable[[str], list[str]] | None:
    try:
        from nltk.tokenize import sent_tokenize
    except ImportError:
        logger.warning(
            "nltk is not installed: patient notes are split into sentences with a regular expression, "
            "whose sentence numbers can differ from those cited by TrialGPT's explanations"
        )
        return None
    try:
        sent_tokenize("The punkt model is installed. It splits sentences.")
    except LookupError:
        logger.warning(
            "the nltk punkt model is not installed (python -m nltk.downloader punkt_tab punkt): patient "
            "notes are split into sentences with a regular expression, whose sentence numbers can differ "
            "from those cited by TrialGPT's explanations"
        )
        return None
    return sent_tokenize


def split_sentences(text: str) -> list[str]:
    tokenizer = _punkt_sentence_tokenizer()
    return list(tokenizer(text)) if tokenizer is not None else regex_split_sentences(text)


def sentence_splitter() -> str:
    return "punkt" if _punkt_sentence_tokenizer() is not None else "regex"


def number_patient_note(text: str) -> str:
    sentences = split_sentences(text)
    sentences.append(CONSENT_SENTENCE)
    return "\n".join(f"{idx}. {sentence}" for idx, sentence in enumerate(sentences))


def split_criteria(criteria: str) -> list[str]:
    kept = []
    for criterion in criteria.split("\n\n"):
        criterion = criterion.strip()
        if "inclusion criteria" in criterion.lower() or "exclusion criteria" in criterion.lower():
            continue
        if len(criterion) < 5:
            continue
        kept.append(criterion)
    return kept


def parse_criteria(criteria: str) -> str:
    return "".join(f"{idx}. {criterion}\n" for idx, criterion in enumerate(split_criteria(criteria)))


def section_criteria(trial: Mapping[str, Any], section: str) -> list[str]:
    return split_criteria(str(trial.get(f"{section}_criteria") or ""))


def _read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _iter_jsonl(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


@dataclass(frozen=True)
class Qrel:
    patient_id: str
    trial_id: str
    label: int


def load_qrels(path: str | os.PathLike[str]) -> list[Qrel]:
    qrels = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            columns = line.split()
            if not columns:
                continue
            if len(columns) != 3:
                raise ValueError(f"{path}:{line_no}: expected 'query-id corpus-id score', got {line.strip()!r}")
            patient_id, trial_id, score = columns
            if line_no == 1 and score == "score":
                continue
            try:
                label = int(score)
            except ValueError:
                raise ValueError(f"{path}:{line_no}: the score {score!r} is not an integer") from None
            if label not in LABELS:
                raise ValueError(f"{path}:{line_no}: label {label} is not one of {LABELS}")
            qrels.append(Qrel(patient_id, trial_id, label))
    return qrels


def load_queries(path: str | os.PathLike[str]) -> dict[str, str]:
    return {str(row["_id"]): row["text"] for row in _iter_jsonl(path)}


def load_corpus_trials(
    path: str | os.PathLike[str], trial_ids: Iterable[str] | None = None
) -> dict[str, dict[str, Any]]:
    wanted = None if trial_ids is None else set(trial_ids)
    trials = {}
    for row in _iter_jsonl(path):
        trial_id = str(row["_id"])
        if wanted is not None and trial_id not in wanted:
            continue
        info = dict(row.get("metadata") or {})
        info.setdefault("brief_title", row.get("title", ""))
        trials[trial_id] = info
    return trials


def load_retrieved_trials(
    path: str | os.PathLike[str], trial_ids: Iterable[str] | None = None
) -> dict[str, dict[str, Any]]:
    wanted = None if trial_ids is None else set(trial_ids)
    trials: dict[str, dict[str, Any]] = {}
    for instance in _read_json(path):
        for label in ("0", "1", "2"):
            for trial in instance.get(label, []):
                trial_id = str(trial["NCTID"])
                if wanted is None or trial_id in wanted:
                    trials.setdefault(trial_id, dict(trial))
    return trials


def load_trials(
    corpus_dir: str | os.PathLike[str], trial_ids: Iterable[str] | None = None
) -> dict[str, dict[str, Any]]:
    corpus_dir = Path(corpus_dir)
    corpus_file, retrieved_file = corpus_dir / "corpus.jsonl", corpus_dir / "retrieved_trials.json"
    if not corpus_file.exists() and not retrieved_file.exists():
        raise FileNotFoundError(f"{corpus_dir} has neither corpus.jsonl nor retrieved_trials.json")
    wanted = None if trial_ids is None else set(trial_ids)
    trials = load_corpus_trials(corpus_file, wanted) if corpus_file.exists() else {}
    if retrieved_file.exists():
        for trial_id, info in load_retrieved_trials(retrieved_file, wanted).items():
            trials.setdefault(trial_id, info)
    return trials


def load_matching_results(path: str | os.PathLike[str], corpus: str) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for patient_id, by_label in _read_json(path).items():
        for trials in by_label.values():
            for trial_id, output in trials.items():
                uid = pair_uid(corpus, patient_id, trial_id)
                if uid in outputs:
                    raise ValueError(f"{path}: patient {patient_id} and trial {trial_id} appear under several labels")
                outputs[uid] = output
    return outputs


@dataclass(frozen=True)
class CriterionJudgment:
    index: int
    reasoning: str
    sentence_ids: tuple[int, ...]
    label: str


@dataclass(frozen=True)
class ReferenceAssessment:
    inclusion: tuple[CriterionJudgment, ...]
    exclusion: tuple[CriterionJudgment, ...]

    @property
    def label(self) -> int:
        return derive_trial_label((j.label for j in self.inclusion), (j.label for j in self.exclusion))


def derive_trial_label(inclusion_labels: Iterable[str], exclusion_labels: Iterable[str]) -> int:
    if "not included" in set(inclusion_labels):
        return 0
    if "excluded" in set(exclusion_labels):
        return 1
    return 2


def _judgment(key: Any, value: Any, num_criteria: int) -> CriterionJudgment | None:
    if not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key or int(key) >= num_criteria:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    reasoning, evidence, label = value
    if not isinstance(reasoning, str) or not isinstance(label, str) or not isinstance(evidence, list):
        return None
    sentence_ids = tuple(
        int(item)
        for item in evidence
        if (isinstance(item, int) and not isinstance(item, bool))
        or (isinstance(item, str) and item.strip().isdecimal())
    )
    return CriterionJudgment(int(key), reasoning.strip(), sentence_ids, " ".join(label.lower().split()))


def parse_matching_output(output: Any, trial: Mapping[str, Any]) -> ReferenceAssessment | None:
    if not isinstance(output, Mapping):
        return None
    sections = {}
    for section in SECTIONS:
        raw = output.get(section)
        if not isinstance(raw, Mapping):
            return None
        num_criteria = len(section_criteria(trial, section))
        judgments = [j for key, value in raw.items() if (j := _judgment(key, value, num_criteria)) is not None]
        if not judgments and (raw or num_criteria > 0):
            return None
        sections[section] = tuple(sorted(judgments, key=lambda j: j.index))
    return ReferenceAssessment(inclusion=sections["inclusion"], exclusion=sections["exclusion"])


def pair_uid(corpus: str, patient_id: str, trial_id: str) -> str:
    return f"{corpus}/{patient_id}/{trial_id}"


@dataclass(frozen=True)
class ClinicalPair:
    corpus: str
    patient_id: str
    trial_id: str
    label: int
    patient_note: str
    trial: Mapping[str, Any]
    reference: ReferenceAssessment | None = None

    @property
    def uid(self) -> str:
        return pair_uid(self.corpus, self.patient_id, self.trial_id)


def label_counts(pairs: Iterable[ClinicalPair]) -> dict[str, int]:
    counts = Counter(pair.label for pair in pairs)
    return {str(label): counts[label] for label in LABELS}


def load_corpus_pairs(dataset_dir: str | os.PathLike[str], corpus: str) -> tuple[list[ClinicalPair], dict[str, Any]]:
    corpus_dir = Path(dataset_dir) / corpus
    qrels = load_qrels(corpus_dir / "qrels" / "test.tsv")
    patients = load_queries(corpus_dir / "queries.jsonl")
    trials = load_trials(corpus_dir, {q.trial_id for q in qrels})
    seen: dict[tuple[str, str], int] = {}
    notes: dict[str, str] = {}
    pairs = []
    repeated = missing_trial = missing_patient = 0
    for qrel in qrels:
        key = (qrel.patient_id, qrel.trial_id)
        if key in seen:
            if seen[key] != qrel.label:
                raise ValueError(f"{corpus}: patient {key[0]} and trial {key[1]} have conflicting labels")
            repeated += 1
            continue
        seen[key] = qrel.label
        if qrel.trial_id not in trials:
            missing_trial += 1
            continue
        if qrel.patient_id not in patients:
            missing_patient += 1
            continue
        if qrel.patient_id not in notes:
            notes[qrel.patient_id] = number_patient_note(patients[qrel.patient_id])
        pairs.append(
            ClinicalPair(
                corpus=corpus,
                patient_id=qrel.patient_id,
                trial_id=qrel.trial_id,
                label=qrel.label,
                patient_note=notes[qrel.patient_id],
                trial=trials[qrel.trial_id],
            )
        )
    if missing_trial or missing_patient:
        logger.warning(
            "%s: %d judgments without trial record and %d without patient note are skipped",
            corpus,
            missing_trial,
            missing_patient,
        )
    qrels_labels = Counter(qrel.label for qrel in qrels)
    stats = {
        "qrels": len(qrels),
        "qrels_labels": {str(label): qrels_labels[label] for label in LABELS},
        "repeated_qrels": repeated,
        "missing_trial": missing_trial,
        "missing_patient": missing_patient,
        "pairs": len(pairs),
        "patients": len({pair.patient_id for pair in pairs}),
        "trials": len({pair.trial_id for pair in pairs}),
        "labels": label_counts(pairs),
    }
    return pairs, stats


def split_rank(uid: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{uid}".encode()).digest()


def split_train_validation(
    pairs: Sequence[ClinicalPair], train_fraction: float, seed: int
) -> tuple[list[ClinicalPair], list[ClinicalPair]]:
    if not 0 < train_fraction <= 1:
        raise ValueError(f"train_fraction must be in (0, 1], got {train_fraction}")
    ranked = sorted(pairs, key=lambda pair: (split_rank(pair.uid, seed), pair.uid))
    num_train = round(train_fraction * len(ranked))

    def by_uid(part: list[ClinicalPair]) -> list[ClinicalPair]:
        return sorted(part, key=lambda pair: pair.uid)

    return by_uid(ranked[:num_train]), by_uid(ranked[num_train:])


REFERENCE_STAGES = ("pairs", "with_output", "kept")


def _stage_counts() -> dict[str, dict[str, int]]:
    return {stage: {str(label): 0 for label in LABELS} for stage in REFERENCE_STAGES}


@dataclass
class ReferenceStats:
    pairs: int = 0
    kept: int = 0
    no_output: int = 0
    unusable_output: int = 0
    label_disagreement: int = 0
    labels: dict[str, dict[str, int]] = field(default_factory=_stage_counts)

    @property
    def dropped(self) -> int:
        return self.no_output + self.unusable_output + self.label_disagreement

    def count(self, stage: str, label: int) -> None:
        self.labels[stage][str(label)] += 1

    def as_dict(self) -> dict[str, Any]:
        counts = {key: value for key, value in asdict(self).items() if key != "labels"}
        return {**counts, "dropped": self.dropped, "labels": {stage: dict(n) for stage, n in self.labels.items()}}


def attach_references(
    pairs: Iterable[ClinicalPair], matching: Mapping[str, Any]
) -> tuple[list[ClinicalPair], ReferenceStats]:
    stats = ReferenceStats()
    kept = []
    for pair in pairs:
        stats.pairs += 1
        stats.count("pairs", pair.label)
        output = matching.get(pair.uid)
        if output is None:
            stats.no_output += 1
            continue
        stats.count("with_output", pair.label)
        assessment = parse_matching_output(output, pair.trial)
        if assessment is None:
            stats.unusable_output += 1
            continue
        if assessment.label != pair.label:
            stats.label_disagreement += 1
            continue
        kept.append(replace(pair, reference=assessment))
        stats.kept += 1
        stats.count("kept", pair.label)
    return kept, stats


@dataclass
class ClinicalDataConfig:
    dataset_dir: str = ""
    train_corpora: list[str] = field(default_factory=lambda: ["sigir", "trec_2021"])
    test_corpora: list[str] = field(default_factory=lambda: ["trec_2022"])
    matching_results: dict[str, str] = field(default_factory=dict)
    train_fraction: float = 0.8
    split_seed: int = 1111
    positive_labels: list[int] = field(default_factory=lambda: [2])
    unparsable_label: int = 0
    system_in_user: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ClinicalDataConfig:
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown keys in config section 'data': {sorted(unknown)}")
        values = dict(data)
        for key in ("train_corpora", "test_corpora"):
            if isinstance(values.get(key), str):
                values[key] = [values[key]]
        if isinstance(values.get("positive_labels"), int):
            values["positive_labels"] = [values["positive_labels"]]
        cfg = cls(**values)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        corpora = [*self.train_corpora, *self.test_corpora]
        checks = [
            (bool(self.dataset_dir), "data.dataset_dir is required"),
            (all(isinstance(c, str) and c for c in corpora), "data.train_corpora and data.test_corpora must be names"),
            (len(set(corpora)) == len(corpora), "a corpus cannot be listed twice in data.train_corpora/test_corpora"),
            (
                isinstance(self.matching_results, Mapping)
                and all(isinstance(k, str) and isinstance(v, str) for k, v in self.matching_results.items()),
                "data.matching_results must map corpus names to file paths",
            ),
            (
                isinstance(self.train_fraction, (int, float)) and 0 < self.train_fraction <= 1,
                "data.train_fraction must be in (0, 1]",
            ),
            (isinstance(self.split_seed, int), "data.split_seed must be an integer"),
            (
                bool(self.positive_labels) and set(self.positive_labels) <= set(LABELS),
                f"data.positive_labels must be a non-empty subset of {list(LABELS)}",
            ),
            (self.unparsable_label in LABELS, f"data.unparsable_label must be one of {list(LABELS)}"),
            (isinstance(self.system_in_user, bool), "data.system_in_user must be true or false"),
        ]
        errors = [message for ok, message in checks if not ok]
        if errors:
            raise ValueError("invalid clinical data configuration:\n  - " + "\n  - ".join(errors))


def load_train_validation_pairs(
    cfg: ClinicalDataConfig,
) -> tuple[list[ClinicalPair], list[ClinicalPair], dict[str, Any]]:
    if not cfg.train_corpora:
        raise ValueError("data.train_corpora is empty")
    pairs: list[ClinicalPair] = []
    corpora = {}
    for corpus in cfg.train_corpora:
        corpus_pairs, corpora[corpus] = load_corpus_pairs(cfg.dataset_dir, corpus)
        pairs += corpus_pairs
    train, validation = split_train_validation(pairs, cfg.train_fraction, cfg.split_seed)
    summary = {
        "corpora": corpora,
        "sentence_splitter": sentence_splitter(),
        "train_fraction": cfg.train_fraction,
        "split_seed": cfg.split_seed,
        "train_pairs": len(train),
        "validation_pairs": len(validation),
    }
    return train, validation, summary


def load_training_pairs(cfg: ClinicalDataConfig) -> tuple[list[ClinicalPair], dict[str, Any]]:
    missing = [corpus for corpus in cfg.train_corpora if corpus not in cfg.matching_results]
    if missing:
        raise ValueError(f"data.matching_results has no TrialGPT-Matching output file for {missing}")
    train, _, summary = load_train_validation_pairs(cfg)
    matching: dict[str, Any] = {}
    for corpus in cfg.train_corpora:
        matching.update(load_matching_results(cfg.matching_results[corpus], corpus))
    kept, stats = attach_references(train, matching)
    summary["references"] = stats.as_dict()
    summary["train_labels"] = label_counts(kept)
    logger.info(
        "training pairs: %d of %d kept (dropped: %d without TrialGPT output, %d unusable outputs, "
        "%d whose derived label disagrees with the dataset label)",
        stats.kept,
        stats.pairs,
        stats.no_output,
        stats.unusable_output,
        stats.label_disagreement,
    )
    return kept, summary


def load_evaluation_pairs(cfg: ClinicalDataConfig, split: str) -> tuple[list[ClinicalPair], dict[str, Any]]:
    if split == "validation":
        _, validation, summary = load_train_validation_pairs(cfg)
        return validation, summary
    if split != "test":
        raise ValueError(f"split must be 'validation' or 'test', got {split!r}")
    if not cfg.test_corpora:
        raise ValueError("data.test_corpora is empty")
    pairs: list[ClinicalPair] = []
    corpora = {}
    for corpus in cfg.test_corpora:
        corpus_pairs, corpora[corpus] = load_corpus_pairs(cfg.dataset_dir, corpus)
        pairs += corpus_pairs
    pairs.sort(key=lambda pair: pair.uid)
    return pairs, {"corpora": corpora, "sentence_splitter": sentence_splitter(), "test_pairs": len(pairs)}
