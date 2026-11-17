from __future__ import annotations

import argparse
import logging
import os
import random
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from arix.config import ARIXConfig, GenerationConfig
from arix.embedding import RationaleProjector
from arix.evaluation import (
    EvaluationSettings,
    RecordShard,
    add_evaluation_arguments,
    check_evaluation_arguments,
    default_output_file,
    diversity_summary,
    evaluation_generation,
    evaluation_settings,
    gather_records,
    load_inference_model,
    load_projector,
    metrics_file,
    read_run_config,
    sample_rollouts,
    sampling_suffix,
    sampling_summary,
    start_evaluation,
    write_outputs,
)
from arix.rollout import stop_token_ids
from arix.task import Example
from arix.utils import read_json, read_jsonl, setup_logging, write_json
from clinical_trials.data import LABELS, ClinicalDataConfig, load_evaluation_pairs
from clinical_trials.task import ClinicalTask, build_examples
from clinical_trials.verifier import PLAIN, binary_counts, binary_metrics, confusion_matrix, parse_final_label

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 16


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_evaluation_arguments(parser, "pair", DEFAULT_BATCH_SIZE)
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--max_pairs", type=int, help="evaluate a seeded random subset of this many pairs")
    parser.add_argument(
        "--allow_missing_pairs",
        action="store_true",
        help="test split: evaluate the loadable pairs when some judgments have no trial record or patient note",
    )
    parser.add_argument("--positive_labels", type=int, nargs="+", choices=LABELS, help="default: data.positive_labels")
    parser.add_argument("--unparsable_label", type=int, choices=LABELS, help="default: data.unparsable_label")
    parser.add_argument(
        "--rescore",
        help="recompute the metrics of this per-pair JSONL instead of generating (written to --output_file, else "
        "<stem>.rescore_pos<labels>.metrics.json)",
    )
    args = parser.parse_args(argv)
    check_evaluation_arguments(parser, args)
    if args.max_pairs is not None and args.max_pairs < 1:
        parser.error("--max_pairs must be >= 1")
    return args


def load_eval_config(args: argparse.Namespace) -> ARIXConfig:
    cfg = read_run_config(args.model, args.config)
    if cfg is None:
        raise FileNotFoundError(f"{args.model} has no config.yaml: give --config (the data paths come from the config)")
    return cfg


def check_test_pairs(data_summary: Mapping[str, Any], allow_missing_pairs: bool) -> None:
    missing = [
        f"{corpus}: {stats['missing_trial']} judgments without trial record and "
        f"{stats['missing_patient']} without patient note"
        for corpus, stats in data_summary["corpora"].items()
        if stats["missing_trial"] or stats["missing_patient"]
    ]
    if not missing:
        return
    if not allow_missing_pairs:
        raise ValueError(
            f"the test split scores every qrels judgment, but {'; '.join(missing)}. Provide the trial collection "
            "(scripts/download_data.sh clinical), or pass --allow_missing_pairs to evaluate the loadable pairs only"
        )
    logger.warning("--allow_missing_pairs: %s; these judgments are not evaluated", "; ".join(missing))


def evaluate_examples(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    task: ClinicalTask,
    examples: Sequence[Example],
    indices: Sequence[int],
    gen: GenerationConfig,
    settings: EvaluationSettings,
    projector: RationaleProjector | None,
    pooling: str,
    unparsable_label: int,
    shard: RecordShard | None = None,
) -> list[dict[str, Any]]:
    stops = set(stop_token_ids(model, tokenizer))
    records: list[dict[str, Any]] = []
    rollouts = sample_rollouts(
        model, tokenizer, task, examples, indices, gen, settings, projector, pooling, longest_first=True
    )
    for chunk in rollouts:
        new = []
        for group in chunk:
            ex, same_pair = group.example, [group.example] * len(group.texts)
            parsed = task.extract_batch(same_pair, group.texts)
            new.append(
                {
                    "index": group.index,
                    "uid": ex.uid,
                    "corpus": ex.meta["corpus"],
                    "patient_id": ex.meta["patient_id"],
                    "trial_id": ex.meta["trial_id"],
                    "label": ex.target,
                    "response_format": task.response_format,
                    "responses": group.texts,
                    "parsed_labels": parsed,
                    "correct": task.verify_batch(same_pair, group.texts),
                    "scored_correct": [(unparsable_label if label is None else label) == ex.target for label in parsed],
                    "finished": [bool(ids) and ids[-1] in stops for ids in group.response_ids],
                    "num_tokens": [len(ids) for ids in group.response_ids],
                    **group.diversity,
                }
            )
        if shard is not None:
            shard.write(new)
        records += new
    return records


def summarize(
    records: Sequence[dict[str, Any]], positive_labels: Collection[int], unparsable_label: int
) -> dict[str, Any]:
    gold = [record["label"] for record in records for _ in record["parsed_labels"]]
    parsed = [label for record in records for label in record["parsed_labels"]]
    predicted = [unparsable_label if label is None else label for label in parsed]
    matrix = confusion_matrix(gold, predicted)
    counts = binary_counts(matrix, positive_labels)
    total = len(predicted)
    summary: dict[str, Any] = {
        "num_pairs": len(records),
        "num_evaluated": total,
        "num_unparsable": sum(label is None for label in parsed),
        "num_truncated": sum(not finished for record in records for finished in record["finished"]),
        "unparsable_label": unparsable_label,
        "positive_labels": sorted(set(positive_labels)),
        "confusion_matrix": matrix,
        **counts,
        **binary_metrics(**counts),
        "accuracy_3class": 100.0 * sum(matrix[k][k] for k in LABELS) / total if total else 0.0,
    }
    summary.update(diversity_summary(records))
    return summary


def print_summary(summary: Mapping[str, Any]) -> None:
    corpora = (summary.get("data") or {}).get("corpora") or {}
    if summary.get("split") == "test" and corpora:
        judgments = sum(int(stats.get("qrels", 0)) for stats in corpora.values())
        scope = f" of {judgments} qrels judgments ({', '.join(corpora)})"
    else:
        scope = f" ({summary['split']} split)" if summary.get("split") else ""
    pairs = "pair" if summary["num_pairs"] == 1 else "pairs"
    print(
        f"{summary['num_pairs']} {pairs} evaluated{scope}, {summary['num_evaluated']} rollouts "
        f"({summary['num_unparsable']} unparsable), positive labels {summary['positive_labels']}: "
        f"precision {summary['precision']:.2f}%, recall {summary['recall']:.2f}%, F1 {summary['f1']:.2f}%, "
        f"accuracy {summary['accuracy']:.2f}%"
    )
    if "self_bleu" in summary:
        print(
            f"Self-BLEU {summary['self_bleu']:.4f}, "
            f"mean pairwise cosine distance {summary['mean_pairwise_cosine_distance']:.4f}"
        )


def output_name(
    split: str,
    corpora: Sequence[str],
    max_pairs: int | None,
    num_samples: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    name = f"{split}_{'_'.join(corpora)}" + (f"_pairs{max_pairs}" if max_pairs is not None else "")
    return name + sampling_suffix(num_samples, temperature, top_p)


def rescore_metrics_file(predictions: str | Path, positive_labels: Collection[int]) -> Path:
    labels = "-".join(str(label) for label in sorted(set(positive_labels)))
    return Path(predictions).with_suffix(f".rescore_pos{labels}.metrics.json")


def rescore(args: argparse.Namespace) -> dict[str, Any] | None:
    if int(os.environ.get("RANK", "0")) != 0:
        return None
    setup_logging()
    cfg = load_eval_config(args)
    data = ClinicalDataConfig.from_dict(cfg.data)
    positive_labels = args.positive_labels or data.positive_labels
    unparsable_label = data.unparsable_label if args.unparsable_label is None else args.unparsable_label
    predictions = Path(args.rescore)
    evaluation_metrics = metrics_file(predictions)
    output = Path(args.output_file) if args.output_file else rescore_metrics_file(predictions, positive_labels)
    if output.resolve() in (predictions.resolve(), evaluation_metrics.resolve()):
        raise ValueError(f"--output_file {output} would overwrite the evaluation's predictions or metrics")

    records = read_jsonl(predictions)
    unrecorded = sum("response_format" not in record for record in records)
    if unrecorded:
        logger.warning(
            "%d of %d records have no response_format (written before it was recorded): parsed as %r, the whole "
            "response",
            unrecorded,
            len(records),
            PLAIN,
        )
    formats = sorted({record.get("response_format", PLAIN) for record in records})
    reparsed = [
        {
            **record,
            "parsed_labels": [
                parse_final_label(text, record.get("response_format", PLAIN)) for text in record["responses"]
            ],
        }
        for record in records
    ]
    changed = sum(
        new != old
        for record, again in zip(records, reparsed, strict=True)
        for new, old in zip(again["parsed_labels"], record["parsed_labels"], strict=True)
    )
    if changed:
        logger.warning("%d rollouts parse to another label than the one stored in %s", changed, predictions)
    summary = read_json(evaluation_metrics) if evaluation_metrics.is_file() else {}
    summary.update(summarize(reparsed, positive_labels, unparsable_label))
    summary["response_format"] = formats[0] if len(formats) == 1 else formats
    summary["rescored_from"] = str(predictions)
    write_json(summary, output)
    print_summary(summary)
    logger.info("wrote %s", output)
    return summary


def main(argv: Sequence[str] | None = None) -> dict[str, Any] | None:
    args = parse_args(argv)
    if args.rescore:
        return rescore(args)
    cfg = load_eval_config(args)
    data = ClinicalDataConfig.from_dict(cfg.data)
    positive_labels = args.positive_labels or data.positive_labels
    unparsable_label = data.unparsable_label if args.unparsable_label is None else args.unparsable_label

    accelerator = start_evaluation(args.seed)
    pairs, data_summary = load_evaluation_pairs(data, args.split)
    if args.split == "test":
        check_test_pairs(data_summary, args.allow_missing_pairs)
    max_pairs = args.max_pairs if args.max_pairs is not None and args.max_pairs < len(pairs) else None
    if max_pairs is not None:
        pairs = sorted(random.Random(data.split_seed).sample(pairs, max_pairs), key=lambda pair: pair.uid)
    settings = evaluation_settings(args, cfg, DEFAULT_BATCH_SIZE)
    gen = evaluation_generation(args, cfg)
    corpora = data.test_corpora if args.split == "test" else data.train_corpora
    name = output_name(args.split, corpora, max_pairs, args.num_samples, args.temperature, args.top_p)
    output_file = Path(args.output_file) if args.output_file else default_output_file(args.model, name)

    model, tokenizer = load_inference_model(args.model, cfg.model, args.torch_dtype, accelerator.device)
    task = ClinicalTask.for_tokenizer(tokenizer)
    logger.info("final answers parsed in the %r response format", task.response_format)
    projector = None
    if args.diversity:
        projector = load_projector(args.model, args.projector, cfg.arix, model.config.hidden_size)
    examples = build_examples(pairs, tokenizer, system_in_user=data.system_in_user)
    indices = list(range(accelerator.process_index, len(examples), accelerator.num_processes))
    with RecordShard(output_file, accelerator.process_index) as shard:
        local = [examples[i] for i in indices]
        records = evaluate_examples(
            model, tokenizer, task, local, indices, gen, settings, projector, cfg.arix.pooling, unparsable_label, shard
        )
    merged = gather_records(accelerator, records)
    if merged is None:
        return None

    summary = {
        "model": args.model,
        "split": args.split,
        "corpora": corpora,
        "max_pairs": max_pairs,
        "response_format": task.response_format,
        **sampling_summary(settings, gen, args.seed),
        **summarize(merged, positive_labels, unparsable_label),
        "data": data_summary,
    }
    write_outputs(merged, summary, output_file, accelerator.num_processes)
    print_summary(summary)
    logger.info("wrote %s", output_file)
    return summary


if __name__ == "__main__":
    main()
