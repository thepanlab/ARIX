from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
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
    read_run_config,
    sample_rollouts,
    sampling_suffix,
    sampling_summary,
    start_evaluation,
    write_outputs,
)
from arix.task import Example
from math_reasoning.data import load_problems
from math_reasoning.task import MathDataConfig, MathTask, make_example

logger = logging.getLogger(__name__)

DEFAULT_MAX_NEW_TOKENS = {"svamp": 512, "gsm8k": 512, "math": 1024, "aime": 1024}
DEFAULT_BATCH_SIZE = 64


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_evaluation_arguments(parser, "problem", DEFAULT_BATCH_SIZE)
    parser.add_argument("--dataset", choices=("svamp", "gsm8k", "math", "aime"))
    parser.add_argument("--engine", choices=("nl", "python"))
    parser.add_argument("--eval_file", help="evaluation problems (default: the config's data.test_file)")
    args = parser.parse_args(argv)
    check_evaluation_arguments(parser, args)
    return args


def eval_data_config(args: argparse.Namespace, cfg: ARIXConfig | None) -> MathDataConfig:
    data = dict(cfg.data) if cfg is not None else {}
    if args.dataset is not None and args.dataset != data.get("dataset"):
        data = {"dataset": args.dataset}
    if args.engine is not None:
        data["engine"] = args.engine
    if args.eval_file is not None:
        data["test_file"] = args.eval_file
    if "dataset" not in data:
        raise ValueError("give --dataset (or a config whose data section names the dataset)")
    return MathDataConfig.from_mapping(data)


def evaluate_examples(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    task: MathTask,
    examples: Sequence[Example],
    indices: Sequence[int],
    gen: GenerationConfig,
    settings: EvaluationSettings,
    projector: RationaleProjector | None,
    pooling: str,
    shard: RecordShard | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    n = settings.num_samples
    for chunk in sample_rollouts(model, tokenizer, task, examples, indices, gen, settings, projector, pooling):
        flat_examples = [group.example for group in chunk for _ in group.texts]
        answers, correct = task.score_batch(flat_examples, [text for group in chunk for text in group.texts])
        new = [
            {
                "index": group.index,
                "uid": group.example.uid,
                "question": group.example.meta["question"],
                "gold": group.example.target,
                "responses": group.texts,
                "predictions": answers[k * n : (k + 1) * n],
                "correct": correct[k * n : (k + 1) * n],
                "num_correct": sum(correct[k * n : (k + 1) * n]),
                **group.diversity,
            }
            for k, group in enumerate(chunk)
        ]
        if shard is not None:
            shard.write(new)
        records += new
    return records


def output_name(
    dataset: str, engine: str, eval_file: str | Path, num_samples: int, temperature: float = 0.0, top_p: float = 1.0
) -> str:
    return f"{dataset}_{engine}_{Path(eval_file).stem}" + sampling_suffix(num_samples, temperature, top_p)


def summarize(records: Sequence[dict[str, Any]], diversity: bool) -> dict[str, Any]:
    correct = sum(record["num_correct"] for record in records)
    total = sum(len(record["correct"]) for record in records)
    summary: dict[str, Any] = {
        "num_problems": len(records),
        "correct": correct,
        "total": total,
        "accuracy": 100.0 * correct / total if total else 0.0,
    }
    if diversity:
        summary.update(diversity_summary(records))
    return summary


def main(argv: Sequence[str] | None = None) -> dict[str, Any] | None:
    args = parse_args(argv)
    accelerator = start_evaluation(args.seed)
    cfg = read_run_config(args.model, args.config)
    data = eval_data_config(args, cfg)
    if data.test_file is None:
        raise ValueError("give --eval_file (or a config whose data section has test_file)")
    eval_file = data.test_file
    settings = evaluation_settings(args, cfg, DEFAULT_BATCH_SIZE)
    gen = evaluation_generation(args, cfg, DEFAULT_MAX_NEW_TOKENS[data.dataset])
    problems = load_problems(eval_file, data.dataset)
    examples = [make_example(problem, data.dataset, data.engine, with_reference=False) for problem in problems]
    name = output_name(data.dataset, data.engine, eval_file, args.num_samples, args.temperature, args.top_p)
    output_file = Path(args.output_file) if args.output_file else default_output_file(args.model, name)

    model, tokenizer = load_inference_model(
        args.model, cfg.model if cfg is not None else None, args.torch_dtype, accelerator.device
    )
    method = cfg.arix if cfg is not None else ARIXConfig().arix
    projector = load_projector(args.model, args.projector, method, model.config.hidden_size) if args.diversity else None
    indices = list(range(accelerator.process_index, len(examples), accelerator.num_processes))
    task = MathTask(data.engine, data.dataset, timeout=data.exec_timeout, num_workers=data.exec_workers)
    try:
        with RecordShard(output_file, accelerator.process_index) as shard:
            local = [examples[i] for i in indices]
            records = evaluate_examples(
                model, tokenizer, task, local, indices, gen, settings, projector, method.pooling, shard
            )
    finally:
        task.close()
    merged = gather_records(accelerator, records)
    if merged is None:
        return None

    summary = {
        "model": args.model,
        "dataset": data.dataset,
        "engine": data.engine,
        "eval_file": eval_file,
        **sampling_summary(settings, gen, args.seed),
        **summarize(merged, settings.diversity),
    }
    write_outputs(merged, summary, output_file, accelerator.num_processes)
    print(
        f"{data.dataset} ({eval_file}): {summary['correct']}/{summary['total']} correct, "
        f"accuracy {summary['accuracy']:.2f}%"
    )
    if "self_bleu" in summary:
        print(
            f"Self-BLEU-{settings.self_bleu_n} ({settings.self_bleu_tokenizer} tokens) {summary['self_bleu']:.4f}, "
            f"mean pairwise cosine distance {summary['mean_pairwise_cosine_distance']:.4f}"
        )
    logger.info("wrote %s", output_file)
    return summary


if __name__ == "__main__":
    main()
