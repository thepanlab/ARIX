from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from datetime import timedelta
from typing import Any

from accelerate import Accelerator, InitProcessGroupKwargs

from arix.config import load_run_config, parse_run_args
from arix.modeling import load_policy_and_tokenizer, load_reference_model
from arix.trainer import ARIXTrainer, check_output_dir
from arix.utils import set_seed, setup_logging
from math_reasoning.task import MathTask, build_examples, task_from_config

logger = logging.getLogger(__name__)

PROCESS_GROUP_TIMEOUT = timedelta(hours=5)


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    return parse_run_args(argv, description=__doc__)


def check_references(task: MathTask, examples: Sequence[Any]) -> None:
    correct = task.verify_batch(list(examples), [example.reference for example in examples])
    failed = [example.uid for example, ok in zip(examples, correct, strict=True) if not ok]
    if failed:
        logger.warning("%d of %d pinned references fail the verifier, e.g. %s", len(failed), len(examples), failed[:5])
    else:
        logger.info("all %d pinned references pass the verifier", len(examples))


def main(argv: Sequence[str] | None = None) -> None:
    args, overrides = parse_args(argv)
    cfg = load_run_config(args.config, overrides)
    check_output_dir(cfg.output)
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=PROCESS_GROUP_TIMEOUT)])
    setup_logging(main_process=accelerator.is_main_process)
    set_seed(cfg.training.seed)
    task = task_from_config(cfg.data)
    try:
        examples = build_examples(cfg.data, split="train")
        logger.info("%s: %d training problems from %s", task.name, len(examples), cfg.data.get("train_file"))
        if accelerator.is_main_process:
            check_references(task, examples)
        model, tokenizer = load_policy_and_tokenizer(cfg.model)
        ref_model = load_reference_model(cfg.model) if cfg.training.kl_beta > 0 else None
        ARIXTrainer(cfg, task, examples, model, tokenizer, ref_model, accelerator).train()
    finally:
        task.close()


if __name__ == "__main__":
    main()
