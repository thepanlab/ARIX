from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

from accelerate import Accelerator, InitProcessGroupKwargs

from arix.config import load_run_config, parse_run_args
from arix.modeling import load_policy_and_tokenizer, load_reference_model
from arix.trainer import ARIXTrainer, check_output_dir
from arix.utils import set_seed, setup_logging, write_json
from clinical_trials.data import ClinicalDataConfig, load_training_pairs
from clinical_trials.task import ClinicalTask, build_examples

logger = logging.getLogger(__name__)

PROCESS_GROUP_TIMEOUT = timedelta(hours=5)


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    return parse_run_args(argv, description=__doc__)


def main(argv: Sequence[str] | None = None) -> None:
    args, overrides = parse_args(argv)
    cfg = load_run_config(args.config, overrides)
    check_output_dir(cfg.output)
    data_cfg = ClinicalDataConfig.from_dict(cfg.data)
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=PROCESS_GROUP_TIMEOUT)])
    setup_logging(main_process=accelerator.is_main_process)
    set_seed(cfg.training.seed)
    pairs, summary = load_training_pairs(data_cfg)
    if accelerator.is_main_process:
        write_json(summary, Path(cfg.output.output_dir) / "data_summary.json")
    model, tokenizer = load_policy_and_tokenizer(cfg.model)
    ref_model = load_reference_model(cfg.model) if cfg.training.kl_beta > 0 else None
    examples = build_examples(pairs, tokenizer, system_in_user=data_cfg.system_in_user)
    task = ClinicalTask.for_tokenizer(tokenizer)
    logger.info(
        "%d training pairs with a pinned TrialGPT reference; final answers parsed in the %r response format",
        len(examples),
        task.response_format,
    )
    ARIXTrainer(cfg, task, examples, model, tokenizer, ref_model, accelerator).train()


if __name__ == "__main__":
    main()
