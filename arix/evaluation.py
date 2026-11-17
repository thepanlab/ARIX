from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO

import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import gather_object
from torch import Tensor, nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from arix.config import (
    ARIXConfig,
    ARIXMethodConfig,
    EvaluationConfig,
    GenerationConfig,
    ModelConfig,
    config_from_dict,
    read_yaml_mapping,
)
from arix.diversity import (
    SELF_BLEU_N,
    SELF_BLEU_TOKENIZERS,
    mean_pairwise_cosine_distance,
    self_bleu,
    word_tokenizer,
)
from arix.embedding import RationaleProjector, rationale_embedding
from arix.modeling import build_sequence_batch, encode_prompt, forward_logprobs_hidden, load_policy_and_tokenizer
from arix.rollout import generate_batch
from arix.task import Example, Task, decode_for_verification
from arix.utils import set_seed, setup_logging, write_json, write_jsonl

logger = logging.getLogger(__name__)

CHECKPOINT_CONFIG = "config.yaml"
PROJECTOR_FILE = "projector.pt"
PROCESS_GROUP_TIMEOUT = timedelta(hours=12)
DEFAULT_EMBED_BATCH_SIZE = 4
DTYPE_CHOICES = ("bfloat16", "float16", "float32", "auto")


def add_evaluation_arguments(parser: argparse.ArgumentParser, unit: str, default_batch_size: int) -> None:
    parser.add_argument("--model", required=True, help="checkpoint directory or Hugging Face model id")
    parser.add_argument("--config", help="ARIX config (default: <model>/config.yaml)")
    parser.add_argument("--num_samples", type=int, default=1, help=f"rollouts per {unit}")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy decoding")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, help="default: the config's generation.max_new_tokens")
    parser.add_argument(
        "--batch_size",
        type=int,
        help=f"sequences per generate() call (default: the config's evaluation.batch_size, else {default_batch_size})",
    )
    parser.add_argument("--diversity", action="store_true", help=f"Self-BLEU and cosine distance per {unit}")
    parser.add_argument("--self_bleu_n", type=int, default=SELF_BLEU_N, help="n-gram order of Self-BLEU")
    parser.add_argument(
        "--self_bleu_tokenizer",
        choices=SELF_BLEU_TOKENIZERS,
        default="regex",
        help="Self-BLEU tokens: 'regex' (word-character runs and punctuation marks) or 'nltk' "
        "(nltk.word_tokenize, as in Texygen; needs NLTK's punkt model)",
    )
    parser.add_argument("--projector", help="projector.pt (default: <model>/projector.pt, else rebuilt from its seed)")
    parser.add_argument(
        "--embed_batch_size",
        type=int,
        help="sequences per embedding forward pass (default: the config's evaluation.embed_batch_size, "
        f"else {DEFAULT_EMBED_BATCH_SIZE})",
    )
    parser.add_argument("--torch_dtype", choices=DTYPE_CHOICES)
    parser.add_argument("--output_file", help=f"per-{unit} JSONL (default: under <model>/eval/ or eval_outputs/)")
    parser.add_argument("--seed", type=int, default=1111, help="sampling seed; process r uses seed + r")


def check_evaluation_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.num_samples < 1:
        parser.error("--num_samples must be >= 1")
    if args.num_samples > 1 and args.temperature <= 0:
        parser.error("--num_samples > 1 needs sampling (--temperature > 0)")
    if args.diversity and args.num_samples < 2:
        parser.error("--diversity needs --num_samples >= 2")
    for option in ("max_new_tokens", "batch_size", "embed_batch_size", "self_bleu_n"):
        value = getattr(args, option)
        if value is not None and value < 1:
            parser.error(f"--{option} must be >= 1")
    if args.diversity:
        try:
            word_tokenizer(args.self_bleu_tokenizer)
        except (ImportError, LookupError) as error:
            parser.error(str(error))


@dataclass(frozen=True)
class EvaluationSettings:
    num_samples: int = 1
    batch_size: int = 64
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE
    diversity: bool = False
    self_bleu_n: int = SELF_BLEU_N
    self_bleu_tokenizer: str = "regex"


def evaluation_settings(
    args: argparse.Namespace, cfg: ARIXConfig | None, default_batch_size: int
) -> EvaluationSettings:
    section = cfg.evaluation if cfg is not None else EvaluationConfig()
    batch_size = args.batch_size if args.batch_size is not None else section.batch_size
    embed_batch_size = args.embed_batch_size if args.embed_batch_size is not None else section.embed_batch_size
    settings = EvaluationSettings(
        num_samples=args.num_samples,
        batch_size=default_batch_size if batch_size is None else batch_size,
        embed_batch_size=DEFAULT_EMBED_BATCH_SIZE if embed_batch_size is None else embed_batch_size,
        diversity=args.diversity,
        self_bleu_n=args.self_bleu_n,
        self_bleu_tokenizer=args.self_bleu_tokenizer,
    )
    if settings.batch_size < 1 or settings.embed_batch_size < 1:
        raise ValueError(
            f"batch sizes must be >= 1, got batch_size {settings.batch_size} and "
            f"embed_batch_size {settings.embed_batch_size} (config section 'evaluation')"
        )
    return settings


def evaluation_generation(
    args: argparse.Namespace, cfg: ARIXConfig | None, default_max_new_tokens: int | None = None
) -> GenerationConfig:
    if cfg is not None:
        base = cfg.generation
    elif default_max_new_tokens is not None:
        base = GenerationConfig(max_new_tokens=default_max_new_tokens)
    else:
        raise ValueError("give --max_new_tokens or a config with a generation section")
    return dataclasses.replace(
        base,
        max_new_tokens=args.max_new_tokens or base.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=0,
    )


def sampling_summary(settings: EvaluationSettings, gen: GenerationConfig, seed: int) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_samples": settings.num_samples,
        "temperature": gen.temperature,
        "top_p": gen.top_p,
        "max_new_tokens": gen.max_new_tokens,
        "seed": seed,
        "batch_size": settings.batch_size,
    }
    if settings.diversity:
        summary["embed_batch_size"] = settings.embed_batch_size
        summary["self_bleu_n"] = settings.self_bleu_n
        summary["self_bleu_tokenizer"] = settings.self_bleu_tokenizer
    return summary


def start_evaluation(seed: int) -> Accelerator:
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=PROCESS_GROUP_TIMEOUT)])
    setup_logging(main_process=accelerator.is_main_process)
    set_seed(seed + accelerator.process_index)
    return accelerator


def read_run_config(model: str, config: str | None = None) -> ARIXConfig | None:
    path = Path(config) if config else Path(model) / CHECKPOINT_CONFIG
    if not path.is_file():
        if config:
            raise FileNotFoundError(config)
        return None
    return config_from_dict(read_yaml_mapping(path))


def load_inference_model(
    model: str, settings: ModelConfig | None, torch_dtype: str | None, device: torch.device
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    settings = settings if settings is not None else ModelConfig()
    policy, tokenizer = load_policy_and_tokenizer(
        ModelConfig(
            name_or_path=model,
            torch_dtype=torch_dtype or settings.torch_dtype,
            attn_implementation=settings.attn_implementation,
            gradient_checkpointing=False,
            trust_remote_code=settings.trust_remote_code,
        )
    )
    policy.to(device).eval()  # type: ignore[arg-type]
    return policy, tokenizer


@torch.no_grad()
def embed_responses(
    model: nn.Module,
    prompt_ids: Sequence[Sequence[int]],
    response_ids: Sequence[Sequence[int]],
    projector: RationaleProjector,
    pooling: str,
    batch_size: int,
    pad_id: int,
) -> Tensor:
    device = next(model.parameters()).device
    projector = projector.to(device)
    embeddings = []
    for start in range(0, len(response_ids), batch_size):
        batch = build_sequence_batch(
            prompt_ids[start : start + batch_size], response_ids[start : start + batch_size], pad_id
        ).to(device)
        logprobs, hidden = forward_logprobs_hidden(model, batch, need_hidden=True)
        if hidden is None:
            raise RuntimeError("the model returned no hidden states")
        embeddings.append(rationale_embedding(hidden, logprobs, batch.response_mask, projector, pooling).cpu())
    return torch.cat(embeddings)


def load_projector(model: str, path: str | None, method: ARIXMethodConfig, hidden_size: int) -> RationaleProjector:
    file = Path(path) if path else Path(model) / PROJECTOR_FILE
    if file.is_file():
        projector = RationaleProjector.load(file)
        if projector.hidden_size != hidden_size:
            raise ValueError(f"{file} maps {projector.hidden_size}-dim hidden states, the model has {hidden_size}")
        return projector
    if path:
        raise FileNotFoundError(path)
    if method.proj_seed is None:
        raise ValueError("arix.proj_seed is unset: build the method settings through arix.config.ARIXConfig")
    logger.info("no %s: projector rebuilt from seed %d (%d dims)", PROJECTOR_FILE, method.proj_seed, method.proj_dim)
    return RationaleProjector(hidden_size, method.proj_dim, method.proj_seed)


@dataclass
class PromptRollouts:
    example: Example
    index: int
    response_ids: list[list[int]]
    texts: list[str]
    diversity: dict[str, float]


def sample_rollouts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    task: Task,
    examples: Sequence[Example],
    indices: Sequence[int],
    gen: GenerationConfig,
    settings: EvaluationSettings,
    projector: RationaleProjector | None = None,
    pooling: str = "softmax",
    longest_first: bool = False,
) -> Iterator[list[PromptRollouts]]:
    if len(examples) != len(indices):
        raise ValueError(f"{len(examples)} examples but {len(indices)} indices")
    if settings.diversity and projector is None:
        raise ValueError("the diversity measurement needs the run's projector")
    n = settings.num_samples
    chunk = max(1, settings.batch_size // n)
    prompt_ids = [encode_prompt(tokenizer, example) for example in examples]
    order = list(range(len(examples)))
    if longest_first:
        order.sort(key=lambda i: len(prompt_ids[i]), reverse=True)
    for start in range(0, len(order), chunk):
        batch = order[start : start + chunk]
        flat_prompts = [prompt_ids[i] for i in batch for _ in range(n)]
        responses = generate_batch(model, tokenizer, flat_prompts, gen, settings.batch_size)
        texts = decode_for_verification(task, tokenizer, responses)
        groups = [slice(k * n, (k + 1) * n) for k in range(len(batch))]
        diversity: list[dict[str, float]] = [{} for _ in batch]
        if settings.diversity and projector is not None:
            emb = embed_responses(
                model, flat_prompts, responses, projector, pooling, settings.embed_batch_size, tokenizer.pad_token_id
            )
            plain = list(tokenizer.batch_decode(responses, skip_special_tokens=True))
            diversity = [
                {
                    "self_bleu": self_bleu(plain[rows], n=settings.self_bleu_n, tokenizer=settings.self_bleu_tokenizer),
                    "cosine_distance": mean_pairwise_cosine_distance(emb[rows]),
                }
                for rows in groups
            ]
        yield [
            PromptRollouts(examples[i], indices[i], responses[rows], texts[rows], scores)
            for i, rows, scores in zip(batch, groups, diversity, strict=True)
        ]
        logger.info("evaluated %d / %d prompts on this process", min(start + chunk, len(order)), len(order))


def diversity_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not records or "self_bleu" not in records[0]:
        return {}
    return {
        "self_bleu": sum(record["self_bleu"] for record in records) / len(records),
        "mean_pairwise_cosine_distance": sum(record["cosine_distance"] for record in records) / len(records),
    }


def sampling_suffix(num_samples: int, temperature: float, top_p: float) -> str:
    if num_samples > 1:
        return f"_samples{num_samples}"
    if temperature > 0:
        return f"_t{temperature:g}" + (f"_p{top_p:g}" if top_p < 1 else "")
    return ""


def default_output_file(model: str, name: str) -> Path:
    model_dir = Path(model)
    root = model_dir / "eval" if model_dir.is_dir() else Path("eval_outputs") / re.sub(r"[^\w.-]+", "_", model)
    return root / f"{name}.jsonl"


def metrics_file(output_file: str | Path) -> Path:
    return Path(output_file).with_suffix(".metrics.json")


def shard_file(output_file: str | Path, rank: int) -> Path:
    return Path(output_file).with_suffix(f".rank{rank}.jsonl")


class RecordShard:
    def __init__(self, output_file: str | Path, rank: int) -> None:
        self.path = shard_file(output_file, rank)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            logger.warning("overwriting %s, left by an earlier evaluation", self.path)
        self._handle: TextIO = open(self.path, "w", encoding="utf-8")

    def write(self, records: Iterable[Mapping[str, Any]]) -> None:
        for record in records:
            self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> RecordShard:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.close()


def gather_records(accelerator: Accelerator, records: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    if accelerator.num_processes > 1:
        records = gather_object(records)
    if not accelerator.is_main_process:
        return None
    return sorted(records, key=lambda record: record["index"])


def write_outputs(
    records: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], output_file: str | Path, num_processes: int
) -> None:
    write_jsonl(records, output_file)
    write_json(summary, metrics_file(output_file))
    for rank in range(num_processes):
        shard_file(output_file, rank).unlink(missing_ok=True)
