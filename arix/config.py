from __future__ import annotations

import argparse
import os
import re
import types
import typing
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

DTYPE_NAMES = ("bfloat16", "float16", "float32", "auto")
KL_ESTIMATORS = ("k3", "full")
RETRIEVAL_MODES = ("similarity", "random")
BANK_UPDATE_RULES = ("redundancy", "fifo", "random")
POOLING_MODES = ("softmax", "mean")
LR_SCHEDULERS = ("constant", "linear", "cosine")
UNTEMPERED_SAMPLING = (
    "for training: rollouts must be drawn from pi_theta_old itself, whose untempered log-probabilities enter "
    "equations (7)-(8)"
)


@dataclass
class ModelConfig:
    name_or_path: str = ""
    tokenizer: str | None = None
    torch_dtype: str = "bfloat16"
    attn_implementation: str | None = None
    gradient_checkpointing: bool = True
    trust_remote_code: bool = False


@dataclass
class TrainingConfig:
    epochs: int = 10
    learning_rate: float = 1e-6
    batch_size: int = 32
    group_size: int = 64
    micro_batch_size: int = 8
    generation_batch_size: int = 64
    ppo_epochs: int = 1
    clip_eps: float = 0.2
    kl_beta: float = 0.04
    kl_estimator: str = "k3"
    adv_eps: float = 1e-4
    std_normalize: bool = True
    length_normalize: bool = True
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    lr_scheduler: str = "constant"
    warmup_steps: int = 0
    seed: int = 1111
    max_steps: int | None = None


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    prefill_chunk_size: int | None = None
    ds3_gather: bool = True


@dataclass
class ARIXMethodConfig:
    use_contrastive: bool = True
    alpha_pos: float = 1e-3
    tau: float = 0.2
    use_partial_reward: bool = True
    partial_scale: float = 0.1
    partial_bias: float = 0.2
    use_bank: bool = True
    bank_size: int = 5
    retrieval: str = "similarity"
    bank_update: str = "redundancy"
    retrieval_start_epoch: int = 2
    proj_dim: int = 64
    pooling: str = "softmax"
    proj_seed: int | None = None
    use_negative: bool = False
    alpha_neg: float = 5e-4


@dataclass
class OutputConfig:
    output_dir: str = ""
    save_every_steps: int = 500
    log_every_steps: int = 1
    save_bank: bool = True
    overwrite: bool = False


@dataclass
class EvaluationConfig:
    batch_size: int | None = None
    embed_batch_size: int | None = None


@dataclass
class ARIXConfig:
    method: str = "arix"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: dict[str, Any] = field(default_factory=dict)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    arix: ARIXMethodConfig = field(default_factory=ARIXMethodConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self) -> None:
        if self.arix.proj_seed is None:
            self.arix.proj_seed = self.training.seed

    def validate(self, world_size: int = 1) -> None:
        t, g, a, o, e = self.training, self.generation, self.arix, self.output, self.evaluation
        checks = [
            (self.method in METHOD_PRESETS, f"method must be one of {sorted(METHOD_PRESETS)}, got {self.method!r}"),
            (bool(self.model.name_or_path), "model.name_or_path is required"),
            (self.model.torch_dtype in DTYPE_NAMES, f"model.torch_dtype must be one of {DTYPE_NAMES}"),
            (world_size >= 1, f"world_size must be >= 1, got {world_size}"),
            (t.epochs >= 1, "training.epochs must be >= 1"),
            (t.learning_rate > 0, "training.learning_rate must be > 0"),
            (t.batch_size >= 1, "training.batch_size must be >= 1"),
            (
                t.batch_size >= world_size,
                f"training.batch_size ({t.batch_size}) must be at least the number of processes ({world_size}), "
                "so that every process trains on at least one prompt per step",
            ),
            (t.group_size >= 2, "training.group_size must be >= 2 (group statistics of equation (6))"),
            (t.micro_batch_size >= 1, "training.micro_batch_size must be >= 1"),
            (t.generation_batch_size >= 1, "training.generation_batch_size must be >= 1"),
            (t.ppo_epochs >= 1, "training.ppo_epochs must be >= 1"),
            (t.clip_eps > 0, "training.clip_eps must be > 0"),
            (t.kl_beta >= 0, "training.kl_beta must be >= 0"),
            (t.kl_estimator in KL_ESTIMATORS, f"training.kl_estimator must be one of {KL_ESTIMATORS}"),
            (t.adv_eps > 0, "training.adv_eps must be > 0"),
            (t.max_grad_norm > 0, "training.max_grad_norm must be > 0"),
            (t.weight_decay >= 0, "training.weight_decay must be >= 0"),
            (t.lr_scheduler in LR_SCHEDULERS, f"training.lr_scheduler must be one of {LR_SCHEDULERS}"),
            (t.warmup_steps >= 0, "training.warmup_steps must be >= 0"),
            (t.max_steps is None or t.max_steps >= 1, "training.max_steps must be >= 1 or null"),
            (g.max_new_tokens >= 1, "generation.max_new_tokens must be >= 1"),
            (g.temperature == 1.0, f"generation.temperature must be 1 {UNTEMPERED_SAMPLING}"),
            (g.top_p == 1.0, f"generation.top_p must be 1 {UNTEMPERED_SAMPLING}"),
            (g.top_k == 0, f"generation.top_k must be 0 {UNTEMPERED_SAMPLING}"),
            (
                g.prefill_chunk_size is None or g.prefill_chunk_size >= 1,
                "generation.prefill_chunk_size must be >= 1 or null",
            ),
            (a.alpha_pos >= 0, "arix.alpha_pos must be >= 0"),
            (a.tau > 0, "arix.tau must be > 0"),
            (a.bank_size >= 1, "arix.bank_size must be >= 1"),
            (a.retrieval in RETRIEVAL_MODES, f"arix.retrieval must be one of {RETRIEVAL_MODES}"),
            (a.bank_update in BANK_UPDATE_RULES, f"arix.bank_update must be one of {BANK_UPDATE_RULES}"),
            (a.retrieval_start_epoch >= 1, "arix.retrieval_start_epoch must be >= 1 (epochs start at 1)"),
            (a.proj_dim >= 1, "arix.proj_dim must be >= 1"),
            (a.pooling in POOLING_MODES, f"arix.pooling must be one of {POOLING_MODES}"),
            (a.alpha_neg >= 0, "arix.alpha_neg must be >= 0"),
            (bool(o.output_dir), "output.output_dir is required"),
            (o.save_every_steps >= 0, "output.save_every_steps must be >= 0"),
            (o.log_every_steps >= 1, "output.log_every_steps must be >= 1"),
            (e.batch_size is None or e.batch_size >= 1, "evaluation.batch_size must be >= 1 or null"),
            (e.embed_batch_size is None or e.embed_batch_size >= 1, "evaluation.embed_batch_size must be >= 1 or null"),
        ]
        errors = [message for ok, message in checks if not ok]
        if errors:
            raise ValueError("invalid ARIX configuration:\n  - " + "\n  - ".join(errors))

    @property
    def needs_embeddings(self) -> bool:
        a = self.arix
        return a.use_contrastive or a.use_partial_reward or a.use_bank or a.use_negative


METHOD_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "arix": {},
    "arix_wo_rb": {"arix": {"use_bank": False}},
    "grpo": {"arix": {"use_contrastive": False, "use_partial_reward": False, "use_bank": False}},
    "dr_grpo": {
        "arix": {"use_contrastive": False, "use_partial_reward": False, "use_bank": False},
        "training": {"std_normalize": False, "length_normalize": False},
    },
}

_SECTIONS = {
    "model": ModelConfig,
    "training": TrainingConfig,
    "generation": GenerationConfig,
    "arix": ARIXMethodConfig,
    "output": OutputConfig,
    "evaluation": EvaluationConfig,
}


def load_config(path: str | os.PathLike[str] | None, overrides: Sequence[str] | None = None) -> ARIXConfig:
    raw = read_yaml_mapping(path) if path is not None else {}
    for override in overrides or ():
        apply_override(raw, override)
    return _validated(config_from_dict(raw))


def load_run_config(
    config_paths: Sequence[str | os.PathLike[str]], overrides: Sequence[str] | None = None
) -> ARIXConfig:
    if not config_paths:
        raise ValueError("at least one config file is required")
    base, *overlays = config_paths
    raw = read_yaml_mapping(base)
    base_output_dir = (raw.get("output") or {}).get("output_dir")
    explicit_output_dir = any(item.removeprefix("--").startswith("output.output_dir=") for item in overrides or ())
    for path in overlays:
        overlay = read_yaml_mapping(path)
        explicit_output_dir |= "output_dir" in (overlay.get("output") or {})
        _merge(raw, overlay)
    if overlays and base_output_dir and not explicit_output_dir:
        suffix = "-".join(Path(path).stem for path in overlays)
        raw["output"] = {**(raw.get("output") or {}), "output_dir": f"{base_output_dir}-{suffix}"}
    for override in overrides or ():
        apply_override(raw, override)
    return _validated(config_from_dict(raw))


_OVERRIDE = re.compile(r"^(?:--)?[A-Za-z_][\w.]*=")


def parse_run_args(
    argv: Sequence[str] | None = None, description: str | None = None
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False
    )
    parser.add_argument("--config", action="append", required=True, help="base config, then optional overlays")
    parser.add_argument("overrides", nargs="*", help="config overrides such as training.seed=2222")
    args, extra = parser.parse_known_args(argv)
    unknown = [item for item in extra if not _OVERRIDE.match(item)]
    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    return args, [*args.overrides, *extra]


def read_yaml_mapping(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is not None and not isinstance(loaded, Mapping):
        raise ValueError(f"{path}: the top level of a config file must be a mapping")
    return dict(loaded or {})


def _validated(cfg: ARIXConfig) -> ARIXConfig:
    cfg.validate(world_size=int(os.environ.get("WORLD_SIZE", "1")))
    return cfg


def _merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> None:
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            merged = dict(base[key])
            _merge(merged, value)
            base[key] = merged
        else:
            base[key] = value


def apply_override(raw: dict[str, Any], override: str) -> None:
    key, sep, value = override.removeprefix("--").partition("=")
    if not sep or not key:
        raise ValueError(f"override {override!r} must have the form key.sub=value")
    *parents, leaf = key.strip().split(".")
    node = raw
    for part in parents:
        child = node.get(part)
        if child is None:
            child = node[part] = {}
        if not isinstance(child, dict):
            raise ValueError(f"override {override!r}: {part!r} is not a section")
        node = child
    node[leaf] = yaml.safe_load(value) if value.strip() else ""


def config_from_dict(raw: Mapping[str, Any]) -> ARIXConfig:
    unknown = set(raw) - {"method", "data", *_SECTIONS}
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    method = raw.get("method", ARIXConfig.method)
    if method not in METHOD_PRESETS:
        raise ValueError(f"method must be one of {sorted(METHOD_PRESETS)}, got {method!r}")
    sections = {name: section_type() for name, section_type in _SECTIONS.items()}
    for name, preset in METHOD_PRESETS[method].items():
        _update_section(sections[name], preset, name)
    for name in _SECTIONS:
        values = raw.get(name)
        if values is None:
            continue
        if not isinstance(values, Mapping):
            raise ValueError(f"config section {name!r} must be a mapping")
        _update_section(sections[name], values, name)
    data = raw.get("data")
    if data is not None and not isinstance(data, Mapping):
        raise ValueError("config section 'data' must be a mapping")
    return ARIXConfig(method=method, data=dict(data or {}), **sections)


def config_to_dict(cfg: ARIXConfig) -> dict[str, Any]:
    return asdict(cfg)


def save_config(cfg: ARIXConfig, path: str | os.PathLike[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config_to_dict(cfg), handle, sort_keys=False)


def _update_section(section: Any, values: Mapping[str, Any], name: str) -> None:
    hints = typing.get_type_hints(type(section))
    known = {f.name for f in fields(section)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown keys in config section {name!r}: {sorted(unknown)}")
    for key, value in values.items():
        setattr(section, key, _coerce(value, hints[key], f"{name}.{key}"))


def _coerce(value: Any, annotation: Any, key: str) -> Any:
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        options = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        if value is None or (isinstance(value, str) and value.strip().lower() in ("none", "null", "~")):
            return None
        return _coerce(value, options[0], key)
    try:
        if annotation is bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.strip().lower() in ("true", "yes", "on", "1"):
                return True
            if isinstance(value, str) and value.strip().lower() in ("false", "no", "off", "0"):
                return False
        elif annotation is int and not isinstance(value, bool):
            if isinstance(value, int):
                return value
            number = float(value)
            if number.is_integer():
                return int(number)
        elif annotation is float and not isinstance(value, bool):
            return float(value)
        elif annotation is str and isinstance(value, (str, int, float)) and not isinstance(value, bool):
            return str(value)
    except (TypeError, ValueError):
        pass
    type_name = getattr(annotation, "__name__", annotation)
    raise ValueError(f"config value {key}={value!r} cannot be converted to {type_name}")
