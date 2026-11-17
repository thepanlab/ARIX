from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from accelerate import Accelerator
from safetensors.torch import load_file
from tiny_architectures import save_mxfp4_gpt_oss
from tokenizers import processors
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

from arix.config import (
    ARIXConfig,
    ARIXMethodConfig,
    GenerationConfig,
    ModelConfig,
    OutputConfig,
    TrainingConfig,
    save_config,
)
from arix.embedding import RationaleProjector
from arix.evaluation import default_output_file, load_inference_model, load_projector, read_run_config
from arix.modeling import dequantization_kwargs, encode_prompt, load_policy_and_tokenizer, load_reference_model
from arix.task import Example
from arix.trainer import ARIXTrainer


def test_dequantization_is_requested_only_for_mxfp4_checkpoints(tiny_model_dir: Path, tmp_path: Path) -> None:
    assert dequantization_kwargs(ModelConfig(name_or_path=str(tiny_model_dir))) == {}
    config = json.loads((tiny_model_dir / "config.json").read_text(encoding="utf-8"))
    config["quantization_config"] = {"quant_method": "mxfp4", "modules_to_not_convert": ["lm_head"]}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    quantization = dequantization_kwargs(ModelConfig(name_or_path=str(tmp_path)))["quantization_config"]
    assert isinstance(quantization, Mxfp4Config) and quantization.dequantize


class AcceptEverything:
    name = "accept_everything"

    def verify_batch(self, _examples: list[Example], responses: list[str]) -> list[bool]:
        return [True] * len(responses)

    def extract_batch(self, _examples: list[Example], responses: list[str]) -> list[Any]:
        return [None] * len(responses)


def expert_tensors(checkpoint: Path) -> dict[str, torch.Tensor]:
    return {name: value for name, value in load_file(checkpoint / "model.safetensors").items() if ".experts." in name}


def test_a_trained_mxfp4_gpt_oss_checkpoint_reloads_exactly(tiny_model_dir: Path, tmp_path: Path) -> None:
    source = save_mxfp4_gpt_oss(tmp_path / "gpt-oss-mxfp4", AutoTokenizer.from_pretrained(tiny_model_dir))
    assert any(name.endswith("_blocks") for name in expert_tensors(source))
    cfg = ARIXConfig(
        model=ModelConfig(name_or_path=str(source), torch_dtype="float32", gradient_checkpointing=False),
        training=TrainingConfig(
            epochs=1, batch_size=1, group_size=2, micro_batch_size=2, generation_batch_size=2, max_steps=1
        ),
        generation=GenerationConfig(max_new_tokens=4),
        arix=ARIXMethodConfig(bank_size=2),
        output=OutputConfig(output_dir=str(tmp_path / "run"), save_every_steps=0),
    )
    model, tokenizer = load_policy_and_tokenizer(cfg.model)
    assert not getattr(model, "_weight_conversions", None)
    experts = model.model.layers[0].mlp.experts
    assert experts.gate_up_proj.dtype == torch.float32 and experts.gate_up_proj.shape == (4, 64, 64)
    initial = experts.gate_up_proj.detach().clone()
    example = Example(uid="u0", prompt="Question:\nTom has 3 apples.\n", reference="3", target="3")
    trainer = ARIXTrainer(
        cfg, AcceptEverything(), [example], model, tokenizer, load_reference_model(cfg.model), Accelerator(cpu=True)
    )
    final = trainer.train()
    assert not torch.equal(trainer.policy.model.layers[0].mlp.experts.gate_up_proj, initial)

    input_ids = torch.tensor([tokenizer("Question:\nTom has 3 apples and 4 pears.\n").input_ids])
    trainer.policy.eval()
    with torch.no_grad():
        trained = trainer.policy(input_ids=input_ids).logits

    def check_reload(checkpoint: Path) -> None:
        names = sorted(expert_tensors(checkpoint))
        assert not any(name.endswith(("_blocks", "_scales")) for name in names), names
        assert "model.layers.0.mlp.experts.gate_up_proj" in names and "model.layers.0.mlp.experts.down_proj" in names
        assert "quantization_config" not in json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        reloaded, _ = load_inference_model(str(checkpoint), cfg.model, None, torch.device("cpu"))
        for (name, value), (_, expected) in zip(
            reloaded.state_dict().items(), trainer.policy.state_dict().items(), strict=True
        ):
            assert torch.equal(value, expected), name
        with torch.no_grad():
            assert torch.equal(reloaded(input_ids=input_ids).logits, trained)

    check_reload(final)
    plain = tmp_path / "plain_save"
    trainer.policy.save_pretrained(plain)
    tokenizer.save_pretrained(plain)
    check_reload(plain)
    record = AutoModelForCausalLM.from_pretrained(source, quantization_config=Mxfp4Config(dequantize=True))
    conversions: Any = record._weight_conversions
    patterns = [pattern for conversion in conversions for pattern in conversion.source_patterns]
    assert any(str(pattern).endswith("_blocks") for pattern in patterns)
    trainer.policy._weight_conversions = conversions
    check_reload(trainer.save_checkpoint("with_conversion_record"))


def test_encode_prompt_adds_special_tokens_only_when_the_example_asks(tiny_model_dir: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tiny_model_dir)
    bos = tokenizer.bos_token_id
    tokenizer.backend_tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", bos)]
    )
    plain = tokenizer("Question:\n1 + 1?", add_special_tokens=False).input_ids
    templated = Example(uid="a", prompt="Question:\n1 + 1?", reference="", target="2")
    assert encode_prompt(tokenizer, templated) == plain
    untemplated = Example(uid="b", prompt=templated.prompt, reference="", target="2", meta={"add_special_tokens": True})
    assert encode_prompt(tokenizer, untemplated) == [bos, *plain]


def test_checkpoint_config_projector_and_output_paths(tiny_model_dir: Path, tmp_path: Path) -> None:
    checkpoint = tmp_path / "final"
    cfg = ARIXConfig(model=ModelConfig(name_or_path="base"), output=OutputConfig(output_dir=str(tmp_path)))
    cfg.arix.proj_dim, cfg.arix.proj_seed = 16, 3
    save_config(cfg, checkpoint / "config.yaml")
    assert read_run_config(str(checkpoint)) == cfg
    assert read_run_config(str(tiny_model_dir)) is None
    assert read_run_config(str(tiny_model_dir), str(checkpoint / "config.yaml")) == cfg
    with pytest.raises(FileNotFoundError):
        read_run_config(str(checkpoint), str(tmp_path / "missing.yaml"))

    saved = RationaleProjector(32, 16, 3)
    saved.save(checkpoint / "projector.pt")
    loaded = load_projector(str(checkpoint), None, cfg.arix, hidden_size=32)
    rebuilt = load_projector(str(tiny_model_dir), None, cfg.arix, hidden_size=32)
    assert torch.equal(loaded.linear.weight, saved.linear.weight)
    assert torch.equal(rebuilt.linear.weight, saved.linear.weight)
    with pytest.raises(ValueError):
        load_projector(str(checkpoint), None, cfg.arix, hidden_size=64)
    with pytest.raises(FileNotFoundError):
        load_projector(str(checkpoint), str(tmp_path / "missing.pt"), cfg.arix, hidden_size=32)

    assert default_output_file(str(checkpoint), "gsm8k_test_set") == checkpoint / "eval" / "gsm8k_test_set.jsonl"
    assert default_output_file("Qwen/Qwen2.5-7B-Instruct", "math_test") == Path(
        "eval_outputs/Qwen_Qwen2.5-7B-Instruct/math_test.jsonl"
    )
