from __future__ import annotations

import contextlib
import logging
import math
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType
from torch import Tensor, nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase, get_scheduler

from arix.bank import UPDATE_ACTIONS, BankEntry, BankStore
from arix.config import ARIXConfig, OutputConfig, save_config
from arix.embedding import RationaleProjector, lcs_mask, masked_subset_embedding, rationale_embedding
from arix.losses import (
    embedding_loss_grads,
    group_advantages,
    grpo_sequence_losses,
    info_nce_negative,
    info_nce_positive,
    partial_rewards,
)
from arix.modeling import (
    SequenceBatch,
    build_sequence_batch,
    encode_prompt,
    forward_outputs,
    full_kl_per_token,
    prepare_reference_model,
)
from arix.rollout import generate_batch, stop_token_ids, zero3_generation
from arix.task import Example, Task, decode_for_verification
from arix.utils import all_reduce_sum, append_jsonl, set_seed, write_json

logger = logging.getLogger(__name__)

FULL_KL_CHUNK = 256
TRAIN_LOG = "train_log.jsonl"
FINAL_CHECKPOINT = "final"
_SCHEDULERS = {"constant": "constant_with_warmup", "linear": "linear", "cosine": "cosine"}


@dataclass(frozen=True)
class LossSettings:
    clip_eps: float
    kl_beta: float
    kl_estimator: str
    length_normalize: bool
    max_new_tokens: int
    pooling: str
    group_size: int
    num_prompts: float

    @classmethod
    def from_config(cls, cfg: ARIXConfig, num_prompts: float) -> LossSettings:
        t = cfg.training
        return cls(
            clip_eps=t.clip_eps,
            kl_beta=t.kl_beta,
            kl_estimator=t.kl_estimator,
            length_normalize=t.length_normalize,
            max_new_tokens=cfg.generation.max_new_tokens,
            pooling=cfg.arix.pooling,
            group_size=t.group_size,
            num_prompts=num_prompts,
        )


@dataclass
class MicroBatch:
    batch: SequenceBatch
    index: list[int]
    old_logp: Tensor | None = None
    ref_logp: Tensor | None = None
    advantages: Tensor | None = None
    emb_grad: Tensor | None = None
    subset_masks: tuple[Tensor, Tensor] | None = None
    subset_grads: tuple[Tensor, Tensor] | None = None


@dataclass
class StepRollouts:
    examples: list[Example]
    responses: list[list[int]]
    texts: list[str]
    verdicts: list[bool]
    correct: Tensor


@dataclass
class StepSignals:
    rewards: Tensor
    anchor_cos: Tensor | None
    pinned_anchor: Tensor | None
    anchors: Tensor | None = None
    anchor_emb: Tensor | None = None
    entry_emb: dict[str, Tensor] = field(default_factory=dict)
    pos_losses: list[float] = field(default_factory=list)
    neg_losses: list[float] = field(default_factory=list)


@torch.no_grad()
def score_micro_batch(
    model: nn.Module,
    ref_model: nn.Module | None,
    batch: SequenceBatch,
    projector: RationaleProjector,
    pooling: str,
    need_emb: bool,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    out = forward_outputs(model, batch, need_hidden=need_emb)
    ref_logp = forward_outputs(ref_model, batch).token_logprobs if ref_model is not None else None
    emb = None
    if need_emb:
        assert out.hidden is not None
        emb = rationale_embedding(out.hidden, out.token_logprobs, batch.response_mask, projector, pooling)
    return out.token_logprobs, ref_logp, emb


def lcs_subset_masks(
    batch: SequenceBatch, responses: Sequence[Sequence[int]], anchor_tokens: Sequence[Sequence[int] | None]
) -> tuple[Tensor, Tensor]:
    mask = batch.response_mask
    lcs, exc = torch.zeros_like(mask), torch.zeros_like(mask)
    for row, (response, anchor) in enumerate(zip(responses, anchor_tokens, strict=True)):
        if anchor is None:
            continue
        start = batch.prompt_lens[row]
        selected = torch.tensor(lcs_mask(response, anchor), dtype=torch.bool, device=mask.device)
        lcs[row, start : start + len(selected)] = selected
        exc[row] = mask[row] & ~lcs[row]
    return lcs, exc


@torch.no_grad()
def subset_embeddings(
    model: nn.Module, batch: SequenceBatch, masks: tuple[Tensor, Tensor], projector: RationaleProjector, pooling: str
) -> tuple[Tensor, Tensor]:
    out = forward_outputs(model, batch, need_hidden=True)
    hidden = out.hidden
    assert hidden is not None
    lcs, exc = (
        masked_subset_embedding(hidden, out.token_logprobs, batch.response_mask, subset, projector, pooling)
        for subset in masks
    )
    return lcs, exc


def subset_gradients(masks: tuple[Tensor, Tensor], lcs_grad: Tensor, exc_grad: Tensor) -> tuple[Tensor, Tensor]:
    return lcs_grad * masks[0].any(dim=1, keepdim=True), exc_grad * masks[1].any(dim=1, keepdim=True)


def micro_batch_loss(
    model: nn.Module,
    ref_model: nn.Module | None,
    mb: MicroBatch,
    projector: RationaleProjector,
    settings: LossSettings,
) -> tuple[Tensor, dict[str, float]]:
    if mb.old_logp is None or mb.advantages is None:
        raise ValueError("the no-grad pass must run before the gradient pass")
    mask = mb.batch.response_mask
    need_hidden = mb.emb_grad is not None or mb.subset_grads is not None
    full_kl = settings.kl_beta > 0 and settings.kl_estimator == "full"
    out = forward_outputs(model, mb.batch, need_hidden=need_hidden, need_logits=full_kl)
    kl_full = None
    if full_kl:
        if ref_model is None:
            raise ValueError("kl_estimator='full' with kl_beta > 0 needs the reference model")
        with torch.no_grad():
            ref_out = forward_outputs(ref_model, mb.batch, need_logits=True)
        assert out.logits is not None and ref_out.logits is not None
        kl_full = full_kl_per_token(out.logits, ref_out.logits, mask, FULL_KL_CHUNK, start=out.logits_start)
    seq = grpo_sequence_losses(
        out.token_logprobs,
        mb.old_logp,
        mb.ref_logp,
        mb.advantages,
        mask,
        clip_eps=settings.clip_eps,
        beta=settings.kl_beta,
        kl_estimator=settings.kl_estimator,
        length_normalize=settings.length_normalize,
        max_new_tokens=settings.max_new_tokens,
        kl_full=kl_full,
    )
    loss = -seq.sum() / settings.group_size
    hidden = out.hidden
    if mb.emb_grad is not None:
        assert hidden is not None
        emb = rationale_embedding(hidden, out.token_logprobs, mask, projector, settings.pooling)
        loss = loss + (emb * mb.emb_grad).sum()
    if mb.subset_grads is not None and mb.subset_masks is not None:
        assert hidden is not None
        for subset, grad in zip(mb.subset_masks, mb.subset_grads, strict=True):
            emb = masked_subset_embedding(hidden, out.token_logprobs, mask, subset, projector, settings.pooling)
            loss = loss + (emb * grad).sum()
    loss = loss / settings.num_prompts

    stats = {"grpo_loss_sum": float(-seq.detach().sum()) / settings.group_size}
    with torch.no_grad():
        if kl_full is not None:
            kl = kl_full
        elif mb.ref_logp is not None:
            log_ratio = (mb.ref_logp - out.token_logprobs.detach()).masked_fill(~mask, 0.0)
            kl = log_ratio.exp() - log_ratio - 1.0
        else:
            kl = None
        if kl is not None:
            stats["kl_sum"] = float(kl.masked_fill(~mask, 0.0).sum())
            stats["kl_tokens"] = float(mask.sum())
    return loss, stats


class ARIXTrainer:
    def __init__(
        self,
        cfg: ARIXConfig,
        task: Task,
        train_examples: Sequence[Example],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        ref_model: PreTrainedModel | None,
        accelerator: Accelerator,
    ) -> None:
        cfg.validate(world_size=accelerator.num_processes)
        check_output_dir(cfg.output)
        t, a = cfg.training, cfg.arix
        if t.kl_beta > 0 and ref_model is None:
            raise ValueError("training.kl_beta > 0 requires a reference model")
        if tokenizer.pad_token_id is None:
            raise ValueError("the tokenizer needs a pad token (see arix.modeling.load_tokenizer)")
        self.cfg, self.task, self.tokenizer, self.accelerator = cfg, task, tokenizer, accelerator
        self.rank, self.world_size = accelerator.process_index, accelerator.num_processes
        set_seed(t.seed + self.rank)

        self.examples = _shard(train_examples, self.rank, self.world_size, t.batch_size)
        self.prompts_per_step = _prompts_per_process(t.batch_size, self.world_size, self.rank)
        self.max_prompts_per_step = math.ceil(t.batch_size / self.world_size)
        self.micro_batches_per_step = math.ceil(self.max_prompts_per_step * t.group_size / t.micro_batch_size)
        self.steps_per_epoch = math.ceil(len(train_examples) / t.batch_size)
        planned = t.epochs * self.steps_per_epoch
        self.total_steps = planned if t.max_steps is None else min(planned, t.max_steps)
        self.prompt_ids = {ex.uid: encode_prompt(tokenizer, ex) for ex in self.examples}

        self.need_emb = cfg.needs_embeddings
        self.kl_k3 = t.kl_beta > 0 and t.kl_estimator == "k3"
        uses_deepspeed = accelerator.distributed_type == DistributedType.DEEPSPEED
        zero_stage = accelerator.state.deepspeed_plugin.zero_stage if uses_deepspeed else 0
        self.zero3, self.collective_backward = _collective_flags(zero_stage)
        proj_seed = t.seed if a.proj_seed is None else a.proj_seed
        self.projector = RationaleProjector(model.config.hidden_size, a.proj_dim, proj_seed).to(accelerator.device)
        self.banks = BankStore(rank=self.rank)
        self.bank_generator = torch.Generator().manual_seed(t.seed + self.rank)
        self.settings = LossSettings.from_config(cfg, t.batch_size / self.world_size)

        if not uses_deepspeed and any(p.dtype in (torch.bfloat16, torch.float16) for p in model.parameters()):
            logger.warning(
                "the policy is trained in half precision without DeepSpeed's fp32 master weights; AdamW "
                "updates of size ~lr=%g are mostly lost to rounding. Load it in float32 with mixed "
                "precision, or use DeepSpeed.",
                t.learning_rate,
            )
        optimizer = torch.optim.AdamW(_parameter_groups(model, t.weight_decay), lr=t.learning_rate)
        if uses_deepspeed:
            _fill_deepspeed_auto_values(accelerator, cfg)
        self.model, self.optimizer = accelerator.prepare(model, optimizer)
        self.scheduler = get_scheduler(
            _SCHEDULERS[t.lr_scheduler],
            self.optimizer,
            num_warmup_steps=t.warmup_steps,
            num_training_steps=self.total_steps * t.ppo_epochs,
        )
        if accelerator.gradient_accumulation_steps != 1:
            raise ValueError("use gradient_accumulation_steps = 1: the trainer accumulates its micro-batches itself")
        self.policy = accelerator.unwrap_model(self.model)
        self.ref_model: nn.Module | None = None
        if t.kl_beta > 0 and ref_model is not None:
            self.ref_model = prepare_reference_model(ref_model, accelerator)
        self.stop_ids = set(stop_token_ids(self.policy, tokenizer))

        self.output_dir = Path(cfg.output.output_dir)
        self.global_step = 0
        self.epoch = 0
        logger.info(
            "ARIX (%s): %d prompts, %d processes, %d prompts/step (%d on process %d), %d steps/epoch, %d steps",
            cfg.method,
            len(train_examples),
            self.world_size,
            t.batch_size,
            self.prompts_per_step,
            self.rank,
            self.steps_per_epoch,
            self.total_steps,
        )

    def train(self) -> Path:
        out = self.cfg.output
        log_file = self.output_dir / TRAIN_LOG
        if out.overwrite and self.accelerator.is_main_process and log_file.exists():
            log_file.write_text("", encoding="utf-8")
            logger.warning("output.overwrite is set: truncated %s", log_file)
        start = time.time()
        for epoch in range(1, self.cfg.training.epochs + 1):
            self.epoch = epoch
            order = self._epoch_order(epoch)
            for index in range(self.steps_per_epoch):
                if self.global_step >= self.total_steps:
                    break
                batch = order[index * self.prompts_per_step : (index + 1) * self.prompts_per_step]
                record = self._train_step(batch, epoch)
                self.global_step += 1
                if self.global_step % out.log_every_steps == 0:
                    self._log(record, time.time() - start)
                if out.save_every_steps and self.global_step % out.save_every_steps == 0:
                    self.save_checkpoint(f"step_{self.global_step}")
            if self.global_step >= self.total_steps:
                break
        return self.save_checkpoint(FINAL_CHECKPOINT)

    def _epoch_order(self, epoch: int) -> list[Example]:
        order = list(self.examples)
        random.Random(self.cfg.training.seed + epoch).shuffle(order)
        needed = self.steps_per_epoch * self.prompts_per_step
        return [order[i % len(order)] for i in range(needed)]

    def _train_step(self, examples: list[Example], epoch: int) -> dict[str, float]:
        t, a = self.cfg.training, self.cfg.arix
        tic = time.perf_counter()
        rollouts = self._sample(examples)
        sampled = time.perf_counter()
        micro_batches = self._micro_batches(rollouts)
        before = time.perf_counter()
        emb = self._no_grad_pass(micro_batches, keep_old=True)
        signals = self._signals(rollouts, micro_batches, emb, epoch)
        record = self._step_record(rollouts, signals)
        scoring_time, update_time = time.perf_counter() - before, 0.0
        refresh = emb is not None and (a.use_contrastive or a.use_negative)
        for ppo_epoch in range(t.ppo_epochs):
            before = time.perf_counter()
            if ppo_epoch > 0 and refresh:
                current = self._no_grad_pass(micro_batches, keep_old=False)
                assert current is not None
                self._contrastive_signals(rollouts, micro_batches, current, signals)
            middle = time.perf_counter()
            grad_stats = self._gradient_pass(micro_batches)
            if ppo_epoch == 0:
                record.update(grad_stats)
            scoring_time += middle - before
            update_time += time.perf_counter() - middle
        if a.use_bank and emb is not None:
            before = time.perf_counter()
            record.update(self._update_banks(rollouts, emb, signals.entry_emb))
            scoring_time += time.perf_counter() - before
        record.update(time_generation=sampled - tic, time_scoring=scoring_time, time_update=update_time)
        record["time_step"] = time.perf_counter() - tic
        return record

    def _sample(self, examples: list[Example]) -> StepRollouts:
        t, gen = self.cfg.training, self.cfg.generation
        prompts = [self.prompt_ids[ex.uid] for ex in examples for _ in range(t.group_size)]
        count = len(prompts)
        gather = self.zero3 and gen.ds3_gather
        synced = self.zero3 and not gather
        if synced:
            prompts += [prompts[-1]] * (self.max_prompts_per_step * t.group_size - count)
        with zero3_generation(self.model, gather):
            responses = generate_batch(
                self.policy, self.tokenizer, prompts, gen, t.generation_batch_size, synced_gpus=synced
            )[:count]
        texts = decode_for_verification(self.task, self.tokenizer, responses)
        flat_examples = [ex for ex in examples for _ in range(t.group_size)]
        verdicts = [bool(v) for v in self.task.verify_batch(flat_examples, texts)]
        correct = torch.tensor(verdicts, dtype=torch.bool, device=self.accelerator.device)
        return StepRollouts(examples, responses, texts, verdicts, correct)

    def _micro_batches(self, rollouts: StepRollouts) -> list[MicroBatch]:
        size, group = self.cfg.training.micro_batch_size, self.cfg.training.group_size
        prompts = [self.prompt_ids[ex.uid] for ex in rollouts.examples for _ in range(group)]
        micro_batches = []
        for start in range(0, len(prompts), size):
            index = list(range(start, min(start + size, len(prompts))))
            batch = build_sequence_batch(
                [prompts[i] for i in index], [rollouts.responses[i] for i in index], self.tokenizer.pad_token_id
            )
            micro_batches.append(MicroBatch(batch=batch.to(self.accelerator.device), index=index))
        return micro_batches

    def _padding(self, micro_batches: list[MicroBatch], collective: bool) -> list[MicroBatch]:
        if not collective:
            return []
        return [micro_batches[-1]] * (self.micro_batches_per_step - len(micro_batches))

    def _no_grad_pass(self, micro_batches: list[MicroBatch], keep_old: bool) -> Tensor | None:
        self.policy.eval()
        ref_model = self.ref_model if keep_old and self.kl_k3 else None
        pooling = self.cfg.arix.pooling
        rows = []
        for mb in micro_batches:
            logp, ref_logp, emb = score_micro_batch(
                self.policy, ref_model, mb.batch, self.projector, pooling, self.need_emb
            )
            if keep_old:
                mb.old_logp, mb.ref_logp = logp, ref_logp
            if emb is not None:
                rows.append(emb)
        for mb in self._padding(micro_batches, self.zero3):
            score_micro_batch(self.policy, ref_model, mb.batch, self.projector, pooling, self.need_emb)
        return torch.cat(rows) if rows else None

    def _embed_entries(self, examples: list[Example]) -> dict[str, Tensor]:
        a = self.cfg.arix
        unique = {ex.uid: ex for ex in examples}
        items: list[tuple[list[int], list[int]]] = []
        sizes = {}
        for uid, ex in unique.items():
            entries = self.banks.get_or_create(ex, self.tokenizer, a.bank_size).entries
            sizes[uid] = len(entries)
            items += [(self.prompt_ids[uid], entry.token_ids) for entry in entries]
        count = len(items)
        if self.zero3:
            items += [items[-1]] * (self.max_prompts_per_step * (a.bank_size if a.use_bank else 1) - count)
        size = self.cfg.training.micro_batch_size
        chunks = []
        for start in range(0, len(items), size):
            chunk = items[start : start + size]
            batch = build_sequence_batch([p for p, _ in chunk], [e for _, e in chunk], self.tokenizer.pad_token_id)
            _, _, emb = score_micro_batch(
                self.policy, None, batch.to(self.accelerator.device), self.projector, a.pooling, need_emb=True
            )
            assert emb is not None
            chunks.append(emb)
        flat = torch.cat(chunks)[:count]
        result, offset = {}, 0
        for uid, size_m in sizes.items():
            result[uid] = flat[offset : offset + size_m]
            offset += size_m
        return result

    def _signals(
        self, rollouts: StepRollouts, micro_batches: list[MicroBatch], emb: Tensor | None, epoch: int
    ) -> StepSignals:
        t, a = self.cfg.training, self.cfg.arix
        group, n, device = t.group_size, len(rollouts.responses), self.accelerator.device
        signals = StepSignals(rewards=torch.zeros(n, device=device), anchor_cos=None, pinned_anchor=None)
        if emb is not None:
            signals.entry_emb = self._embed_entries(rollouts.examples)
            anchors = self._anchors(rollouts.examples, emb, signals.entry_emb, epoch)
            anchor_emb = torch.cat(
                [
                    signals.entry_emb[ex.uid][anchors[p * group : (p + 1) * group]]
                    for p, ex in enumerate(rollouts.examples)
                ]
            )
            signals.anchors, signals.anchor_emb = anchors, anchor_emb
            signals.anchor_cos = (emb * anchor_emb).sum(dim=-1)
            signals.pinned_anchor = anchors == 0
        cos = signals.anchor_cos if signals.anchor_cos is not None else torch.zeros(n, device=device)
        advantages = torch.zeros(n, device=device)
        for p in range(len(rollouts.examples)):
            rows = slice(p * group, (p + 1) * group)
            signals.rewards[rows] = partial_rewards(
                rollouts.correct[rows], cos[rows], a.use_partial_reward, a.partial_scale, a.partial_bias
            )
            advantages[rows] = group_advantages(signals.rewards[rows], t.adv_eps, t.std_normalize)
        for mb in micro_batches:
            mb.advantages = advantages[mb.index]
        if emb is not None:
            signals.pos_losses, signals.neg_losses = self._contrastive_signals(rollouts, micro_batches, emb, signals)
        return signals

    def _anchors(self, examples: list[Example], emb: Tensor, entry_emb: dict[str, Tensor], epoch: int) -> Tensor:
        a, group = self.cfg.arix, self.cfg.training.group_size
        anchors = torch.zeros(len(emb), dtype=torch.long, device=emb.device)
        if a.use_bank and epoch >= a.retrieval_start_epoch:
            for p, ex in enumerate(examples):
                rows = slice(p * group, (p + 1) * group)
                bank = self.banks.get_or_create(ex, self.tokenizer, a.bank_size)
                anchors[rows] = bank.retrieve(emb[rows], entry_emb[ex.uid], a.retrieval, self.bank_generator)
        return anchors

    def _contrastive_signals(
        self, rollouts: StepRollouts, micro_batches: list[MicroBatch], emb: Tensor, signals: StepSignals
    ) -> tuple[list[float], list[float]]:
        a, group = self.cfg.arix, self.cfg.training.group_size
        anchors, anchor_emb = signals.anchors, signals.anchor_emb
        if anchors is None or anchor_emb is None:
            raise ValueError("the anchors must be chosen before the contrastive signals")
        subset_emb = self._subset_pass(rollouts, micro_batches, anchors) if a.use_negative else None
        emb_grad = torch.zeros_like(emb)
        lcs_grad, exc_grad = torch.zeros_like(emb), torch.zeros_like(emb)
        alpha_pos = a.alpha_pos if a.use_contrastive else 0.0
        pos_losses: list[float] = []
        neg_losses: list[float] = []
        for p in range(len(rollouts.examples)):
            rows = slice(p * group, (p + 1) * group)
            correct = rollouts.correct[rows]
            pos_losses.append(float(info_nce_positive(emb[rows], anchor_emb[rows], correct, a.tau)))
            neg = None
            if subset_emb is not None:
                lcs_emb, exc_emb = subset_emb[0][rows], subset_emb[1][rows]
                neg_losses.append(float(info_nce_negative(lcs_emb, exc_emb, anchor_emb[rows], ~correct, a.tau)))
                neg = {"lcs_emb": lcs_emb, "exc_emb": exc_emb, "incorrect": ~correct, "alpha_neg": a.alpha_neg}
            if not (a.use_contrastive or a.use_negative):
                continue
            _, grads = embedding_loss_grads(emb[rows], anchor_emb[rows], correct, a.tau, alpha_pos, neg)
            if neg is None:
                emb_grad[rows] = grads
            else:
                emb_grad[rows], lcs_grad[rows], exc_grad[rows] = grads[0], grads[1], grads[2]
        for mb in micro_batches:
            if a.use_contrastive:
                mb.emb_grad = emb_grad[mb.index]
            if a.use_negative and mb.subset_masks is not None:
                mb.subset_grads = subset_gradients(mb.subset_masks, lcs_grad[mb.index], exc_grad[mb.index])
        return pos_losses, neg_losses

    def _subset_pass(
        self, rollouts: StepRollouts, micro_batches: list[MicroBatch], anchors: Tensor
    ) -> tuple[Tensor, Tensor]:
        a, group = self.cfg.arix, self.cfg.training.group_size
        if any(mb.subset_masks is None for mb in micro_batches):
            anchor_tokens: list[list[int] | None] = [None] * len(rollouts.responses)
            for p, ex in enumerate(rollouts.examples):
                entries = self.banks.get_or_create(ex, self.tokenizer, a.bank_size).entries
                for i in range(p * group, (p + 1) * group):
                    if not rollouts.verdicts[i]:
                        anchor_tokens[i] = entries[int(anchors[i])].token_ids
            for mb in micro_batches:
                mb.subset_masks = lcs_subset_masks(
                    mb.batch, [rollouts.responses[i] for i in mb.index], [anchor_tokens[i] for i in mb.index]
                )
        lcs_rows, exc_rows = [], []
        for mb in micro_batches:
            assert mb.subset_masks is not None
            lcs_emb, exc_emb = subset_embeddings(self.policy, mb.batch, mb.subset_masks, self.projector, a.pooling)
            lcs_rows.append(lcs_emb)
            exc_rows.append(exc_emb)
        for mb in self._padding(micro_batches, self.zero3):
            assert mb.subset_masks is not None
            subset_embeddings(self.policy, mb.batch, mb.subset_masks, self.projector, a.pooling)
        return torch.cat(lcs_rows), torch.cat(exc_rows)

    def _update_banks(self, rollouts: StepRollouts, emb: Tensor, entry_emb: dict[str, Tensor]) -> dict[str, float]:
        a, group = self.cfg.arix, self.cfg.training.group_size
        counts = dict.fromkeys(UPDATE_ACTIONS, 0)
        for p, ex in enumerate(rollouts.examples):
            bank = self.banks.get_or_create(ex, self.tokenizer, a.bank_size)
            current = entry_emb[ex.uid]
            for i in range(p * group, (p + 1) * group):
                if not rollouts.verdicts[i]:
                    continue
                candidate = BankEntry(
                    token_ids=list(rollouts.responses[i]),
                    text=rollouts.texts[i],
                    pinned=False,
                    source="rollout",
                    added_step=self.global_step + 1,
                    insertion_order=0,
                )
                action, current = bank.update(candidate, emb[i], current, a.bank_update, self.bank_generator)
                counts[action] += 1
            entry_emb[ex.uid] = current
        return {f"bank_{action}": float(count) for action, count in counts.items()}

    def _gradient_pass(self, micro_batches: list[MicroBatch]) -> dict[str, float]:
        self.model.train(self.cfg.model.gradient_checkpointing)
        totals = {"grpo_loss_sum": 0.0, "kl_sum": 0.0, "kl_tokens": 0.0}
        real = len(micro_batches)
        schedule = [*micro_batches, *self._padding(micro_batches, self.collective_backward)]
        last = len(schedule) - 1
        for k, mb in enumerate(schedule):
            sync = k == last
            _set_gradient_sync(self.accelerator, sync)
            with contextlib.nullcontext() if sync else self.accelerator.no_sync(self.model):
                loss, stats = micro_batch_loss(self.model, self.ref_model, mb, self.projector, self.settings)
                self.accelerator.backward(loss if k < real else loss * 0.0)
            if k < real:
                for key, value in stats.items():
                    totals[key] += value
        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.training.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        totals["grad_norm"] = float("nan") if grad_norm is None else float(grad_norm)
        return totals

    def _step_record(self, rollouts: StepRollouts, signals: StepSignals) -> dict[str, float]:
        lengths = [len(r) for r in rollouts.responses]
        record = {
            "num_prompts": float(len(rollouts.examples)),
            "num_rollouts": float(len(lengths)),
            "num_correct": float(sum(rollouts.verdicts)),
            "reward_sum": float(signals.rewards.sum()),
            "response_tokens": float(sum(lengths)),
            "truncated": float(sum(r[-1] not in self.stop_ids for r in rollouts.responses)),
        }
        if signals.anchor_cos is not None and signals.pinned_anchor is not None:
            record["anchor_cos_sum"] = float(signals.anchor_cos.sum())
            record["pinned_anchor_count"] = float(signals.pinned_anchor.sum())
            record["pos_loss_sum"] = sum(signals.pos_losses)
        if signals.neg_losses:
            record["neg_loss_sum"] = sum(signals.neg_losses)
        return record

    def _log(self, record: dict[str, float], elapsed: float) -> None:
        a, device = self.cfg.arix, self.accelerator.device
        local = {key: value for key, value in record.items() if not key.startswith("time_") and key != "grad_norm"}
        local["peak_memory_bytes"] = _peak_memory_bytes(device)
        totals = all_reduce_sum(self.accelerator, local)
        if a.use_bank:
            stats = self.banks.stats()
            totals.update(
                all_reduce_sum(
                    self.accelerator,
                    {
                        "bank_entries": stats["mean_size"] * stats["num_banks"],
                        "bank_full": stats["full_fraction"] * stats["num_banks"],
                        "bank_count": float(stats["num_banks"]),
                    },
                )
            )
        if not self.accelerator.is_main_process:
            return
        prompts, rollouts = totals["num_prompts"], totals["num_rollouts"]

        def ratio(key: str, denominator: float) -> float | None:
            return totals[key] / denominator if key in totals and denominator > 0 else None

        grpo = totals["grpo_loss_sum"] / prompts
        pos, neg = ratio("pos_loss_sum", prompts), ratio("neg_loss_sum", prompts)
        loss = grpo + (a.alpha_pos * pos if a.use_contrastive and pos is not None else 0.0)
        loss += a.alpha_neg * neg if a.use_negative and neg is not None else 0.0
        entry = {
            "step": self.global_step,
            "epoch": self.epoch,
            "lr": self.scheduler.get_last_lr()[0],
            "loss": loss,
            "grpo_loss": grpo,
            "pos_loss": pos,
            "neg_loss": neg,
            "mean_reward": totals["reward_sum"] / rollouts,
            "fraction_correct": totals["num_correct"] / rollouts,
            "mean_anchor_cos": ratio("anchor_cos_sum", rollouts),
            "pinned_anchor_fraction": ratio("pinned_anchor_count", rollouts),
            "kl": ratio("kl_sum", totals["kl_tokens"]),
            "mean_response_tokens": totals["response_tokens"] / rollouts,
            "fraction_truncated": totals["truncated"] / rollouts,
            "grad_norm": record["grad_norm"],
            "peak_gpu_memory_gib": totals["peak_memory_bytes"] / 2**30 if device.type == "cuda" else None,
        }
        if a.use_bank:
            entry.update(
                bank_mean_size=ratio("bank_entries", totals["bank_count"]),
                bank_full_fraction=ratio("bank_full", totals["bank_count"]),
                bank_inserted=totals["bank_inserted"],
                bank_replaced=totals["bank_replaced"],
                bank_rejected=totals["bank_rejected"],
            )
        entry.update({key: value for key, value in record.items() if key.startswith("time_")}, elapsed=elapsed)
        append_jsonl(entry, self.output_dir / TRAIN_LOG)
        logger.info(
            "step %d/%d epoch %d: loss %.4f reward %.3f correct %.3f",
            self.global_step,
            self.total_steps,
            self.epoch,
            loss,
            entry["mean_reward"],
            entry["fraction_correct"],
        )

    def save_checkpoint(self, name: str) -> Path:
        directory = self.output_dir / name
        accelerator = self.accelerator
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(self.model)
        self.policy.save_pretrained(
            directory, is_main_process=accelerator.is_main_process, state_dict=state_dict, save_original_format=False
        )
        if self.cfg.output.save_bank and self.need_emb:
            self.banks.save(directory)
        if accelerator.is_main_process:
            self.tokenizer.save_pretrained(directory)
            self.projector.save(directory / "projector.pt")
            save_config(self.cfg, directory / "config.yaml")
            write_json(self._trainer_state(), directory / "trainer_state.json")
        accelerator.wait_for_everyone()
        return directory

    def _trainer_state(self) -> dict[str, Any]:
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "steps_per_epoch": self.steps_per_epoch,
            "total_steps": self.total_steps,
            "num_processes": self.world_size,
            "prompts_per_process_per_step": [
                _prompts_per_process(self.cfg.training.batch_size, self.world_size, rank)
                for rank in range(self.world_size)
            ],
            "method": self.cfg.method,
        }


def check_output_dir(output: OutputConfig) -> None:
    directory = Path(output.output_dir)
    found = [name for name in (TRAIN_LOG, FINAL_CHECKPOINT) if (directory / name).exists()]
    if found and not output.overwrite:
        raise FileExistsError(
            f"{directory} already holds a training run ({', '.join(found)}); choose another output.output_dir "
            "or set output.overwrite=true to train into it anyway (train_log.jsonl is then truncated)"
        )


def _prompts_per_process(batch_size: int, world_size: int, rank: int) -> int:
    return batch_size // world_size + (1 if rank < batch_size % world_size else 0)


def _collective_flags(zero_stage: int) -> tuple[bool, bool]:
    return zero_stage == 3, zero_stage >= 2


def _shard(examples: Sequence[Example], rank: int, world_size: int, batch_size: int) -> list[Example]:
    ordered = sorted(examples, key=lambda ex: ex.uid)
    uids = [ex.uid for ex in ordered]
    if len(set(uids)) != len(uids):
        raise ValueError("Example.uid must be unique: it keys the reference bank")
    if batch_size < world_size:
        raise ValueError(f"a minibatch of {batch_size} prompts cannot be shared by {world_size} processes")
    if len(ordered) < world_size:
        raise ValueError(f"{len(ordered)} training examples cannot be sharded over {world_size} processes")
    return [ex for i, ex in enumerate(ordered) if i % batch_size % world_size == rank]


def _parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("the policy has no trainable parameters")
    decay = [p for p in params if len(getattr(p, "ds_shape", p.shape)) >= 2]
    no_decay = [p for p in params if len(getattr(p, "ds_shape", p.shape)) < 2]
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return [group for group in groups if group["params"]]


def _fill_deepspeed_auto_values(accelerator: Accelerator, cfg: ARIXConfig) -> None:
    config = accelerator.state.deepspeed_plugin.deepspeed_config
    for key, value in (
        ("train_micro_batch_size_per_gpu", cfg.training.micro_batch_size),
        ("gradient_clipping", cfg.training.max_grad_norm),
    ):
        if config.get(key, "auto") == "auto":
            config[key] = value
        elif config[key] != value:
            logger.warning("DeepSpeed config sets %s=%s; the ARIX config value %s is not used", key, config[key], value)


def _set_gradient_sync(accelerator: Accelerator, sync: bool) -> None:
    accelerator.gradient_state._set_sync_gradients(sync)
    wrapper = getattr(accelerator, "deepspeed_engine_wrapped", None)
    if wrapper is not None:
        wrapper.engine.set_gradient_accumulation_boundary(sync)


def _peak_memory_bytes(device: torch.device) -> float:
    return float(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0.0
