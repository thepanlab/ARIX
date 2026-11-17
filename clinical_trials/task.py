from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from arix.task import Example
from clinical_trials.data import ClinicalPair
from clinical_trials.prompts import build_messages, detect_response_format, format_assessment, render_prompt
from clinical_trials.verifier import (
    HARMONY,
    THINK,
    THINK_CLOSE,
    THINK_OPEN,
    check_response_format,
    is_correct,
    parse_final_label,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def _dropped_markers(tokenizer: PreTrainedTokenizerBase, markers: Sequence[str]) -> dict[int, str]:
    dropped = {}
    for marker in markers:
        ids = tokenizer.encode(marker, add_special_tokens=False)
        if len(ids) == 1 and tokenizer.decode(ids, skip_special_tokens=True) == "":
            dropped[ids[0]] = marker
    return dropped


def _decode_keeping(tokenizer: PreTrainedTokenizerBase, ids: list[int], keep: Mapping[int, str]) -> str:
    parts: list[str] = []
    start = 0
    for position, token in enumerate(ids):
        if token in keep:
            parts += [str(tokenizer.decode(ids[start:position], skip_special_tokens=True)), keep[token]]
            start = position + 1
    parts.append(str(tokenizer.decode(ids[start:], skip_special_tokens=True)))
    return "".join(parts)


def decode_responses(
    tokenizer: PreTrainedTokenizerBase, responses: Sequence[Sequence[int]], response_format: str
) -> list[str]:
    check_response_format(response_format)
    token_ids = [list(response) for response in responses]
    if response_format == HARMONY:
        return list(tokenizer.batch_decode(token_ids, skip_special_tokens=False))
    texts = list(tokenizer.batch_decode(token_ids, skip_special_tokens=True))
    if response_format == THINK:
        keep = _dropped_markers(tokenizer, (THINK_OPEN, THINK_CLOSE))
        if keep:
            texts = [
                _decode_keeping(tokenizer, ids, keep) if keep.keys() & set(ids) else text
                for ids, text in zip(token_ids, texts, strict=True)
            ]
    return texts


class ClinicalTask:
    name = "clinical_trials"

    def __init__(self, response_format: str | None = None) -> None:
        self.response_format: str | None = None if response_format is None else check_response_format(response_format)

    @classmethod
    def for_tokenizer(cls, tokenizer: PreTrainedTokenizerBase) -> ClinicalTask:
        return cls(detect_response_format(tokenizer))

    def decode_responses(self, tokenizer: PreTrainedTokenizerBase, responses: list[list[int]]) -> list[str]:
        if self.response_format is None:
            self.response_format = detect_response_format(tokenizer)
        return decode_responses(tokenizer, responses, self.response_format)

    def _known_format(self) -> str:
        if self.response_format is None:
            raise ValueError(
                "the response format is not known: build the task with ClinicalTask.for_tokenizer(tokenizer) or "
                "ClinicalTask(response_format), or decode the responses with ClinicalTask.decode_responses first"
            )
        return self.response_format

    def verify_batch(self, examples: list[Example], responses: list[str]) -> list[bool]:
        response_format = self._known_format()
        return [
            is_correct(response, ex.target, response_format) for ex, response in zip(examples, responses, strict=True)
        ]

    def extract_batch(self, examples: list[Example], responses: list[str]) -> list[int | None]:
        if len(examples) != len(responses):
            raise ValueError(f"{len(examples)} examples but {len(responses)} responses")
        response_format = self._known_format()
        return [parse_final_label(response, response_format) for response in responses]


def build_example(pair: ClinicalPair, tokenizer: PreTrainedTokenizerBase, system_in_user: bool = False) -> Example:
    messages = build_messages(pair.patient_note, pair.trial, system_in_user=system_in_user)
    prompt, templated = render_prompt(messages, tokenizer)
    return Example(
        uid=pair.uid,
        prompt=prompt,
        reference=format_assessment(pair.reference) if pair.reference is not None else "",
        target=pair.label,
        meta={
            "add_special_tokens": not templated,
            "corpus": pair.corpus,
            "patient_id": pair.patient_id,
            "trial_id": pair.trial_id,
        },
    )


def build_examples(
    pairs: Sequence[ClinicalPair], tokenizer: PreTrainedTokenizerBase, system_in_user: bool = False
) -> list[Example]:
    return [build_example(pair, tokenizer, system_in_user=system_in_user) for pair in pairs]
