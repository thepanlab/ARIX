from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@dataclass
class Example:
    uid: str
    prompt: str
    reference: str
    target: Any
    meta: dict = field(default_factory=dict)


@runtime_checkable
class Task(Protocol):
    name: str

    def verify_batch(self, examples: list[Example], responses: list[str]) -> list[bool]: ...

    def extract_batch(self, examples: list[Example], responses: list[str]) -> list[Any]: ...


@runtime_checkable
class ResponseDecoder(Protocol):
    def decode_responses(self, tokenizer: PreTrainedTokenizerBase, responses: list[list[int]]) -> list[str]: ...


def decode_for_verification(
    task: Task, tokenizer: PreTrainedTokenizerBase, responses: Sequence[Sequence[int]]
) -> list[str]:
    token_ids = [list(response) for response in responses]
    if isinstance(task, ResponseDecoder):
        texts = list(task.decode_responses(tokenizer, token_ids))
    else:
        texts = list(tokenizer.batch_decode(token_ids, skip_special_tokens=True))
    if len(texts) != len(token_ids):
        raise ValueError(f"{len(token_ids)} responses were decoded into {len(texts)} texts")
    return texts
