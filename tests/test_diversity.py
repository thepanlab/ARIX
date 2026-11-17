from __future__ import annotations

import inspect
import math
import random
import sys

import pytest
import torch

import arix.diversity
from arix.diversity import (
    SELF_BLEU_N,
    SELF_BLEU_TOKENIZERS,
    mean_pairwise_cosine_distance,
    self_bleu,
    sentence_bleu_method1,
    tokenize,
    word_tokenizer,
)

TEXTS = ["a b c d", "a b c e", "a  b\nd"]


def _nltk_bleu():
    bleu_score = pytest.importorskip("nltk.translate.bleu_score")
    return bleu_score.sentence_bleu, bleu_score.SmoothingFunction(epsilon=0.1).method1


def _random_tokens(rng: random.Random, vocabulary: list[str], low: int, high: int) -> list[str]:
    return [rng.choice(vocabulary) for _ in range(rng.randint(low, high))]


def test_self_bleu_of_three_token_sequences_by_hand() -> None:
    first = (1.0 * (2 / 3) * (1 / 2) * 0.1) ** 0.25
    second = ((3 / 4) * (2 / 3) * (1 / 2) * 0.1) ** 0.25
    third = math.exp(1.0 - 4 / 3) * (1.0 * (1 / 2) * 0.1 * 0.1) ** 0.25
    expected = (first + second + third) / 3
    assert self_bleu(TEXTS) == pytest.approx(expected, abs=1e-12)
    assert self_bleu(TEXTS, n=4, tokenizer="regex") == pytest.approx(expected, abs=1e-12)
    bleu2 = ((2 / 3) ** 0.5 + ((3 / 4) * (2 / 3)) ** 0.5 + math.exp(1.0 - 4 / 3) * (1 / 2) ** 0.5) / 3
    assert self_bleu(TEXTS, n=2) == pytest.approx(bleu2, abs=1e-12)
    assert abs(expected - bleu2) > 0.3
    assert self_bleu(["w x y z", "w x y z"]) == pytest.approx(1.0)
    assert self_bleu(["x y z", "x y z"]) == pytest.approx(0.1**0.25)


def test_the_internal_bleu_matches_nltk_on_random_token_lists() -> None:
    sentence_bleu, smoothing = _nltk_bleu()
    rng = random.Random(0)
    short = 0
    for _ in range(600):
        vocabulary = [f"w{k}" for k in range(rng.randint(2, 8))]
        max_n = rng.randint(1, 5)
        hypothesis = _random_tokens(rng, vocabulary, 0, 12)
        references = [_random_tokens(rng, vocabulary, 0, 12) for _ in range(rng.randint(1, 4))]
        short += len(hypothesis) < max_n
        expected = sentence_bleu(references, hypothesis, (1.0 / max_n,) * max_n, smoothing_function=smoothing)
        assert sentence_bleu_method1(references, hypothesis, max_n) == pytest.approx(float(expected), abs=1e-12)
    assert short > 50


@pytest.mark.parametrize("n", [3, 4])
def test_self_bleu_is_the_leave_one_out_mean_of_nltk_sentence_bleu(n: int, monkeypatch: pytest.MonkeyPatch) -> None:
    sentence_bleu, smoothing = _nltk_bleu()
    rng = random.Random(n)
    vocabulary = ["x", "=", "3", ".", "14", "so", "the", "answer", "is", "$", "\\frac", "{", "}"]
    for _ in range(20):
        texts = [" ".join(_random_tokens(rng, vocabulary, 1, 15)) for _ in range(rng.randint(2, 6))]
        tokens = [tokenize(text) for text in texts]
        expected = sum(
            sentence_bleu(tokens[:i] + tokens[i + 1 :], hypothesis, (1.0 / n,) * n, smoothing_function=smoothing)
            for i, hypothesis in enumerate(tokens)
        ) / len(tokens)
        assert self_bleu(texts, n=n) == pytest.approx(float(expected), abs=1e-12)
        if n == SELF_BLEU_N:
            assert self_bleu(texts) == pytest.approx(float(expected), abs=1e-12)
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "nltk.translate.bleu_score", None)
            assert self_bleu(texts, n=n) == pytest.approx(float(expected), abs=1e-12)


def test_the_default_settings_and_the_regex_tokenizer_are_pinned() -> None:
    parameters = inspect.signature(self_bleu).parameters
    assert SELF_BLEU_N == 4 and parameters["n"].default == 4 and parameters["tokenizer"].default == "regex"
    assert SELF_BLEU_TOKENIZERS == ("regex", "nltk")
    assert tokenize("$1,234.50 didn't \\frac") == ["$", "1", ",", "234", ".", "50", "didn", "'", "t", "\\", "frac"]
    assert tokenize("x = 3.14\n\tdef solution():") == ["x", "=", "3", ".", "14", "def", "solution", "(", ")", ":"]
    assert word_tokenizer("regex") is tokenize


def test_the_nltk_tokenizer_is_used_only_when_asked_and_needs_punkt(monkeypatch: pytest.MonkeyPatch) -> None:
    nltk = pytest.importorskip("nltk")
    calls: list[str] = []

    def by_spaces(text: str) -> list[str]:
        calls.append(text)
        return text.split(" ")

    monkeypatch.setattr(nltk, "word_tokenize", by_spaces)
    texts = ["a b, c", "a b c"]
    assert self_bleu(texts) == pytest.approx(self_bleu(texts, tokenizer="regex")) and not calls
    with_nltk = self_bleu(texts, tokenizer="nltk")
    assert calls[-2:] == texts
    assert with_nltk != pytest.approx(self_bleu(texts))

    def without_punkt(_text: str) -> list[str]:
        raise LookupError("Resource 'punkt_tab' not found.")

    monkeypatch.setattr(nltk, "word_tokenize", without_punkt)
    with pytest.raises(LookupError, match="python -m nltk.downloader punkt_tab punkt"):
        self_bleu(texts, tokenizer="nltk")
    with pytest.raises(LookupError, match="punkt"):
        word_tokenizer("nltk")
    assert self_bleu(texts) == pytest.approx(self_bleu(texts, tokenizer="regex"))


def test_self_bleu_rejects_bad_settings() -> None:
    with pytest.raises(ValueError, match="at least two"):
        self_bleu(["only one"])
    with pytest.raises(ValueError, match="n must be >= 1"):
        self_bleu(TEXTS, n=0)
    with pytest.raises(ValueError, match="tokenizer"):
        self_bleu(TEXTS, tokenizer="whitespace")
    assert self_bleu(["", ""]) == 0.0


def test_mean_pairwise_cosine_distance() -> None:
    assert mean_pairwise_cosine_distance(torch.eye(3)) == pytest.approx(1.0)
    row = torch.randn(1, 5, generator=torch.Generator().manual_seed(0))
    assert mean_pairwise_cosine_distance(row.repeat(4, 1)) == pytest.approx(0.0, abs=1e-12)
    rows = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    expected = (1.0 + 2 * (1.0 - 1.0 / math.sqrt(2.0))) / 3
    assert mean_pairwise_cosine_distance(rows) == pytest.approx(expected, abs=1e-12)
    scales = torch.tensor([[2.0], [0.5], [7.0]])
    assert mean_pairwise_cosine_distance(rows * scales) == pytest.approx(expected, abs=1e-12)
    assert mean_pairwise_cosine_distance(torch.tensor([[1.0, 2.0], [-1.0, -2.0]])) == pytest.approx(2.0)
    for bad in (torch.ones(1, 3), torch.ones(3)):
        with pytest.raises(ValueError):
            mean_pairwise_cosine_distance(bad)


def test_the_diversity_module_uses_nltk_when_it_is_installed() -> None:
    pytest.importorskip("nltk.translate.bleu_score")
    assert arix.diversity._sentence_bleu_function() is not sentence_bleu_method1
