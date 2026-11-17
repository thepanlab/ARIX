from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

SELF_BLEU_N = 4
SELF_BLEU_TOKENIZERS = ("regex", "nltk")
_TOKEN = re.compile(r"\w+|[^\w\s]")
_EPSILON = 0.1


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text)


def word_tokenizer(name: str) -> Callable[[str], list[str]]:
    if name == "regex":
        return tokenize
    if name != "nltk":
        raise ValueError(f"the Self-BLEU tokenizer must be one of {SELF_BLEU_TOKENIZERS}, got {name!r}")
    try:
        from nltk import word_tokenize
    except ImportError as error:
        raise ImportError("the 'nltk' Self-BLEU tokenizer needs NLTK (pip install nltk)") from error
    try:
        word_tokenize("NLTK splits sentences with punkt. Then it splits words.")
    except LookupError as error:
        raise LookupError(
            "the 'nltk' Self-BLEU tokenizer (nltk.word_tokenize, as in Texygen) needs NLTK's punkt model: "
            "python -m nltk.downloader punkt_tab punkt"
        ) from error
    return word_tokenize


def self_bleu(texts: Sequence[str], n: int = SELF_BLEU_N, tokenizer: str = "regex") -> float:
    if len(texts) < 2:
        raise ValueError("Self-BLEU needs at least two texts")
    if n < 1:
        raise ValueError("the n-gram order n must be >= 1")
    split = word_tokenizer(tokenizer)
    tokens = [list(split(text)) for text in texts]
    bleu = _sentence_bleu_function()
    scores = [bleu(tokens[:i] + tokens[i + 1 :], hypothesis, n) for i, hypothesis in enumerate(tokens)]
    return sum(scores) / len(scores)


def mean_pairwise_cosine_distance(emb: Tensor) -> float:
    if emb.dim() != 2 or emb.shape[0] < 2:
        raise ValueError("need a [N, D] tensor with N >= 2")
    unit = F.normalize(emb.double(), dim=-1)
    cosine = unit @ unit.T
    i, j = torch.triu_indices(len(unit), len(unit), offset=1)
    return float((1.0 - cosine[i, j]).mean())


def _sentence_bleu_function() -> Callable[[list[list[str]], list[str], int], float]:
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    except ImportError:
        return sentence_bleu_method1
    smoothing = SmoothingFunction(epsilon=_EPSILON).method1

    def nltk_bleu(references: list[list[str]], hypothesis: list[str], max_n: int) -> float:
        return float(sentence_bleu(references, hypothesis, (1.0 / max_n,) * max_n, smoothing_function=smoothing))

    return nltk_bleu


def sentence_bleu_method1(references: Sequence[Sequence[str]], hypothesis: Sequence[str], max_n: int) -> float:
    hyp_len = len(hypothesis)
    if hyp_len == 0:
        return 0.0
    log_precision = 0.0
    for n in range(1, max_n + 1):
        hyp_counts = Counter(_ngrams(hypothesis, n))
        max_ref_counts: Counter[tuple[str, ...]] = Counter()
        for reference in references:
            max_ref_counts |= Counter(_ngrams(reference, n))
        matches = sum(min(count, max_ref_counts[gram]) for gram, count in hyp_counts.items())
        total = max(1, sum(hyp_counts.values()))
        if n == 1 and matches == 0:
            return 0.0
        precision = matches / total if matches > 0 else _EPSILON / total
        log_precision += math.log(precision) / max_n
    closest_ref_len = min((len(r) for r in references), key=lambda length: (abs(length - hyp_len), length))
    brevity = 1.0 if hyp_len > closest_ref_len else math.exp(1.0 - closest_ref_len / hyp_len)
    return brevity * math.exp(log_precision)


def _ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
