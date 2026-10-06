"""A deterministic, dependency-free feature-hashing encoder.

This is **not** a learned contrastive model. It exists so that every part of
the framework (indexing, retrieval, agents, serving, evaluation, CI) can run
instantly on any machine with zero downloads, and as a lexical baseline to
beat when you fine-tune a real model.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

import numpy as np

from clmkit.encoders.base import Encoder

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@lru_cache(maxsize=1 << 16)
def _hash(feature: str, dim: int) -> tuple[int, float]:
    # blake2b is stable across processes (unlike builtin hash(), which is salted).
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    index = int.from_bytes(digest[:4], "little") % dim
    sign = 1.0 if digest[4] & 1 else -1.0
    return index, sign


class HashingEncoder(Encoder):
    """Signed feature hashing of word n-grams + character n-grams.

    Args:
        dim: output dimensionality.
        word_ngrams: inclusive range of word n-gram sizes.
        char_ngrams: inclusive range of character n-gram sizes (``None`` disables);
            character n-grams make the encoder somewhat robust to typos and inflection.
        char_weight: weight of character features relative to word features.
        lowercase: case-fold before tokenising.
    """

    name = "hashing"

    def __init__(
        self,
        dim: int = 512,
        *,
        word_ngrams: tuple[int, int] = (1, 2),
        char_ngrams: tuple[int, int] | None = (3, 5),
        char_weight: float = 0.5,
        lowercase: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        if dim < 2:
            raise ValueError("dim must be >= 2")
        self._dim = int(dim)
        self.word_ngrams = tuple(word_ngrams)
        self.char_ngrams = tuple(char_ngrams) if char_ngrams else None
        self.char_weight = float(char_weight)
        self.lowercase = lowercase
        self.name = f"hashing-{self._dim}"

    @property
    def native_dim(self) -> int:
        return self._dim

    def fingerprint_config(self) -> dict[str, Any]:
        return {
            **super().fingerprint_config(),
            "word_ngrams": self.word_ngrams,
            "char_ngrams": self.char_ngrams,
            "char_weight": self.char_weight,
            "lowercase": self.lowercase,
        }

    def _features(self, text: str) -> Iterator[tuple[str, float]]:
        if self.lowercase:
            text = text.lower()
        tokens = _TOKEN_RE.findall(text)
        lo, hi = self.word_ngrams
        for n in range(lo, hi + 1):
            for i in range(len(tokens) - n + 1):
                yield "w:" + " ".join(tokens[i : i + n]), 1.0
        if self.char_ngrams:
            clo, chi = self.char_ngrams
            for tok in tokens:
                padded = f"<{tok}>"
                for n in range(clo, chi + 1):
                    for i in range(len(padded) - n + 1):
                        yield "c:" + padded[i : i + n], self.char_weight

    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature, weight in self._features(text):
                index, sign = _hash(feature, self._dim)
                out[row, index] += sign * weight
        # Sublinear scaling dampens very frequent features (like log-TF).
        return np.sign(out) * np.log1p(np.abs(out))
