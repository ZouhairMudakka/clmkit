"""scikit-learn integration: use any clmkit encoder as a pipeline step.

>>> from sklearn.pipeline import make_pipeline
>>> from sklearn.linear_model import LogisticRegression
>>> from clmkit.integrations.sklearn import EmbeddingTransformer
>>> clf = make_pipeline(EmbeddingTransformer("hashing", encoder_kwargs={"dim": 256}),
...                     LogisticRegression(max_iter=1000))
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import numpy as np

from clmkit.utils import require

require("sklearn")
from sklearn.base import BaseEstimator, TransformerMixin  # noqa: E402

from clmkit.encoders import Encoder, load_encoder  # noqa: E402
from clmkit.types import EncodeKind  # noqa: E402


class EmbeddingTransformer(TransformerMixin, BaseEstimator):
    """Stateless text -> embedding transformer (``fit`` is a no-op).

    Args:
        encoder: an encoder spec accepted by :func:`clmkit.load_encoder`, or an Encoder.
        kind: ``"document"`` (default) or ``"query"``.
        instruction: optional task instruction (instruction-aware models).
        dim: Matryoshka truncation.
        batch_size: encoding batch size.
        encoder_kwargs: kwargs used when building ``encoder`` from a spec.
    """

    def __init__(
        self,
        encoder: Any = "hashing",
        *,
        kind: str = "document",
        instruction: str | None = None,
        dim: int | None = None,
        batch_size: int = 32,
        encoder_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.encoder = encoder
        self.kind = kind
        self.instruction = instruction
        self.dim = dim
        self.batch_size = batch_size
        self.encoder_kwargs = encoder_kwargs

    def _encoder(self) -> Encoder:
        if not hasattr(self, "encoder_"):
            self.encoder_ = load_encoder(self.encoder, **(self.encoder_kwargs or {}))
        return self.encoder_

    def set_params(self, **params: Any) -> EmbeddingTransformer:
        super().set_params(**params)
        if "encoder" in params or "encoder_kwargs" in params:
            self.__dict__.pop("encoder_", None)
        return self

    def fit(self, X: Iterable[str], y: Any = None) -> EmbeddingTransformer:
        self._encoder()
        return self

    def transform(self, X: Iterable[str]) -> np.ndarray:
        texts = [str(x) for x in X]
        return self._encoder().encode(
            texts,
            kind=cast(EncodeKind, self.kind),
            instruction=self.instruction,
            dim=self.dim,
            batch_size=self.batch_size,
        )

    def _more_tags(self) -> dict[str, Any]:  # scikit-learn < 1.6
        return {"X_types": ["string"], "requires_fit": False}

    def __sklearn_tags__(self) -> Any:  # scikit-learn >= 1.6
        tags = super().__sklearn_tags__()
        tags.input_tags.string = True
        tags.input_tags.two_d_array = False
        tags.requires_fit = False
        return tags
