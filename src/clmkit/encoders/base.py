"""The :class:`Encoder` contract every backend implements."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np

from clmkit.types import EncodeKind
from clmkit.utils import l2_normalize, truncate_dims

InstructionArg = str | Sequence[str | None] | None


class Encoder(ABC):
    """Maps text to dense vectors. Subclasses implement :meth:`_encode` and :attr:`native_dim`.

    Shared behaviour handled here so every backend gets it for free:

    * asymmetric query/document **prompt templates** (``{instruction}``/``{text}``),
    * Matryoshka (MRL) **dimension truncation** via ``dim=`` or ``output_dim``,
    * **L2 normalisation** (so dot product == cosine similarity),
    * input validation and ``str`` vs ``list[str]`` convenience.
    """

    #: Human-readable identifier (used in saved indexes to detect encoder mismatch).
    name: str = "encoder"

    def __init__(
        self,
        *,
        query_template: str = "{text}",
        document_template: str = "{text}",
        default_instruction: str | None = None,
        output_dim: int | None = None,
    ) -> None:
        for tpl in (query_template, document_template):
            if "{text}" not in tpl:
                raise ValueError(f"template must contain '{{text}}': {tpl!r}")
        self.query_template = query_template
        self.document_template = document_template
        self.default_instruction = default_instruction
        self.output_dim = output_dim

    # ------------------------------------------------------------ abstract --
    @property
    @abstractmethod
    def native_dim(self) -> int:
        """Dimensionality produced by the underlying model before truncation."""

    @abstractmethod
    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        """Encode already-formatted texts into a ``(n, native_dim)`` float array."""

    # -------------------------------------------------------------- public --
    @property
    def dim(self) -> int:
        """Effective output dimensionality (after ``output_dim`` truncation)."""
        return self.output_dim or self.native_dim

    def format_texts(self, texts: Sequence[str], kind: EncodeKind, instruction: InstructionArg = None) -> list[str]:
        """Apply the query/document prompt template.

        ``instruction`` may be a single string, one per text, or ``None`` to use
        :attr:`default_instruction`. Documents normally carry no instruction.
        """
        template = self.query_template if kind == "query" else self.document_template
        if isinstance(instruction, str) or instruction is None:
            instructions: Sequence[str | None] = [instruction] * len(texts)
        else:
            instructions = list(instruction)
            if len(instructions) != len(texts):
                raise ValueError("instruction list must match the number of texts")
        out = []
        for text, inst in zip(texts, instructions, strict=True):
            inst = inst if inst is not None else self.default_instruction
            if "{instruction}" in template and inst is None:
                # A template that needs an instruction but has none degrades to the raw text.
                out.append(text)
            else:
                out.append(template.format(text=text, instruction=inst or ""))
        return out

    def encode(
        self,
        texts: str | Sequence[str],
        *,
        kind: EncodeKind = "document",
        instruction: InstructionArg = None,
        batch_size: int = 32,
        normalize: bool = True,
        dim: int | None = None,
    ) -> np.ndarray:
        """Encode text(s). Returns ``(dim,)`` for a single string, else ``(n, dim)`` float32."""
        if kind not in ("query", "document"):
            raise ValueError(f"kind must be 'query' or 'document', got {kind!r}")
        single = isinstance(texts, str)
        items: list[str] = [texts] if isinstance(texts, str) else list(texts)
        bad = [type(t).__name__ for t in items if not isinstance(t, str)]
        if bad:
            raise TypeError(f"encode() expects strings, got {sorted(set(bad))}")
        target_dim = dim or self.output_dim
        if not items:
            return np.zeros((0, target_dim or self.native_dim), dtype=np.float32)
        formatted = self.format_texts(items, kind, instruction)
        raw = self._encode(formatted, batch_size)
        with np.errstate(over="ignore", invalid="ignore"):
            emb = np.asarray(raw, dtype=np.float32)
        if emb.shape != (len(items), self.native_dim):
            raise RuntimeError(
                f"{type(self).__name__} returned shape {emb.shape}, expected {(len(items), self.native_dim)}"
            )
        if not np.isfinite(emb).all():
            raise RuntimeError(f"{type(self).__name__} returned non-finite float32 embeddings")
        emb = truncate_dims(emb, target_dim)
        if normalize:
            emb = l2_normalize(emb)
        return emb[0] if single else emb

    def encode_queries(
        self,
        texts: str | Sequence[str],
        *,
        instruction: InstructionArg = None,
        batch_size: int = 32,
        normalize: bool = True,
        dim: int | None = None,
    ) -> np.ndarray:
        return self.encode(
            texts, kind="query", instruction=instruction, batch_size=batch_size, normalize=normalize, dim=dim
        )

    def encode_documents(
        self, texts: str | Sequence[str], *, batch_size: int = 32, normalize: bool = True, dim: int | None = None
    ) -> np.ndarray:
        return self.encode(texts, kind="document", batch_size=batch_size, normalize=normalize, dim=dim)

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Cosine similarity matrix between row-sets ``a`` and ``b`` (normalises defensively)."""
        a2 = l2_normalize(np.atleast_2d(a))
        b2 = l2_normalize(np.atleast_2d(b))
        return a2 @ b2.T

    def fingerprint_config(self) -> dict[str, Any]:
        """Canonical vector configuration; subclasses should include their own settings.

        This is a compatibility check, not proof of immutable model weights. Callers
        must pin mutable/custom model state with the retriever's ``encoder_identity``.
        Per-call query instructions are intentionally excluded.
        """
        return {
            "encoder_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "name": self.name,
            "native_dim": self.native_dim,
            "dim": self.dim,
            "query_template": self.query_template,
            "document_template": self.document_template,
            "default_instruction": self.default_instruction,
        }

    def fingerprint(self) -> str:
        """Versioned digest of the vector-producing configuration (without credentials)."""
        canonical = json.dumps(self.fingerprint_config(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return "v2:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, dim={self.dim})"
