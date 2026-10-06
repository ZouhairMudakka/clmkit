"""Vector index contract."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

Metric = Literal["cosine", "dot"]
IndexResults = list[list[tuple[str, float]]]

INDEX_META = "index.json"


class VectorIndex(ABC):
    """Stores ``(id, vector)`` pairs and answers top-k similarity queries.

    Ids are strings. Metadata/text live in the :class:`~clmkit.retrieval.Retriever`
    document store, keeping indexes swappable (numpy, FAISS, a vector DB, ...).
    """

    kind: str = "base"

    def __init__(self, dim: int, metric: Metric = "cosine") -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        if metric not in ("cosine", "dot"):
            raise ValueError(f"metric must be 'cosine' or 'dot', got {metric!r}")
        self.dim = int(dim)
        self.metric: Metric = metric

    @abstractmethod
    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        """Insert vectors. Raises ``ValueError`` on duplicate ids."""

    @abstractmethod
    def search(self, queries: np.ndarray, k: int = 10, *, allowed_ids: Iterable[str] | None = None) -> IndexResults:
        """Top-``k`` ``(id, score)`` per query row, best first. ``allowed_ids`` restricts candidates."""

    @abstractmethod
    def remove(self, ids: Iterable[str]) -> int:
        """Delete ids; returns how many were present."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __contains__(self, id_: object) -> bool: ...

    @abstractmethod
    def ids(self) -> list[str]: ...

    @abstractmethod
    def save(self, path: str | Path) -> Path: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> VectorIndex: ...

    # -------------------------------------------------------------- helpers --
    def _check_vectors(self, ids: Sequence[str], vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected vectors of shape (n, {self.dim}), got {vectors.shape}")
        if len(ids) != vectors.shape[0]:
            raise ValueError(f"{len(ids)} ids for {vectors.shape[0]} vectors")
        if not np.isfinite(vectors).all():
            raise ValueError("vectors contain NaN or inf")
        if any(not isinstance(i, str) or not i for i in ids):
            raise TypeError("ids must be non-empty strings")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate ids within one add() call")
        dup = [i for i in ids if i in self]
        if dup:
            raise ValueError(f"ids already in index: {dup[:5]}")
        if self.metric == "cosine":
            from clmkit.utils import l2_normalize

            vectors = l2_normalize(vectors)
        return vectors

    def _prepare_queries(self, queries: np.ndarray) -> np.ndarray:
        q = np.asarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        if q.ndim != 2 or q.shape[1] != self.dim:
            raise ValueError(f"expected queries of shape (n, {self.dim}), got {q.shape}")
        if not np.isfinite(q).all():
            raise ValueError("queries contain NaN or inf")
        if self.metric == "cosine":
            from clmkit.utils import l2_normalize

            q = l2_normalize(q)
        return q

    def _write_meta(self, path: Path, extra: dict[str, Any] | None = None) -> None:
        meta = {"kind": self.kind, "dim": self.dim, "metric": self.metric, "format_version": 1, **(extra or {})}
        (path / INDEX_META).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @staticmethod
    def read_meta(path: str | Path) -> dict[str, Any]:
        """Read supported metadata; snapshots lacking a version use legacy v1."""
        meta = json.loads((Path(path) / INDEX_META).read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            raise ValueError(f"corrupt index at {path}: metadata must be an object")
        version = meta.get("format_version", 1)
        if type(version) is not int or version != 1:
            raise ValueError(f"unsupported index format_version {version!r} at {path}")
        return meta


def load_index(path: str | Path) -> VectorIndex:
    """Load any saved index, dispatching on the ``kind`` stored in ``index.json``."""
    from clmkit.registry import INDEXES

    kind = VectorIndex.read_meta(path)["kind"]
    return INDEXES.get(kind).load(path)
