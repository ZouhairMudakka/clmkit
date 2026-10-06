"""Exact (brute-force) index in numpy; memory use scales with corpus and vector dimension."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from clmkit.index.base import IndexResults, Metric, VectorIndex


class NumpyIndex(VectorIndex):
    """Exact top-k by matrix multiplication.

    Persistence uses ``.npy`` + JSON only (loaded with ``allow_pickle=False``).
    Load trusted, complete snapshots; multi-file saves are not transactional.
    """

    kind = "numpy"

    def __init__(self, dim: int, metric: Metric = "cosine", *, query_chunk: int = 1024) -> None:
        super().__init__(dim, metric)
        if isinstance(query_chunk, bool) or not isinstance(query_chunk, int) or query_chunk < 1:
            raise ValueError("query_chunk must be a positive integer")
        self._vectors = np.zeros((0, self.dim), dtype=np.float32)
        self._ids: list[str] = []
        self._pos: dict[str, int] = {}
        self.query_chunk = query_chunk

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, id_: object) -> bool:
        return id_ in self._pos

    def ids(self) -> list[str]:
        return list(self._ids)

    def get_vectors(self, ids: Sequence[str]) -> np.ndarray:
        return self._vectors[[self._pos[i] for i in ids]]

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        ids = list(ids)
        vectors = self._check_vectors(ids, vectors)
        start = len(self._ids)
        self._vectors = np.concatenate([self._vectors, vectors], axis=0)
        self._ids.extend(ids)
        self._pos.update({id_: start + j for j, id_ in enumerate(ids)})

    def remove(self, ids: Iterable[str]) -> int:
        drop = {i for i in ids if i in self._pos}
        if not drop:
            return 0
        keep = [j for j, id_ in enumerate(self._ids) if id_ not in drop]
        self._vectors = self._vectors[keep]
        self._ids = [self._ids[j] for j in keep]
        self._pos = {id_: j for j, id_ in enumerate(self._ids)}
        return len(drop)

    def search(self, queries: np.ndarray, k: int = 10, *, allowed_ids: Iterable[str] | None = None) -> IndexResults:
        if k < 1:
            raise ValueError("k must be >= 1")
        q = self._prepare_queries(queries)
        if allowed_ids is not None:
            rows = np.fromiter((self._pos[i] for i in set(allowed_ids) if i in self._pos), dtype=np.int64)
            rows.sort()
        else:
            rows = None
        matrix = self._vectors if rows is None else self._vectors[rows]
        n = matrix.shape[0]
        if n == 0:
            return [[] for _ in range(q.shape[0])]
        k_eff = min(k, n)
        results: IndexResults = []
        for start in range(0, q.shape[0], self.query_chunk):
            scores = q[start : start + self.query_chunk] @ matrix.T
            if k_eff < n:
                top = np.argpartition(-scores, k_eff - 1, axis=1)[:, :k_eff]
            else:
                top = np.tile(np.arange(n), (scores.shape[0], 1))
            top_scores = np.take_along_axis(scores, top, axis=1)
            order = np.argsort(-top_scores, axis=1, kind="stable")
            top = np.take_along_axis(top, order, axis=1)
            top_scores = np.take_along_axis(top_scores, order, axis=1)
            for r_idx, r_scores in zip(top, top_scores, strict=True):
                src = r_idx if rows is None else rows[r_idx]
                results.append([(self._ids[j], float(s)) for j, s in zip(src, r_scores, strict=True)])
        return results

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self._vectors, allow_pickle=False)
        (path / "ids.json").write_text(json.dumps(self._ids), encoding="utf-8")
        self._write_meta(path, {"query_chunk": self.query_chunk})
        return path

    @classmethod
    def load(cls, path: str | Path) -> NumpyIndex:
        path = Path(path)
        meta = cls.read_meta(path)
        if meta.get("kind") != cls.kind:
            raise ValueError(f"{path} holds a {meta.get('kind')!r} index, not {cls.kind!r}")
        index = cls(int(meta["dim"]), meta["metric"], query_chunk=meta.get("query_chunk", 1024))
        vectors = np.load(path / "vectors.npy", allow_pickle=False)
        ids = json.loads((path / "ids.json").read_text(encoding="utf-8"))
        if vectors.ndim != 2 or vectors.shape[1] != index.dim:
            raise ValueError(f"corrupt index at {path}: bad vector shape {vectors.shape}")
        if not isinstance(ids, list) or len(ids) != vectors.shape[0]:
            raise ValueError(f"corrupt index at {path}: ids/vectors mismatch")
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"corrupt index at {path}: invalid or duplicate ids")
        index._vectors = vectors.astype(np.float32, copy=False)
        if not np.isfinite(index._vectors).all():
            raise ValueError(f"corrupt index at {path}: vectors contain NaN or inf")
        index._ids = ids
        index._pos = {id_: j for j, id_ in enumerate(index._ids)}
        return index
