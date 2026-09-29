"""FAISS-backed index (``pip install "clmkit[faiss]"``) for large corpora.

Supports exact inner product (``"flat"``) and approximate HNSW (``"hnsw"``).
Filtering (``allowed_ids``) is done by over-fetching then post-filtering, which is
exact for ``"flat"`` only when enough candidates survive; see docs/GAP_ANALYSIS.md.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from clmkit.index.base import IndexResults, Metric, VectorIndex
from clmkit.utils import require


class FaissIndex(VectorIndex):
    kind = "faiss"

    def __init__(
        self,
        dim: int,
        metric: Metric = "cosine",
        *,
        factory: str = "flat",
        hnsw_m: int = 32,
        ef_search: int = 64,
        filter_overfetch: int = 10,
    ) -> None:
        super().__init__(dim, metric)
        self._faiss = require("faiss")
        if factory not in ("flat", "hnsw"):
            raise ValueError("factory must be 'flat' or 'hnsw'")
        self.factory = factory
        self.hnsw_m = hnsw_m
        self.ef_search = ef_search
        self.filter_overfetch = filter_overfetch
        self._index = self._new_index()
        self._ids: dict[int, str] = {}
        self._rev: dict[str, int] = {}
        self._next = 0

    def _new_index(self):
        faiss = self._faiss
        if self.factory == "hnsw":
            base = faiss.IndexHNSWFlat(self.dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            base.hnsw.efSearch = self.ef_search
        else:
            base = faiss.IndexFlatIP(self.dim)
        return faiss.IndexIDMap2(base)

    def __len__(self) -> int:
        return len(self._rev)

    def __contains__(self, id_: object) -> bool:
        return id_ in self._rev

    def ids(self) -> list[str]:
        return list(self._rev)

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        ids = list(ids)
        vectors = self._check_vectors(ids, vectors)
        nums = np.arange(self._next, self._next + len(ids), dtype=np.int64)
        self._next += len(ids)
        self._index.add_with_ids(np.ascontiguousarray(vectors), nums)
        for num, id_ in zip(nums.tolist(), ids, strict=True):
            self._ids[num] = id_
            self._rev[id_] = num

    def remove(self, ids: Iterable[str]) -> int:
        nums = [self._rev.pop(i) for i in list(ids) if i in self._rev]
        if not nums:
            return 0
        if self.factory == "hnsw":
            # HNSW does not support removal: rebuild from the remaining vectors.
            keep = list(self._rev.items())
            vecs = np.stack([self._index.reconstruct(n) for _, n in keep]) if keep else None
            for n in nums:
                self._ids.pop(n, None)
            self._index = self._new_index()
            if vecs is not None:
                self._index.add_with_ids(vecs, np.array([n for _, n in keep], dtype=np.int64))
        else:
            self._index.remove_ids(np.array(nums, dtype=np.int64))
            for n in nums:
                self._ids.pop(n, None)
        return len(nums)

    def search(self, queries: np.ndarray, k: int = 10, *, allowed_ids: Iterable[str] | None = None) -> IndexResults:
        if k < 1:
            raise ValueError("k must be >= 1")
        q = np.ascontiguousarray(self._prepare_queries(queries))
        if len(self) == 0:
            return [[] for _ in range(q.shape[0])]
        allowed = set(allowed_ids) if allowed_ids is not None else None
        fetch = k if allowed is None else min(len(self), k * self.filter_overfetch)
        scores, nums = self._index.search(q, min(fetch, len(self)))
        results: IndexResults = []
        for row_s, row_n in zip(scores, nums, strict=True):
            hits = []
            for s, n in zip(row_s, row_n, strict=True):
                if n < 0:
                    continue
                id_ = self._ids[int(n)]
                if allowed is not None and id_ not in allowed:
                    continue
                hits.append((id_, float(s)))
                if len(hits) == k:
                    break
            results.append(hits)
        return results

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(path / "faiss.index"))
        (path / "ids.json").write_text(json.dumps({str(k): v for k, v in self._ids.items()}), encoding="utf-8")
        self._write_meta(
            path, {"factory": self.factory, "hnsw_m": self.hnsw_m, "ef_search": self.ef_search, "next": self._next}
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> FaissIndex:
        path = Path(path)
        meta = cls.read_meta(path)
        index = cls(
            int(meta["dim"]),
            meta["metric"],
            factory=meta.get("factory", "flat"),
            hnsw_m=meta.get("hnsw_m", 32),
            ef_search=meta.get("ef_search", 64),
        )
        index._index = index._faiss.read_index(str(path / "faiss.index"))
        raw = json.loads((path / "ids.json").read_text(encoding="utf-8"))
        index._ids = {int(k): str(v) for k, v in raw.items()}
        index._rev = {v: k for k, v in index._ids.items()}
        index._next = int(meta.get("next", max(index._ids, default=-1) + 1))
        return index
