"""High-level semantic search: encoder + vector index + document store (+ optional reranker)."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, overload

from clmkit.encoders.base import Encoder
from clmkit.index.base import VectorIndex, load_index
from clmkit.index.numpy_index import NumpyIndex
from clmkit.rerank import Reranker
from clmkit.types import SearchHit

logger = logging.getLogger(__name__)

MetadataFilter = Mapping[str, Any] | Callable[[dict[str, Any]], bool]


@dataclass
class Document:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class EncoderMismatchWarning(UserWarning):
    """The index was built with a different encoder than the one querying it."""


class Retriever:
    """Index documents and search them semantically.

    >>> from clmkit import HashingEncoder, Retriever
    >>> r = Retriever(HashingEncoder(dim=256))
    >>> _ = r.add(["Paris is the capital of France.", "Bananas are yellow."])
    >>> r.search("capital city of France", k=1)[0].text
    'Paris is the capital of France.'
    """

    def __init__(
        self,
        encoder: Encoder,
        index: VectorIndex | None = None,
        *,
        reranker: Reranker | None = None,
        query_instruction: str | None = None,
    ) -> None:
        self.encoder = encoder
        self.index = index if index is not None else NumpyIndex(encoder.dim)
        if self.index.dim != encoder.dim:
            raise ValueError(f"index dim {self.index.dim} != encoder dim {encoder.dim}")
        self.reranker = reranker
        self.query_instruction = query_instruction
        self.documents: dict[str, Document] = {}

    def __len__(self) -> int:
        return len(self.documents)

    def __contains__(self, id_: object) -> bool:
        return id_ in self.documents

    # --------------------------------------------------------------- write --
    def add(
        self,
        texts: Sequence[str],
        *,
        ids: Sequence[str] | None = None,
        metadata: Sequence[Mapping[str, Any] | None] | None = None,
        batch_size: int = 32,
    ) -> list[str]:
        """Embed and index documents. Returns their ids (random UUIDs unless given)."""
        if isinstance(texts, str):
            raise TypeError("add() expects a sequence of strings; wrap a single text in a list")
        texts = list(texts)
        ids = [uuid.uuid4().hex for _ in texts] if ids is None else [str(i) for i in ids]
        metas = [dict(m or {}) for m in metadata] if metadata is not None else [{} for _ in texts]
        if not (len(texts) == len(ids) == len(metas)):
            raise ValueError("texts, ids and metadata must have the same length")
        if not texts:
            return []
        for m in metas:
            json.dumps(m)  # fail early on metadata that cannot be persisted
        vectors = self.encoder.encode(texts, kind="document", batch_size=batch_size)
        self.index.add(ids, vectors)  # validates duplicates before we touch the store
        for id_, text, meta in zip(ids, texts, metas, strict=True):
            self.documents[id_] = Document(text, meta)
        return ids

    def delete(self, ids: Sequence[str]) -> int:
        removed = self.index.remove(ids)
        for id_ in ids:
            self.documents.pop(id_, None)
        return removed

    def get(self, id_: str) -> Document | None:
        return self.documents.get(id_)

    # ---------------------------------------------------------------- read --
    def _allowed(self, flt: MetadataFilter | None) -> set[str] | None:
        if flt is None:
            return None
        if callable(flt):
            return {i for i, d in self.documents.items() if flt(d.metadata)}
        return {i for i, d in self.documents.items() if all(d.metadata.get(k) == v for k, v in flt.items())}

    @overload
    def search(
        self,
        query: str,
        k: int = ...,
        *,
        filter: MetadataFilter | None = ...,
        rerank: bool | None = ...,
        rerank_candidates: int | None = ...,
        instruction: str | None = ...,
        min_score: float | None = ...,
    ) -> list[SearchHit]: ...
    @overload
    def search(
        self,
        query: list[str],
        k: int = ...,
        *,
        filter: MetadataFilter | None = ...,
        rerank: bool | None = ...,
        rerank_candidates: int | None = ...,
        instruction: str | None = ...,
        min_score: float | None = ...,
    ) -> list[list[SearchHit]]: ...

    def search(
        self,
        query: str | Sequence[str],
        k: int = 5,
        *,
        filter: MetadataFilter | None = None,
        rerank: bool | None = None,
        rerank_candidates: int | None = None,
        instruction: str | None = None,
        min_score: float | None = None,
    ) -> list[SearchHit] | list[list[SearchHit]]:
        """Semantic search.

        Args:
            query: one query or a list of queries.
            k: results per query.
            filter: metadata equality dict (``{"source": "wiki"}``) or predicate ``f(metadata) -> bool``.
            rerank: use the reranker (default: whenever one is configured).
            rerank_candidates: first-stage pool size for reranking (default ``4 * k``).
            instruction: task instruction for instruction-aware encoders (e.g. Qwen3).
            min_score: drop hits scoring below this (applied after reranking).
        """
        single = isinstance(query, str)
        queries: list[str] = [query] if isinstance(query, str) else list(query)
        use_rerank = self.reranker is not None if rerank is None else rerank
        if use_rerank and self.reranker is None:
            raise ValueError("rerank=True but no reranker is configured")
        pool = max(k, rerank_candidates or 4 * k) if use_rerank else k
        inst = instruction or self.query_instruction
        qvecs = self.encoder.encode(queries, kind="query", instruction=inst)
        raw = self.index.search(qvecs, pool, allowed_ids=self._allowed(filter))
        results: list[list[SearchHit]] = []
        for q, row in zip(queries, raw, strict=True):
            hits = [self._hit(id_, score) for id_, score in row if id_ in self.documents]
            if use_rerank and self.reranker is not None:
                hits = self.reranker.rerank(q, hits, k=k, instruction=inst)
            hits = hits[:k]
            if min_score is not None:
                hits = [h for h in hits if h.score >= min_score]
            results.append(hits)
        return results[0] if single else results

    def _hit(self, id_: str, score: float) -> SearchHit:
        doc = self.documents[id_]
        return SearchHit(id=id_, score=score, text=doc.text, metadata=dict(doc.metadata))

    # --------------------------------------------------------- persistence --
    def save(self, path: str | Path, *, extra_meta: Mapping[str, Any] | None = None) -> Path:
        """Persist index + documents (JSON Lines) + encoder fingerprint. No pickle.

        ``extra_meta`` is stored in ``retriever.json`` (the CLI records the encoder spec
        there so ``clmkit search --index DIR`` can rebuild the right encoder).
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.index.save(path / "index")
        with (path / "documents.jsonl").open("w", encoding="utf-8") as fh:
            for id_, doc in self.documents.items():
                fh.write(json.dumps({"id": id_, "text": doc.text, "metadata": doc.metadata}, ensure_ascii=False))
                fh.write("\n")
        meta = {
            **(extra_meta or {}),
            "encoder": self.encoder.fingerprint(),
            "count": len(self),
            "query_instruction": self.query_instruction,
        }
        (path / "retriever.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def read_meta(path: str | Path) -> dict[str, Any]:
        return json.loads((Path(path) / "retriever.json").read_text(encoding="utf-8"))

    @classmethod
    def load(
        cls, path: str | Path, encoder: Encoder, *, reranker: Reranker | None = None, strict: bool = False
    ) -> Retriever:
        """Load a saved retriever. ``strict=True`` raises if the encoder fingerprint differs."""
        import warnings

        path = Path(path)
        meta = json.loads((path / "retriever.json").read_text(encoding="utf-8"))
        if meta.get("encoder") != encoder.fingerprint():
            msg = (
                f"index at {path} was built with encoder {meta.get('encoder')!r} "
                f"but is being queried with {encoder.fingerprint()!r}"
            )
            if strict:
                raise ValueError(msg)
            warnings.warn(msg, EncoderMismatchWarning, stacklevel=2)
        retriever = cls(
            encoder, load_index(path / "index"), reranker=reranker, query_instruction=meta.get("query_instruction")
        )
        with (path / "documents.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    retriever.documents[str(rec["id"])] = Document(rec["text"], rec.get("metadata") or {})
        if len(retriever.documents) != len(retriever.index):
            raise ValueError(
                f"corrupt retriever at {path}: {len(retriever.documents)} documents vs {len(retriever.index)} vectors"
            )
        return retriever
