"""Long-term semantic memory for agents, with optional recency decay."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from clmkit.encoders.base import Encoder
from clmkit.index.base import VectorIndex
from clmkit.retrieval import MetadataFilter, Retriever
from clmkit.types import SearchHit

_CREATED = "_created_at"


class SemanticMemory:
    """Store facts/observations/episodes and recall the most relevant ones.

    ``score = cosine * 0.5 ** (age / half_life)`` when ``recency_half_life`` (seconds)
    is set, so stale memories fade unless they are very relevant.

    >>> from clmkit import HashingEncoder
    >>> mem = SemanticMemory(HashingEncoder(dim=256))
    >>> _ = mem.remember("The user's favourite language is Python", {"kind": "preference"})
    >>> mem.recall("which programming language does the user like?", k=1)[0].metadata["kind"]
    'preference'
    """

    def __init__(
        self,
        encoder: Encoder,
        *,
        index: VectorIndex | None = None,
        recency_half_life: float | None = None,
        instruction: str | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if recency_half_life is not None and recency_half_life <= 0:
            raise ValueError("recency_half_life must be > 0")
        self.retriever = Retriever(encoder, index, query_instruction=instruction)
        self.recency_half_life = recency_half_life
        self.clock = clock

    def __len__(self) -> int:
        return len(self.retriever)

    def remember(self, text: str, metadata: Mapping[str, Any] | None = None, *, id: str | None = None) -> str:
        meta = {**(metadata or {}), _CREATED: self.clock()}
        return self.retriever.add([text], ids=[id] if id else None, metadata=[meta])[0]

    def remember_many(
        self, texts: Sequence[str], metadata: Sequence[Mapping[str, Any] | None] | None = None
    ) -> list[str]:
        now = self.clock()
        metas = [{**(m or {}), _CREATED: now} for m in (metadata or [None] * len(texts))]
        return self.retriever.add(list(texts), metadata=metas)

    def recall(
        self, query: str, k: int = 5, *, min_score: float | None = None, filter: MetadataFilter | None = None
    ) -> list[SearchHit]:
        pool = k * 4 if self.recency_half_life else k
        hits = self.retriever.search(query, k=pool, filter=filter)
        if self.recency_half_life:
            now = self.clock()
            decayed = []
            for h in hits:
                age = max(0.0, now - float(h.metadata.get(_CREATED, now)))
                decayed.append(replace(h, score=h.score * 0.5 ** (age / self.recency_half_life)))
            hits = sorted(decayed, key=lambda h: h.score, reverse=True)
        hits = hits[:k]
        if min_score is not None:
            hits = [h for h in hits if h.score >= min_score]
        return hits

    def forget(self, ids: Sequence[str]) -> int:
        return self.retriever.delete(list(ids))

    def forget_older_than(self, seconds: float) -> int:
        cutoff = self.clock() - seconds
        old = [i for i, d in self.retriever.documents.items() if float(d.metadata.get(_CREATED, 0)) < cutoff]
        return self.forget(old)

    def save(self, path: str | Path) -> Path:
        return self.retriever.save(path)

    @classmethod
    def load(cls, path: str | Path, encoder: Encoder, **kwargs: Any) -> SemanticMemory:
        mem = cls(encoder, **kwargs)
        mem.retriever = Retriever.load(path, encoder, strict=True)
        return mem
