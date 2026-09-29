"""Shared lightweight data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EncodeKind = Literal["query", "document"]
"""Asymmetric models (Qwen3-Embedding, E5, BGE) encode queries and documents differently."""


@dataclass(frozen=True)
class SearchHit:
    """One retrieval result."""

    id: str
    score: float
    text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "score": self.score, "text": self.text, "metadata": dict(self.metadata)}
