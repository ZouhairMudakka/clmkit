"""Synthetic assistant memory scoped by a server-supplied user identity.

Run: python templates/assistant_memory.py
Hashing is a deterministic lexical baseline, not a learned semantic model.
Authenticate users in your application. Never let a model or unverified request
choose trusted_user_id; this wrapper does not provide authentication or encryption.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from clmkit import Encoder, load_encoder
from clmkit.agents import SemanticMemory


class ScopedMemory:
    """Server-side wrapper that applies the same scope on every write and read.

    Keep the underlying store private to the application; do not expose the raw
    SemanticMemory tools. Store only data you have permission to retain.
    """

    def __init__(self, encoder: Encoder) -> None:
        self._memory = SemanticMemory(encoder)

    @staticmethod
    def _scope(trusted_user_id: str) -> dict[str, str]:
        if not trusted_user_id.strip():
            raise ValueError("trusted_user_id must be non-empty")
        return {"user_id": trusted_user_id}

    def remember(
        self,
        text: str,
        *,
        trusted_user_id: str,
        memory_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        scope = self._scope(trusted_user_id)
        if not text.strip() or not memory_id.strip():
            raise ValueError("text and memory_id must be non-empty")
        # JSON tuples avoid delimiter collisions between user IDs and memory IDs.
        scoped_id = json.dumps([trusted_user_id, memory_id], separators=(",", ":"))
        return self._memory.remember(text, {**(metadata or {}), **scope}, id=scoped_id)

    def recall(self, query: str, *, trusted_user_id: str, k: int = 3, min_score: float = 0.2) -> list[dict[str, Any]]:
        """Recall scoped facts; calibrate min_score on your encoder and corpus."""
        scope = self._scope(trusted_user_id)
        if not query.strip():
            return []
        return [hit.to_dict() for hit in self._memory.recall(query, k=k, filter=scope, min_score=min_score)]

    def save(self, path: str | Path) -> Path:
        """Write a trusted single-writer snapshot; multi-file saves are not atomic."""
        return self._memory.save(path)

    @classmethod
    def load(cls, path: str | Path, encoder: Encoder) -> ScopedMemory:
        loaded = cls(encoder)
        # SemanticMemory.load uses Retriever.load(..., strict=True) internally.
        loaded._memory = SemanticMemory.load(path, encoder)
        return loaded


def demo(encoder: Encoder | None = None) -> dict[str, Any]:
    encoder = encoder if encoder is not None else load_encoder("hashing")
    memory = ScopedMemory(encoder)
    memory.remember("I prefer concise project updates by email.", trusted_user_id="alice", memory_id="updates")
    memory.remember("I prefer detailed project updates by phone.", trusted_user_id="bob", memory_id="updates")
    query = "project updates preference"
    before = memory.recall(query, trusted_user_id="alice")
    with TemporaryDirectory(prefix="clmkit-memory-") as directory:
        memory.save(Path(directory) / "memory")
        loaded = ScopedMemory.load(Path(directory) / "memory", encoder)
        after = loaded.recall(query, trusted_user_id="alice")
    return {
        "use_case": "assistant_memory",
        "synthetic_data": True,
        "encoder": encoder.name,
        "alice": after,
        "bob": loaded.recall(query, trusted_user_id="bob"),
        "unknown_user": loaded.recall(query, trusted_user_id="charlie"),
        "reload_equal": before == after,
    }


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2, allow_nan=False))
