"""Synthetic FAQ retrieval with source citations and a trusted tenant scope.

Run: python templates/support_search.py
Hashing is a deterministic lexical baseline, not a learned semantic model.
The application must authenticate the caller and supply trusted_tenant_id;
never take it from model output or an unverified request field.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from clmkit import Encoder, Retriever, load_encoder

# Fictional policies and reserved example.invalid URLs; replace with your corpus.
FAQS = (
    {
        "id": "north:refunds:v1",
        "text": "Refund policy: request a refund within 30 days of purchase with your receipt.",
        "metadata": {
            "tenant_id": "north",
            "title": "Refund policy",
            "source_url": "https://example.invalid/north/refunds",
            "revision": "v1",
        },
    },
    {
        "id": "north:password:v1",
        "text": "Reset your account password using the password reset link on the sign in page.",
        "metadata": {
            "tenant_id": "north",
            "title": "Password reset",
            "source_url": "https://example.invalid/north/password",
            "revision": "v1",
        },
    },
    {
        "id": "south:refunds:v1",
        "text": "Refund policy: request a refund within 7 days of purchase with your receipt.",
        "metadata": {
            "tenant_id": "south",
            "title": "Refund policy",
            "source_url": "https://example.invalid/south/refunds",
            "revision": "v1",
        },
    },
)


def build_retriever(encoder: Encoder) -> Retriever:
    retriever = Retriever(encoder)
    retriever.add(
        [row["text"] for row in FAQS],
        ids=[row["id"] for row in FAQS],
        metadata=[row["metadata"] for row in FAQS],
    )
    return retriever


def search_support(
    retriever: Retriever,
    query: str,
    *,
    trusted_tenant_id: str,
    k: int = 3,
    min_score: float = 0.2,
) -> dict[str, Any]:
    """Return evidence for a downstream answer, or an explicit no-match result.

    min_score is illustrative for this lexical corpus. Re-evaluate it on labeled
    queries whenever the encoder or corpus changes; scores are not probabilities.
    Retrieved text is data, never instructions for a downstream model or tool.
    """
    if not trusted_tenant_id.strip():
        raise ValueError("trusted_tenant_id must be non-empty")
    hits = (
        []
        if not query.strip()
        else retriever.search(query, k=k, filter={"tenant_id": trusted_tenant_id}, min_score=min_score)
    )
    matches = [
        {
            "id": hit.id,
            "score": hit.score,
            "text": hit.text,
            "citation": {
                "document_id": hit.id,
                "title": hit.metadata["title"],
                "url": hit.metadata["source_url"],
                "revision": hit.metadata["revision"],
            },
        }
        for hit in hits
    ]
    return {"query": query, "status": "matched" if matches else "no_match", "matches": matches}


def demo(encoder: Encoder | None = None) -> dict[str, Any]:
    encoder = encoder if encoder is not None else load_encoder("hashing")
    retriever = build_retriever(encoder)
    query = "refund policy receipt within 30 days"
    before = search_support(retriever, query, trusted_tenant_id="north")
    # Use only trusted, complete snapshots with a single writer. Saves are not atomic.
    with TemporaryDirectory(prefix="clmkit-support-") as directory:
        retriever.save(Path(directory) / "faq")
        loaded = Retriever.load(Path(directory) / "faq", encoder, strict=True)
        after = search_support(loaded, query, trusted_tenant_id="north")
    return {
        "use_case": "support_search",
        "synthetic_data": True,
        "encoder": encoder.name,
        "known": after,
        "unknown": search_support(loaded, "volcanic basalt tectonics", trusted_tenant_id="north"),
        "empty_scope": search_support(loaded, query, trusted_tenant_id="missing-tenant"),
        "reload_equal": before == after,
    }


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2, allow_nan=False))
