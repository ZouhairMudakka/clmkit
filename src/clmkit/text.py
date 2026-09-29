"""Text utilities for RAG ingestion."""

from __future__ import annotations

import re

_PARA_RE = re.compile(r"\n\s*\n")
_SENT_RE = re.compile(r"(?<=[.!?。！？])\s+")


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 100) -> list[str]:
    """Split ``text`` into chunks of at most ``max_chars``, preferring paragraph then
    sentence boundaries, with ``overlap`` characters carried between chunks.

    Character-based sizing keeps this tokenizer-agnostic; size it to roughly
    ``max_tokens * 3.5`` for English with Qwen/BPE tokenizers.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap must satisfy 0 <= overlap < max_chars")
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    # Break into units no longer than max_chars: paragraphs -> sentences -> hard splits.
    units: list[str] = []
    for para in _PARA_RE.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            units.append(para)
            continue
        for sent in _SENT_RE.split(para):
            sent = sent.strip()
            while len(sent) > max_chars:
                units.append(sent[:max_chars])
                sent = sent[max_chars:]
            if sent:
                units.append(sent)

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n{unit}" if current else unit
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            candidate = f"{tail} {unit}".strip() if tail else unit
            current = candidate if len(candidate) <= max_chars else unit
        else:
            current = unit
    if current:
        chunks.append(current)
    return chunks
