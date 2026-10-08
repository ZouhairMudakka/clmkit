"""Training data: examples, JSONL I/O, false-negative-aware batching, hard-negative mining.

JSONL schema (one object per line)::

    {"query": "...", "positive": "...", "negatives": ["...", "..."], "instruction": "...", "score": 0.8}

Aliases accepted: ``anchor``/``question`` for query, ``pos``/``document`` for positive,
``neg``/``hard_negatives`` for negatives. ``negatives``, ``instruction``, ``score`` and
``label`` (the query/positive relevance class) are optional.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from clmkit.encoders.base import Encoder

_ALIASES = {
    "query": ("query", "anchor", "question"),
    "positive": ("positive", "pos", "document"),
    "negatives": ("negatives", "neg", "hard_negatives"),
}


@dataclass
class ContrastiveExample:
    query: str
    positive: str
    negatives: list[str] = field(default_factory=list)
    instruction: str | None = None
    score: float | None = None
    #: Shared relevance class for the query and positive (e.g. an intent).
    label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(self.positive, str) or not self.positive.strip():
            raise ValueError("positive must be a non-empty string")
        if isinstance(self.negatives, str):
            self.negatives = [self.negatives]
        if not all(isinstance(n, str) for n in self.negatives):
            raise ValueError("negatives must be strings")
        if self.label is not None and (not isinstance(self.label, str) or not self.label.strip()):
            raise ValueError("label must be a non-empty string or None")
        if self.score is not None:
            self.score = float(self.score)
            if not math.isfinite(self.score):
                raise ValueError("score must be finite")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContrastiveExample:
        def pick(key: str) -> Any:
            for alias in _ALIASES[key]:
                if alias in data:
                    return data[alias]
            return None

        return cls(
            query=pick("query"),
            positive=pick("positive"),
            negatives=pick("negatives") or [],
            instruction=data.get("instruction"),
            score=data.get("score"),
            label=data.get("label"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, [])}


def load_examples(path: str | Path) -> list[ContrastiveExample]:
    """Load ``.jsonl`` (one example per line) or ``.json`` (a list of examples)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        rows = json.loads(text)
        if not isinstance(rows, list):
            raise ValueError(f"{path}: expected a JSON list of examples")
        numbered = list(enumerate(rows, start=1))
    else:
        numbered = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            if line.strip():
                try:
                    numbered.append((lineno, json.loads(line)))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
    examples = []
    for lineno, row in numbered:
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{lineno}: expected an object, got {type(row).__name__}")
        try:
            examples.append(ContrastiveExample.from_dict(row))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return examples


def save_examples(path: str | Path, examples: Sequence[ContrastiveExample]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
    return path


def iter_batches(
    examples: Sequence[ContrastiveExample],
    batch_size: int,
    *,
    shuffle: bool = True,
    seed: int = 0,
    drop_last: bool = False,
    avoid_duplicates: bool = True,
    avoid_same_label: bool = False,
) -> Iterator[list[ContrastiveExample]]:
    """Yield batches for in-batch-negative training.

    With ``avoid_duplicates`` an example is deferred to a later batch if its query or
    positive text already appears in the current one: a duplicate positive would be
    treated as a *negative* for the other query - a guaranteed false negative.
    ``avoid_same_label`` additionally requires labels on every example and permits
    at most one example per label in the entire batch, retaining text deduplication
    even when ``avoid_duplicates=False``. This protects in-batch positives only;
    the trainer rejects explicit negatives in this mode unless disabled there.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if avoid_same_label and any(ex.label is None for ex in examples):
        raise ValueError("avoid_same_label requires a label on every example")
    order = list(range(len(examples)))
    if shuffle:
        random.Random(seed).shuffle(order)  # noqa: S311 - data shuffling, not crypto  # nosec B311
    if not (avoid_duplicates or avoid_same_label) or batch_size == 1:
        for start in range(0, len(order), batch_size):
            batch = [examples[i] for i in order[start : start + batch_size]]
            if len(batch) == batch_size or not drop_last:
                yield batch
        return

    # Assign each example to its earliest compatible batch. This is the same
    # ordering as repeatedly retrying deferred examples, without rescanning all
    # of them on every batch. A text's occupied positions form a successor map:
    # path compression skips consecutive occupied batches (e.g. a shared positive).
    # Store a single position directly until that text occurs a second time.
    occupied: dict[tuple[str, str], int | dict[int, int]] = {}

    def next_free(slots: int | dict[int, int], position: int) -> int:
        if isinstance(slots, int):
            return position + 1 if slots == position else position
        end = position
        while end in slots:
            end = slots[end]
        while position in slots:
            following = slots[position]
            slots[position] = end
            position = following
        return end

    batches: dict[int, list[ContrastiveExample]] = {}
    first = 0
    seen: set[tuple[str, str]] = set()
    for i in order:
        ex = examples[i]
        keys = {("text", ex.query), ("text", ex.positive)}
        if avoid_same_label and ex.label is not None:
            keys.add(("label", ex.label))
        # Keep the common no-conflict path as cheap as ordinary batching. Build
        # successor maps only once there are actually deferred examples.
        if not occupied:
            if not keys & seen:
                batch = batches.setdefault(first, [])
                batch.append(ex)
                seen.update(keys)
                if len(batch) == batch_size:
                    yield batches.pop(first)
                    first += 1
                    seen.clear()
                continue
            occupied.update((text, first) for text in seen)
            seen.clear()
        key_slots = [occupied.get(key, -1) for key in keys]
        position = first
        while True:
            candidate = position
            for slots in key_slots:
                candidate = next_free(slots, candidate)
            if len(batches.get(candidate, ())) == batch_size:
                candidate += 1
            if candidate == position:
                break
            position = candidate
        batch = batches.setdefault(position, [])
        batch.append(ex)
        for key in keys:
            occupied_slots = occupied.get(key)
            if occupied_slots is None:
                occupied[key] = position
            else:
                if isinstance(occupied_slots, int):
                    occupied_slots = {occupied_slots: occupied_slots + 1}
                    occupied[key] = occupied_slots
                occupied_slots[position] = next_free(occupied_slots, position + 1)
        # Emit completed leading batches immediately, preserving early iteration.
        while first in batches and len(batches[first]) == batch_size:
            yield batches.pop(first)
            first += 1
        if not batches:
            occupied.clear()
    for position in sorted(batches):
        batch = batches[position]
        if len(batch) < batch_size and drop_last:
            return
        yield batch


def split_examples(
    examples: Sequence[ContrastiveExample], eval_fraction: float = 0.1, seed: int = 0
) -> tuple[list[ContrastiveExample], list[ContrastiveExample]]:
    if not 0 < eval_fraction < 1:
        raise ValueError("eval_fraction must be in (0, 1)")
    items = list(examples)
    random.Random(seed).shuffle(items)  # noqa: S311 - data shuffling, not crypto  # nosec B311
    cut = max(1, round(len(items) * eval_fraction))
    return items[cut:], items[:cut]


def mine_hard_negatives(
    encoder: Encoder,
    examples: Sequence[ContrastiveExample],
    corpus: Sequence[str],
    *,
    num_negatives: int = 4,
    skip_top: int = 0,
    max_relative_score: float | None = 0.95,
    batch_size: int = 32,
    query_block_size: int = 128,
    corpus_block_size: int = 4096,
    corpus_labels: Sequence[str] | None = None,
) -> list[ContrastiveExample]:
    """Attach hard negatives retrieved from ``corpus`` with ``encoder``.

    Guards against false negatives (the #1 failure mode of naive mining):

    * the example's own positive is never used;
    * ``skip_top`` ignores the N most similar documents outright;
    * ``max_relative_score`` drops candidates scoring above ``ratio * sim(query, positive)``
      ("positive-aware" mining, NV-Retriever 2024).

    Labelled examples require aligned ``corpus_labels``; same-label candidates
    are excluded. Existing negatives must have known, different labels. Labels
    are retained on output, but label-aware training currently requires
    ``max_negatives=0`` because negative labels are not carried into the loss.

    Scores are computed in query/corpus blocks; the corpus embedding matrix is
    still resident. Ranking uses descending score, breaking ties by first corpus
    occurrence. As before, only ``skip_top + 3 * num_negatives + 1`` candidates
    are inspected before exclusions, so fewer negatives may be returned.
    """
    if num_negatives < 0 or skip_top < 0:
        raise ValueError("num_negatives and skip_top must be >= 0")
    if batch_size < 1 or query_block_size < 1 or corpus_block_size < 1:
        raise ValueError("batch_size, query_block_size and corpus_block_size must be >= 1")
    if max_relative_score is not None and not math.isfinite(max_relative_score):
        raise ValueError("max_relative_score must be finite or None")
    if num_negatives == 0:
        return [
            ContrastiveExample(e.query, e.positive, list(e.negatives), e.instruction, e.score, e.label)
            for e in examples
        ]
    labels_by_text: dict[str, str] = {}
    labelled = corpus_labels is not None or any(ex.label is not None for ex in examples)
    if labelled:
        if corpus_labels is None or len(corpus_labels) != len(corpus):
            raise ValueError("label-aware mining requires corpus_labels aligned with corpus")
        if any(ex.label is None for ex in examples):
            raise ValueError("label-aware mining requires a label on every example")
        for text, label in zip(corpus, corpus_labels, strict=True):
            if not isinstance(label, str) or not label.strip():
                raise ValueError("corpus_labels must be non-empty strings")
            if text in labels_by_text and labels_by_text[text] != label:
                raise ValueError("duplicate corpus text has conflicting labels")
            labels_by_text[text] = label
        for ex in examples:
            for text in (ex.query, ex.positive):
                if text in labels_by_text and labels_by_text[text] != ex.label:
                    raise ValueError("example text has a conflicting corpus label")
            if any(n not in labels_by_text or labels_by_text[n] == ex.label for n in ex.negatives):
                raise ValueError("existing negatives must have known corpus labels different from the example label")
    corpus = list(dict.fromkeys(corpus))  # dedupe, keep order
    if not corpus:
        raise ValueError("corpus is empty")
    if not examples:
        return []
    doc_emb = encoder.encode(corpus, kind="document", batch_size=batch_size)
    fetch = min(len(corpus), skip_top + num_negatives * 3 + 1)
    mined: list[ContrastiveExample] = []
    for start in range(0, len(examples), query_block_size):
        block = examples[start : start + query_block_size]
        q_emb = encoder.encode(
            [e.query for e in block], kind="query", instruction=[e.instruction for e in block], batch_size=batch_size
        )
        p_emb = encoder.encode([e.positive for e in block], kind="document", batch_size=batch_size)
        pos_scores = (q_emb * p_emb).sum(axis=1)
        top_scores = np.empty((len(block), 0), dtype=q_emb.dtype)
        top_ids = np.empty((len(block), 0), dtype=np.int64)
        for doc_start in range(0, len(corpus), corpus_block_size):
            sims = q_emb @ doc_emb[doc_start : doc_start + corpus_block_size].T
            ids = np.broadcast_to(np.arange(doc_start, doc_start + sims.shape[1]), sims.shape)
            scores = np.concatenate((top_scores, sims), axis=1)
            indices = np.concatenate((top_ids, ids), axis=1)
            order = np.lexsort((indices, -scores), axis=1)[:, :fetch]
            top_scores = np.take_along_axis(scores, order, axis=1)
            top_ids = np.take_along_axis(indices, order, axis=1)
        for i, ex in enumerate(block):
            negs: list[str] = []
            existing = set(ex.negatives)
            for rank, (j, score) in enumerate(zip(top_ids[i], top_scores[i], strict=True)):
                if rank < skip_top:
                    continue
                cand = corpus[int(j)]
                if cand == ex.positive or cand in existing:
                    continue
                if labelled and labels_by_text[cand] == ex.label:
                    continue
                if max_relative_score is not None and score > max_relative_score * pos_scores[i]:
                    continue
                negs.append(cand)
                if len(negs) == num_negatives:
                    break
            mined.append(
                ContrastiveExample(ex.query, ex.positive, [*ex.negatives, *negs], ex.instruction, ex.score, ex.label)
            )
    return mined
