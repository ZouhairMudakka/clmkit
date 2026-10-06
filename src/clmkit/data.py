"""Training data: examples, JSONL I/O, false-negative-aware batching, hard-negative mining.

JSONL schema (one object per line)::

    {"query": "...", "positive": "...", "negatives": ["...", "..."], "instruction": "...", "score": 0.8}

Aliases accepted: ``anchor``/``question`` for query, ``pos``/``document`` for positive,
``neg``/``hard_negatives`` for negatives. ``negatives``, ``instruction`` and ``score`` are optional.
"""

from __future__ import annotations

import json
import math
import random
from collections import deque
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

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(self.positive, str) or not self.positive.strip():
            raise ValueError("positive must be a non-empty string")
        if isinstance(self.negatives, str):
            self.negatives = [self.negatives]
        if not all(isinstance(n, str) for n in self.negatives):
            raise ValueError("negatives must be strings")
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
) -> Iterator[list[ContrastiveExample]]:
    """Yield batches for in-batch-negative training.

    With ``avoid_duplicates`` an example is deferred to a later batch if its query or
    positive text already appears in the current one: a duplicate positive would be
    treated as a *negative* for the other query - a guaranteed false negative.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    order = list(range(len(examples)))
    if shuffle:
        random.Random(seed).shuffle(order)  # noqa: S311 - data shuffling, not crypto  # nosec B311
    queue = deque(examples[i] for i in order)
    waiting: list[ContrastiveExample] = []  # deferred duplicates, retried first in the next batch

    while queue or waiting:
        batch: list[ContrastiveExample] = []
        seen: set[str] = set()

        def fits(ex: ContrastiveExample, seen: set[str] = seen) -> bool:
            return not (avoid_duplicates and (ex.query in seen or ex.positive in seen))

        leftovers = []
        for ex in waiting:
            if len(batch) < batch_size and fits(ex):
                batch.append(ex)
                seen.update((ex.query, ex.positive))
            else:
                leftovers.append(ex)
        waiting = leftovers
        while len(batch) < batch_size and queue:
            ex = queue.popleft()
            if fits(ex):
                batch.append(ex)
                seen.update((ex.query, ex.positive))
            else:
                waiting.append(ex)
        # `batch` is never empty here: with an empty `seen`, the first candidate always fits.
        if len(batch) == batch_size or not drop_last:
            yield batch
        elif not queue:
            return


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
) -> list[ContrastiveExample]:
    """Attach hard negatives retrieved from ``corpus`` with ``encoder``.

    Guards against false negatives (the #1 failure mode of naive mining):

    * the example's own positive is never used;
    * ``skip_top`` ignores the N most similar documents outright;
    * ``max_relative_score`` drops candidates scoring above ``ratio * sim(query, positive)``
      ("positive-aware" mining, NV-Retriever 2024).
    """
    if num_negatives < 0 or skip_top < 0:
        raise ValueError("num_negatives and skip_top must be >= 0")
    if num_negatives == 0:
        return [ContrastiveExample(e.query, e.positive, list(e.negatives), e.instruction, e.score) for e in examples]
    corpus = list(dict.fromkeys(corpus))  # dedupe, keep order
    if not corpus:
        raise ValueError("corpus is empty")
    doc_emb = encoder.encode(corpus, kind="document", batch_size=batch_size)
    q_emb = encoder.encode(
        [e.query for e in examples], kind="query", instruction=[e.instruction for e in examples], batch_size=batch_size
    )
    p_emb = encoder.encode([e.positive for e in examples], kind="document", batch_size=batch_size)
    pos_scores = (q_emb * p_emb).sum(axis=1)
    sims = q_emb @ doc_emb.T
    fetch = min(len(corpus), skip_top + num_negatives * 3 + 1)
    top = np.argsort(-sims, axis=1)[:, :fetch]
    mined = []
    for i, ex in enumerate(examples):
        negs: list[str] = []
        existing = set(ex.negatives)
        for rank, j in enumerate(top[i]):
            if rank < skip_top:
                continue
            cand = corpus[int(j)]
            if cand == ex.positive or cand in existing or cand in negs:
                continue
            if max_relative_score is not None and sims[i, j] > max_relative_score * pos_scores[i]:
                continue
            negs.append(cand)
            if len(negs) == num_negatives:
                break
        mined.append(ContrastiveExample(ex.query, ex.positive, [*ex.negatives, *negs], ex.instruction, ex.score))
    return mined
