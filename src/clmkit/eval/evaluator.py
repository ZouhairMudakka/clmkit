"""Evaluators: callables ``encoder -> {metric: value}`` usable standalone or inside training."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from clmkit.eval.metrics import evaluate_run, pearson, spearman
from clmkit.index.numpy_index import NumpyIndex

if TYPE_CHECKING:
    from clmkit.data import ContrastiveExample
    from clmkit.encoders.base import Encoder


class RetrievalEvaluator:
    """Encode a corpus, retrieve for each query, score against qrels.

    Args:
        queries: ``{qid: text}``.
        corpus: ``{doc_id: text}``.
        qrels: ``{qid: {doc_id: relevance}}``.
        ks: cut-offs.
        instruction: shared query instruction, or a mapping from query ID to instruction.
    """

    def __init__(
        self,
        queries: Mapping[str, str],
        corpus: Mapping[str, str],
        qrels: Mapping[str, Mapping[str, float]],
        *,
        ks: Iterable[int] = (1, 5, 10),
        metrics: Iterable[str] = ("ndcg", "mrr", "recall", "map"),
        instruction: str | Mapping[str, str | None] | None = None,
        batch_size: int = 32,
    ) -> None:
        if not queries or not corpus:
            raise ValueError("queries and corpus must be non-empty")
        self.queries = dict(queries)
        self.corpus = dict(corpus)
        self.qrels = {q: dict(r) for q, r in qrels.items()}
        self.ks = sorted(set(ks))
        if not self.ks or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in self.ks):
            raise ValueError("ks must contain positive integer cut-offs")
        self.metrics = tuple(metrics)
        self.instruction = instruction
        self.batch_size = batch_size

    @classmethod
    def from_examples(cls, examples: Sequence[ContrastiveExample], **kwargs: object) -> RetrievalEvaluator:
        """Positives are relevant; the corpus is all positives + negatives (deduplicated)."""
        doc_ids: dict[str, str] = {}
        queries: dict[str, str] = {}
        qrels: dict[str, dict[str, float]] = {}
        instructions: dict[str, str | None] = {}
        for i, ex in enumerate(examples):
            for text in (ex.positive, *ex.negatives):
                doc_ids.setdefault(text, f"d{len(doc_ids)}")
            qid = f"q{i}"
            queries[qid] = ex.query
            instructions[qid] = ex.instruction
            qrels[qid] = {doc_ids[ex.positive]: 1.0}
        corpus = {did: text for text, did in doc_ids.items()}
        if "instruction" not in kwargs:
            kwargs["instruction"] = instructions
        return cls(queries, corpus, qrels, **kwargs)  # type: ignore[arg-type]

    def run(self, encoder: Encoder) -> dict[str, list[str]]:
        doc_ids = list(self.corpus)
        index = NumpyIndex(encoder.dim)
        index.add(
            doc_ids, encoder.encode([self.corpus[d] for d in doc_ids], kind="document", batch_size=self.batch_size)
        )
        qids = list(self.queries)
        instruction = (
            [self.instruction.get(q) for q in qids] if isinstance(self.instruction, Mapping) else self.instruction
        )
        qvecs = encoder.encode(
            [self.queries[q] for q in qids], kind="query", instruction=instruction, batch_size=self.batch_size
        )
        hits = index.search(qvecs, k=max(self.ks))
        return {q: [d for d, _ in row] for q, row in zip(qids, hits, strict=True)}

    def __call__(self, encoder: Encoder) -> dict[str, float]:
        return evaluate_run(self.run(encoder), self.qrels, self.ks, self.metrics)


class STSEvaluator:
    """Correlation between cosine similarity and gold similarity scores."""

    def __init__(self, pairs: Sequence[tuple[str, str]], scores: Sequence[float], *, batch_size: int = 32) -> None:
        if len(pairs) != len(scores) or len(pairs) < 2:
            raise ValueError("need >= 2 pairs with one score each")
        self.pairs = list(pairs)
        self.scores = [float(s) for s in scores]
        self.batch_size = batch_size

    def __call__(self, encoder: Encoder) -> dict[str, float]:
        a = encoder.encode([p[0] for p in self.pairs], batch_size=self.batch_size)
        b = encoder.encode([p[1] for p in self.pairs], batch_size=self.batch_size)
        cos = np.sum(a * b, axis=1).tolist()
        return {"spearman": spearman(cos, self.scores), "pearson": pearson(cos, self.scores)}


def load_beir(
    path: str | Path, split: str = "test"
) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, float]]]:
    """Load a BEIR-format dataset directory: ``corpus.jsonl``, ``queries.jsonl``, ``qrels/<split>.tsv``.

    Returns ``(queries, corpus, qrels)``; titles are prepended to document text.
    """
    root = Path(path)
    corpus: dict[str, str] = {}
    with (root / "corpus.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                title = (rec.get("title") or "").strip()
                corpus[str(rec["_id"])] = f"{title}\n{rec['text']}" if title else rec["text"]
    queries: dict[str, str] = {}
    with (root / "queries.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                queries[str(rec["_id"])] = rec["text"]
    qrels: dict[str, dict[str, float]] = {}
    with (root / "qrels" / f"{split}.tsv").open(encoding="utf-8", newline="") as fh:
        for lineno, row in enumerate(csv.reader(fh, delimiter="\t"), start=1):
            if len(row) < 3:
                continue
            try:
                score = float(row[2])
            except ValueError:
                if lineno == 1:  # "query-id  corpus-id  score" header row
                    continue
                raise ValueError(f"{root}/qrels/{split}.tsv:{lineno}: bad score {row[2]!r}") from None
            qrels.setdefault(row[0], {})[row[1]] = score
    queries = {q: t for q, t in queries.items() if q in qrels}
    return queries, corpus, qrels
