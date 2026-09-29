"""Standard IR and STS metrics in pure Python/numpy (definitions match trec_eval/BEIR).

* ``ranked``: document ids in rank order (best first).
* ``relevant``: ``{doc_id: graded_relevance}``; ids with relevance <= 0 count as non-relevant.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

Qrels = Mapping[str, Mapping[str, float]]
Run = Mapping[str, Sequence[str]]


def _positive(relevant: Mapping[str, float]) -> set[str]:
    return {d for d, r in relevant.items() if r > 0}


def recall_at_k(ranked: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    rel = _positive(relevant)
    if not rel:
        return 0.0
    return len(rel.intersection(ranked[:k])) / len(rel)


def precision_at_k(ranked: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    rel = _positive(relevant)
    return sum(1 for d in ranked[:k] if d in rel) / k


def hit_rate_at_k(ranked: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    rel = _positive(relevant)
    return 1.0 if any(d in rel for d in ranked[:k]) else 0.0


def mrr_at_k(ranked: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    rel = _positive(relevant)
    for rank, d in enumerate(ranked[:k], start=1):
        if d in rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    """nDCG with linear gains and log2 discount (as in trec_eval / BEIR)."""
    dcg = sum(max(relevant.get(d, 0.0), 0.0) / math.log2(rank + 1) for rank, d in enumerate(ranked[:k], start=1))
    ideal = sorted((r for r in relevant.values() if r > 0), reverse=True)[:k]
    idcg = sum(r / math.log2(rank + 1) for rank, r in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(ranked: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    rel = _positive(relevant)
    if not rel:
        return 0.0
    hits, total = 0, 0.0
    for rank, d in enumerate(ranked[:k], start=1):
        if d in rel:
            hits += 1
            total += hits / rank
    return total / len(rel)  # trec_eval `map_cut` normalises by all relevant docs


_METRICS = {
    "recall": recall_at_k,
    "precision": precision_at_k,
    "hit": hit_rate_at_k,
    "mrr": mrr_at_k,
    "ndcg": ndcg_at_k,
    "map": average_precision_at_k,
}


def evaluate_run(
    run: Run, qrels: Qrels, ks: Iterable[int] = (1, 5, 10), metrics: Iterable[str] = ("ndcg", "mrr", "recall", "map")
) -> dict[str, float]:
    """Macro-average metrics over the queries in ``qrels`` (missing runs score 0).

    Returns keys like ``"ndcg@10"``.
    """
    ks = sorted(set(ks))
    names = list(metrics)
    unknown = set(names) - set(_METRICS)
    if unknown:
        raise ValueError(f"unknown metrics {sorted(unknown)}; choose from {sorted(_METRICS)}")
    qids = [q for q, rel in qrels.items() if _positive(rel)]
    out: dict[str, float] = {}
    for name in names:
        fn = _METRICS[name]
        for k in ks:
            vals = [fn(list(run.get(q, [])), qrels[q], k) for q in qids]
            out[f"{name}@{k}"] = float(np.mean(vals)) if vals else 0.0
    return out


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks with ties (like scipy.stats.rankdata(method='average'))."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    a, b = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if a.shape != b.shape or a.size < 2:
        raise ValueError("pearson needs two equal-length sequences with >= 2 items")
    a, b = a - a.mean(), b - b.mean()
    denom = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom else 0.0


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson(
        _rankdata(np.asarray(x, dtype=np.float64)).tolist(), _rankdata(np.asarray(y, dtype=np.float64)).tolist()
    )
