"""Explicit query coverage, intent-proxy retrieval and rejection metrics.

These helpers deliberately keep no-positive queries in the rejection denominator.
Ranking quality and abstention are different measurements, not interchangeable scores.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import numpy as np

from clmkit.eval.metrics import hit_rate_at_k, mrr_at_k, ndcg_at_k, recall_at_k


def _finite_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def validate_predictions(
    queries: Sequence[dict], predictions: Mapping[str, Sequence[dict]], corpus_ids: set[str]
) -> None:
    ids = [q.get("id") for q in queries]
    if any(not isinstance(qid, str) or not qid for qid in ids):
        raise ValueError("Every query needs a non-empty string id")
    if any(not isinstance(did, str) or not did for did in corpus_ids):
        raise ValueError("Corpus ids must be non-empty strings")
    if len(set(ids)) != len(ids) or set(predictions) != set(ids):
        raise ValueError("Predictions must cover every query exactly once, with no extra queries")
    for qid, hits in predictions.items():
        seen = set()
        for hit in hits:
            did = hit.get("id")
            if not isinstance(did, str) or did in seen or did not in corpus_ids or not _finite_number(hit.get("score")):
                raise ValueError(f"Invalid/duplicate document or nonfinite score for {qid}")
            seen.add(hit["id"])


def ranking_metrics(
    queries: Sequence[dict],
    qrels: Mapping[str, Mapping[str, float]],
    predictions: Mapping[str, Sequence[dict]],
    corpus_ids: set[str],
) -> dict[str, Any]:
    validate_predictions(queries, predictions, corpus_ids)
    if set(qrels) - {q["id"] for q in queries}:
        raise ValueError("Relevance judgments contain unknown query ids")
    rows = {}
    by_label: dict[str, list[float]] = defaultdict(list)
    for query in queries:
        qid = query["id"]
        relevant = qrels.get(qid, {})
        if any(not _finite_number(value) for value in relevant.values()):
            raise ValueError("Relevance scores must be finite numbers")
        if any(d not in corpus_ids for d in relevant):
            raise ValueError("Relevant document is missing from the corpus")
        if not any(value > 0 for value in relevant.values()):
            continue
        ranked = [h["id"] for h in predictions[qid]]
        row = {
            "hit@1": hit_rate_at_k(ranked, relevant, 1),
            "hit@5": hit_rate_at_k(ranked, relevant, 5),
            "mrr@10": mrr_at_k(ranked, relevant, 10),
            "ndcg@10": ndcg_at_k(ranked, relevant, 10),
            "recall@10": recall_at_k(ranked, relevant, 10),
            "recall@100": recall_at_k(ranked, relevant, 100),
        }
        rows[qid] = row
        if "label" in query:
            by_label[query["label"]].append(row["hit@1"])
    aggregate = {key: float(np.mean([r[key] for r in rows.values()])) for key in next(iter(rows.values()), {})}
    per_label = {key: float(np.mean(values)) for key, values in sorted(by_label.items())}
    return {
        "total_queries": len(queries),
        "rankable_queries": len(rows),
        "no_positive_queries": len(queries) - len(rows),
        "metrics": aggregate,
        "macro_intent_hit@1": float(np.mean(list(per_label.values()))) if per_label else None,
        "per_intent_hit@1": per_label,
        "per_query": rows,
    }


def _validate_rejection_labels(queries: Sequence[dict], corpus_labels: Mapping[str, str]) -> None:
    if any(not isinstance(label, str) or not label for label in corpus_labels.values()):
        raise ValueError("Corpus labels must be non-empty strings")
    for query in queries:
        if not isinstance(query.get("in_scope"), bool):
            raise ValueError("Every rejection query needs an explicit boolean in_scope label")
        if query["in_scope"] and (not isinstance(query.get("label"), str) or not query["label"]):
            raise ValueError("Every in-scope query needs a non-empty string label")


def rejection_metrics(
    queries: Sequence[dict],
    predictions: Mapping[str, Sequence[dict]],
    corpus_labels: Mapping[str, str],
    threshold: float,
) -> dict[str, Any]:
    validate_predictions(queries, predictions, set(corpus_labels))
    _validate_rejection_labels(queries, corpus_labels)
    if not _finite_number(threshold):
        raise ValueError("Threshold must be finite")
    inside = outside = accepted_inside = accepted_outside = correct_inside = 0
    for query in queries:
        hits = predictions[query["id"]]
        accepted = bool(hits and hits[0]["score"] >= threshold)
        if query["in_scope"]:
            inside += 1
            accepted_inside += accepted
            correct_inside += bool(accepted and corpus_labels[hits[0]["id"]] == query["label"])
        else:
            outside += 1
            accepted_outside += accepted
    if not inside or not outside:
        raise ValueError("Rejection evaluation requires both in-scope and out-of-scope queries")
    return {
        "threshold": threshold,
        "in_scope_queries": inside,
        "out_of_scope_queries": outside,
        "out_of_scope_false_acceptance": accepted_outside / outside,
        "out_of_scope_rejection_recall": 1 - accepted_outside / outside,
        "in_scope_coverage": accepted_inside / inside,
        "accepted_in_scope_accuracy": correct_inside / accepted_inside if accepted_inside else None,
        "accepted_in_scope": accepted_inside,
        "accepted_out_of_scope": accepted_outside,
    }


def select_rejection_threshold(
    queries: Sequence[dict],
    predictions: Mapping[str, Sequence[dict]],
    corpus_labels: Mapping[str, str],
    max_false_acceptance: float = 0.05,
) -> dict[str, Any]:
    """Choose using development data only; >= makes boundary ties explicit."""
    if not _finite_number(max_false_acceptance) or not 0 <= max_false_acceptance < 1:
        raise ValueError("max_false_acceptance must lie in [0, 1)")
    validate_predictions(queries, predictions, set(corpus_labels))
    _validate_rejection_labels(queries, corpus_labels)
    scores = sorted({hits[0]["score"] for hits in predictions.values() if hits})
    if not scores:
        raise ValueError("Cannot calibrate an entirely empty retrieval run")
    candidates = sorted(set(scores + [math.nextafter(float(s), math.inf) for s in scores]))
    candidates = [threshold for threshold in candidates if math.isfinite(threshold)]
    # O(n log n) threshold sweep; avoid repeatedly scanning all queries per threshold.
    inside = [predictions[q["id"]][0]["score"] for q in queries if q["in_scope"] and predictions[q["id"]]]
    outside = [predictions[q["id"]][0]["score"] for q in queries if not q["in_scope"] and predictions[q["id"]]]
    n_inside = sum(q["in_scope"] for q in queries)
    n_outside = len(queries) - n_inside
    if not n_inside or not n_outside:
        raise ValueError("Threshold selection requires both query classes")
    a, b = np.sort(inside), np.sort(outside)
    curve = []
    for threshold in candidates:
        coverage = (len(a) - int(np.searchsorted(a, threshold, side="left"))) / n_inside
        false_acceptance = (len(b) - int(np.searchsorted(b, threshold, side="left"))) / n_outside
        curve.append(
            {"threshold": threshold, "in_scope_coverage": coverage, "out_of_scope_false_acceptance": false_acceptance}
        )
    eligible = [r for r in curve if r["out_of_scope_false_acceptance"] <= max_false_acceptance]
    if not eligible:
        raise ValueError("No finite threshold satisfies the target; score range prevents representing all-reject")
    chosen = min(eligible, key=lambda r: (-r["in_scope_coverage"], r["out_of_scope_false_acceptance"], r["threshold"]))
    return {
        "selection_split": "dev",
        "target_false_acceptance": max_false_acceptance,
        "selected": rejection_metrics(queries, predictions, corpus_labels, chosen["threshold"]),
        "curve": curve,
    }


def paired_interval(
    baseline: Mapping[str, float],
    adapted_runs: Sequence[Mapping[str, float]],
    groups: Mapping[str, str] | None = None,
    *,
    seed: int = 2026,
    samples: int = 2000,
) -> dict:
    if not adapted_runs or not baseline or isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("Need baseline, adapted runs and positive bootstrap sample count")
    ids = sorted(baseline)
    if any(set(run) != set(ids) for run in adapted_runs):
        raise ValueError("Paired runs must cover the identical query IDs")
    if groups is not None and set(groups) != set(ids):
        raise ValueError("Group assignments must cover exactly the paired queries")
    deltas = np.array([[run[q] - baseline[q] for q in ids] for run in adapted_runs], dtype=float)
    if not np.isfinite(deltas).all():
        raise ValueError("Paired scores must be finite")
    per_query = deltas.mean(axis=0)
    members: dict[str, list[int]] = defaultdict(list)
    for i, qid in enumerate(ids):
        members[groups[qid] if groups else qid].append(i)
    units = list(members.values())
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        selected = np.concatenate([units[i] for i in rng.integers(0, len(units), len(units))])
        values.append(float(per_query[selected].mean()))
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "mean_difference": float(per_query.mean()),
        "ci95": [float(low), float(high)],
        "seed_differences": deltas.mean(axis=1).tolist(),
        "queries": len(ids),
        "groups": len(units),
        "bootstrap_samples": samples,
        "conditional_on_trained_seeds": True,
    }
