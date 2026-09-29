"""Evaluation: IR metrics, STS correlation, and ready-made evaluators."""

from clmkit.eval.evaluator import RetrievalEvaluator, STSEvaluator, load_beir
from clmkit.eval.metrics import (
    average_precision_at_k,
    evaluate_run,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    pearson,
    precision_at_k,
    recall_at_k,
    spearman,
)

__all__ = [
    "RetrievalEvaluator",
    "STSEvaluator",
    "average_precision_at_k",
    "evaluate_run",
    "hit_rate_at_k",
    "load_beir",
    "mrr_at_k",
    "ndcg_at_k",
    "pearson",
    "precision_at_k",
    "recall_at_k",
    "spearman",
]
