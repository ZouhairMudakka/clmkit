"""Independent hand-calculated contracts for the public evidence protocol."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))

from evidence_data import grouped_split
from evidence_metrics import (
    paired_interval,
    ranking_metrics,
    rejection_metrics,
    select_rejection_threshold,
    validate_predictions,
)


def _hits(*items):
    return [{"id": did, "score": score} for did, score in items]


def test_ranking_hand_calculation_and_separate_no_positive_denominator():
    queries = [{"id": qid, "label": label} for qid, label in [("q1", "A"), ("q2", "A"), ("q3", "B"), ("q4", "OOS")]]
    qrels = {"q1": {"a": 3, "b": 1}, "q2": {"c": 1}, "q3": {"a": 1}, "q4": {"x": 0}}
    predictions = {"q1": _hits(("x", 1), ("b", 0.9), ("a", 0.8)), "q2": [], "q3": _hits(("a", 1)), "q4": []}
    result = ranking_metrics(queries, qrels, predictions, {"a", "b", "c", "x"})
    assert (result["total_queries"], result["rankable_queries"], result["no_positive_queries"]) == (4, 3, 1)
    assert result["metrics"]["hit@1"] == pytest.approx(1 / 3)
    assert result["metrics"]["hit@5"] == pytest.approx(2 / 3)
    assert result["metrics"]["mrr@10"] == pytest.approx((1 / 2 + 0 + 1) / 3)
    expected_q1_ndcg = (1 / math.log2(3) + 3 / 2) / (3 + 1 / math.log2(3))
    assert result["metrics"]["ndcg@10"] == pytest.approx((expected_q1_ndcg + 1) / 3)
    assert result["metrics"]["recall@10"] == pytest.approx(2 / 3)
    assert result["metrics"]["recall@100"] == pytest.approx(2 / 3)
    assert result["macro_intent_hit@1"] == 0.5
    assert result["per_intent_hit@1"] == {"A": 0.0, "B": 1.0}
    assert "q4" not in result["per_query"]


def test_hit_and_recall_are_not_interchangeable_for_intent_gallery():
    result = ranking_metrics([{"id": "q"}], {"q": {"a": 1, "b": 1, "c": 1}}, {"q": _hits(("a", 1))}, {"a", "b", "c"})
    assert result["metrics"]["hit@1"] == 1
    assert result["metrics"]["recall@10"] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    ("queries", "predictions", "message"),
    [
        ([{"id": "q"}], {}, "cover every query"),
        ([{"id": "q"}], {"q": [], "extra": []}, "cover every query"),
        ([{"id": "q"}, {"id": "q"}], {"q": []}, "cover every query"),
        ([{}], {}, "string id"),
        ([{"id": 2}], {2: []}, "string id"),
        ([{"id": "q"}], {"q": _hits(("a", 1), ("a", 0))}, "Invalid/duplicate"),
        ([{"id": "q"}], {"q": _hits(("absent", 1))}, "Invalid/duplicate"),
        ([{"id": "q"}], {"q": _hits(("a", float("nan")))}, "Invalid/duplicate"),
        ([{"id": "q"}], {"q": [{"id": "a"}]}, "Invalid/duplicate"),
    ],
)
def test_prediction_contract_rejects_malformed_coverage(queries, predictions, message):
    with pytest.raises(ValueError, match=message):
        validate_predictions(queries, predictions, {"a"})


@pytest.mark.parametrize(
    ("qrels", "message"),
    [
        ({"other": {"a": 1}}, "unknown query"),
        ({"q": {"missing": 1}}, "missing from the corpus"),
        ({"q": {"a": float("nan")}}, "finite"),
        ({"q": {"a": float("inf")}}, "finite"),
        ({"q": {"a": "1"}}, "finite"),
    ],
)
def test_relevance_contract_validates_even_unrankable_queries(qrels, message):
    with pytest.raises(ValueError, match=message):
        ranking_metrics([{"id": "q"}], qrels, {"q": []}, {"a"})


def _rejection_fixture():
    queries = [{"id": f"i{i}", "label": "A", "in_scope": True} for i in range(4)]
    queries += [{"id": f"o{i}", "in_scope": False} for i in range(3)]
    predictions = {
        "i0": _hits(("a", 0.9)),
        "i1": _hits(("b", 0.8)),
        "i2": _hits(("a", 0.7)),
        "i3": [],
        "o0": _hits(("a", 0.8)),
        "o1": _hits(("b", 0.6)),
        "o2": [],
    }
    return queries, predictions, {"a": "A", "b": "B"}


def test_rejection_denominators_include_missing_hits_and_keep_boundary_ties():
    result = rejection_metrics(*_rejection_fixture(), 0.8)
    assert result["in_scope_coverage"] == 1 / 2
    assert result["out_of_scope_false_acceptance"] == 1 / 3
    assert result["out_of_scope_rejection_recall"] == pytest.approx(2 / 3)
    assert result["accepted_in_scope_accuracy"] == 1 / 2
    result = rejection_metrics(*_rejection_fixture(), 1)
    assert result["accepted_in_scope"] == result["accepted_out_of_scope"] == 0
    assert result["accepted_in_scope_accuracy"] is None


@pytest.mark.parametrize(("target", "coverage", "false_acceptance"), [(0, 1 / 4, 0), (1 / 3, 3 / 4, 1 / 3)])
def test_threshold_selection_matches_hand_optimum(target, coverage, false_acceptance):
    result = select_rejection_threshold(*_rejection_fixture(), max_false_acceptance=target)
    assert result["selection_split"] == "dev"
    assert result["selected"]["in_scope_coverage"] == coverage
    assert result["selected"]["out_of_scope_false_acceptance"] == false_acceptance
    for row in result["curve"]:
        independently_counted = rejection_metrics(*_rejection_fixture(), row["threshold"])
        assert row["in_scope_coverage"] == independently_counted["in_scope_coverage"]
        assert row["out_of_scope_false_acceptance"] == independently_counted["out_of_scope_false_acceptance"]


def test_threshold_all_reject_when_class_scores_tie():
    queries = [{"id": "i", "in_scope": True, "label": "A"}, {"id": "o", "in_scope": False}]
    predictions = {qid: _hits(("a", 0.5)) for qid in ("i", "o")}
    result = select_rejection_threshold(queries, predictions, {"a": "A"})["selected"]
    assert result["threshold"] > 0.5
    assert result["in_scope_coverage"] == result["out_of_scope_false_acceptance"] == 0
    assert result["accepted_in_scope_accuracy"] is None


def test_threshold_unrepresentable_all_reject_fails_without_infinite_json_values():
    queries = [{"id": "i", "in_scope": True, "label": "A"}, {"id": "o", "in_scope": False}]
    predictions = {qid: _hits(("a", sys.float_info.max)) for qid in ("i", "o")}
    with pytest.raises(ValueError, match="No finite threshold"):
        select_rejection_threshold(queries, predictions, {"a": "A"})


def test_rejection_requires_both_classes_and_in_scope_intent_labels():
    queries, predictions, labels = _rejection_fixture()
    del queries[0]["label"]
    with pytest.raises(ValueError, match="string label"):
        rejection_metrics(queries, predictions, labels, 0.8)
    queries = [{"id": "o", "in_scope": False}]
    with pytest.raises(ValueError, match="both query classes"):
        select_rejection_threshold(queries, {"o": _hits(("a", 0.5))}, labels)


@pytest.mark.parametrize("in_scope", [None, 1, "false"])
def test_threshold_selection_rejects_nonboolean_scope(in_scope):
    queries, predictions, labels = _rejection_fixture()
    queries[0]["in_scope"] = in_scope
    with pytest.raises(ValueError, match="explicit boolean"):
        select_rejection_threshold(queries, predictions, labels)


def test_no_positive_ranking_and_empty_rejection_predictions_remain_explicit():
    queries, _, labels = _rejection_fixture()
    predictions = {q["id"]: [] for q in queries}
    ranking = ranking_metrics(queries, {}, predictions, set(labels))
    assert ranking["rankable_queries"] == 0
    assert ranking["no_positive_queries"] == 7
    assert ranking["metrics"] == {}
    assert rejection_metrics(queries, predictions, labels, 0)["out_of_scope_false_acceptance"] == 0
    with pytest.raises(ValueError, match="entirely empty"):
        select_rejection_threshold(queries, predictions, labels)


def test_paired_interval_resamples_groups_and_averages_fixed_seeds():
    baseline = dict.fromkeys("abcd", 0.5)
    runs = [{"a": 1, "b": 1, "c": 1, "d": 0}, dict(baseline)]
    result = paired_interval(baseline, runs, {"a": "g1", "b": "g1", "c": "g1", "d": "g2"})
    # Group draws: g1,g1 -> +.25; g2,g2 -> -.25; mixed -> +.125.
    assert result["mean_difference"] == 0.125
    assert result["ci95"] == [-0.25, 0.25]
    assert result["seed_differences"] == [0.25, 0.0]
    assert result["groups"] == 2 and result["queries"] == 4
    assert result["conditional_on_trained_seeds"] is True
    assert result == paired_interval(baseline, runs, {"a": "g1", "b": "g1", "c": "g1", "d": "g2"})
    single_group = paired_interval(baseline, runs, dict.fromkeys(baseline, "one"), samples=20)
    assert single_group["ci95"] == [0.125, 0.125]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"adapted_runs": [{"other": 1}]},
        {"groups": {}},
        {"samples": 0},
        {"samples": 1.5},
        {"samples": True},
        {"adapted_runs": [{"q": float("nan")}]},
    ],
)
def test_paired_interval_rejects_nonpaired_or_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        paired_interval(**{"baseline": {"q": 0}, "adapted_runs": [{"q": 1}], **kwargs})


def test_grouped_split_preserves_strata_and_whole_duplicate_groups():
    records = [
        {"id": f"{label}-{group}-{item}", "label": label, "text": f"{label} group {group}"}
        for label in ("A", "B")
        for group in range(4)
        for item in range(2)
    ]
    train, dev = grouped_split(records, dev_fraction=0.25)
    assert {row["label"] for row in train} == {row["label"] for row in dev} == {"A", "B"}
    assert len(train) == 12 and len(dev) == 4
    assert {row["text"] for row in train}.isdisjoint(row["text"] for row in dev)
    permuted_train, permuted_dev = grouped_split(records[::-1], dev_fraction=0.25)
    assert {row["id"] for row in train} == {row["id"] for row in permuted_train}
    assert {row["id"] for row in dev} == {row["id"] for row in permuted_dev}
