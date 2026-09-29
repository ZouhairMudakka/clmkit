from __future__ import annotations

import json
import math

import numpy as np
import pytest

from clmkit import HashingEncoder
from clmkit.data import (
    ContrastiveExample,
    iter_batches,
    load_examples,
    mine_hard_negatives,
    save_examples,
    split_examples,
)
from clmkit.eval import (
    RetrievalEvaluator,
    STSEvaluator,
    average_precision_at_k,
    evaluate_run,
    hit_rate_at_k,
    load_beir,
    mrr_at_k,
    ndcg_at_k,
    pearson,
    precision_at_k,
    recall_at_k,
    spearman,
)

# ------------------------------------------------------------------ metrics --
RANKED = ["a", "b", "c", "d", "e"]
REL = {"b": 1.0, "d": 1.0, "z": 1.0}


def test_binary_metrics_hand_computed() -> None:
    assert recall_at_k(RANKED, REL, 2) == pytest.approx(1 / 3)
    assert recall_at_k(RANKED, REL, 5) == pytest.approx(2 / 3)
    assert precision_at_k(RANKED, REL, 4) == pytest.approx(0.5)
    assert hit_rate_at_k(RANKED, REL, 1) == 0.0 and hit_rate_at_k(RANKED, REL, 2) == 1.0
    assert mrr_at_k(RANKED, REL, 5) == pytest.approx(0.5)
    assert mrr_at_k(RANKED, REL, 1) == 0.0
    # AP@5: hits at ranks 2 and 4 -> (1/2 + 2/4) / 3 relevant
    assert average_precision_at_k(RANKED, REL, 5) == pytest.approx((0.5 + 0.5) / 3)


def test_ndcg_graded() -> None:
    rel = {"a": 3.0, "b": 2.0, "c": 0.0}
    assert ndcg_at_k(["a", "b", "c"], rel, 3) == pytest.approx(1.0)
    dcg = 2 / math.log2(2) + 3 / math.log2(3)
    idcg = 3 / math.log2(2) + 2 / math.log2(3)
    assert ndcg_at_k(["b", "a"], rel, 2) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["x"], {}, 1) == 0.0


def test_evaluate_run_macro_average() -> None:
    run = {"q1": ["a", "b"], "q2": ["x", "y"]}
    qrels = {"q1": {"a": 1}, "q2": {"y": 1}, "q3": {"m": 1}, "q4": {"n": 0}}
    out = evaluate_run(run, qrels, ks=[1, 2], metrics=["recall", "mrr"])
    # q4 has no positive judgements -> excluded; q3 missing from run -> scores 0
    assert out["recall@1"] == pytest.approx(1 / 3)
    assert out["mrr@2"] == pytest.approx((1 + 0.5 + 0) / 3)
    with pytest.raises(ValueError, match="unknown metrics"):
        evaluate_run(run, qrels, metrics=["bleu"])


def test_correlations_match_scipy() -> None:
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(1)
    x = rng.normal(size=50)
    y = x + rng.normal(size=50)
    y[3] = y[4]  # ties
    assert spearman(x, y) == pytest.approx(scipy_stats.spearmanr(x, y).statistic)
    assert pearson(x, y) == pytest.approx(scipy_stats.pearsonr(x, y).statistic)
    with pytest.raises(ValueError):
        pearson([1.0], [1.0])


# --------------------------------------------------------------------- data --
def test_example_validation_and_aliases() -> None:
    ex = ContrastiveExample.from_dict({"anchor": "q", "pos": "p", "neg": "n", "score": "0.5"})
    assert ex.negatives == ["n"] and ex.score == 0.5
    assert ex.to_dict() == {"query": "q", "positive": "p", "negatives": ["n"], "score": 0.5}
    with pytest.raises(ValueError, match="query"):
        ContrastiveExample("  ", "p")
    with pytest.raises(ValueError, match="positive"):
        ContrastiveExample("q", "")
    with pytest.raises(ValueError, match="negatives"):
        ContrastiveExample("q", "p", [1])  # type: ignore[list-item]


def test_jsonl_roundtrip_and_errors(tmp_path, toy_examples) -> None:  # type: ignore[no-untyped-def]
    path = save_examples(tmp_path / "sub" / "train.jsonl", toy_examples)
    assert load_examples(path) == toy_examples
    (tmp_path / "list.json").write_text(json.dumps([{"query": "q", "positive": "p"}]))
    assert len(load_examples(tmp_path / "list.json")) == 1
    (tmp_path / "bad.jsonl").write_text('{"query": "q", "positive": "p"}\n{oops\n')
    with pytest.raises(ValueError, match=r"bad.jsonl:2: invalid JSON"):
        load_examples(tmp_path / "bad.jsonl")
    (tmp_path / "missing.jsonl").write_text('{"query": "q"}\n')
    with pytest.raises(ValueError, match=r"missing.jsonl:1: positive"):
        load_examples(tmp_path / "missing.jsonl")
    (tmp_path / "notobj.jsonl").write_text("[1, 2]\n")
    with pytest.raises(ValueError, match="expected an object"):
        load_examples(tmp_path / "notobj.jsonl")


def test_iter_batches_avoids_in_batch_duplicates() -> None:
    exs = [ContrastiveExample(f"q{i}", f"p{i % 3}") for i in range(9)]  # only 3 distinct positives
    batches = list(iter_batches(exs, batch_size=3, seed=0))
    assert sum(len(b) for b in batches) == 9
    for b in batches:
        assert len({e.positive for e in b}) == len(b)
    assert [len(b) for b in iter_batches(exs, 4, avoid_duplicates=False, shuffle=False)] == [4, 4, 1]
    assert [len(b) for b in iter_batches(exs, 4, avoid_duplicates=False, drop_last=True)] == [4, 4]
    with pytest.raises(ValueError):
        next(iter_batches(exs, 0))


def test_iter_batches_is_linear_time_on_large_input() -> None:
    exs = [ContrastiveExample(f"q{i}", f"p{i}") for i in range(20_000)]
    assert sum(len(b) for b in iter_batches(exs, 64)) == 20_000


def test_split(toy_examples) -> None:  # type: ignore[no-untyped-def]
    train, ev = split_examples(toy_examples, 0.25, seed=3)
    assert len(ev) == 5 and len(train) == 15
    assert {e.query for e in train}.isdisjoint({e.query for e in ev})
    with pytest.raises(ValueError):
        split_examples(toy_examples, 1.5)


def test_mine_hard_negatives_excludes_positive_and_false_negatives(toy_examples) -> None:  # type: ignore[no-untyped-def]
    enc = HashingEncoder(dim=512)
    corpus = [e.positive for e in toy_examples]
    mined = mine_hard_negatives(enc, toy_examples, corpus, num_negatives=3, max_relative_score=None)
    for ex in mined:
        assert len(ex.negatives) == 3 and ex.positive not in ex.negatives
        assert len(set(ex.negatives)) == 3
    # a near-duplicate of the positive is a likely false negative and must be dropped
    ex = ContrastiveExample("what do cats eat", "cats eat fish and meat")
    near_dup = "Cats eat fish, and meat!"
    kept = mine_hard_negatives(enc, [ex], [near_dup, "the sun is hot"], num_negatives=2)[0]
    assert near_dup not in kept.negatives and kept.negatives == ["the sun is hot"]
    naive = mine_hard_negatives(enc, [ex], [near_dup, "the sun is hot"], num_negatives=2, max_relative_score=None)[0]
    assert near_dup in naive.negatives
    with pytest.raises(ValueError, match="empty"):
        mine_hard_negatives(enc, [ex], [])


# ---------------------------------------------------------------- evaluators --
def test_retrieval_evaluator_from_examples(toy_examples) -> None:  # type: ignore[no-untyped-def]
    ev = RetrievalEvaluator.from_examples(toy_examples, ks=[1, 10])
    metrics = ev(HashingEncoder(dim=1024))
    assert set(metrics) == {f"{m}@{k}" for m in ("ndcg", "mrr", "recall", "map") for k in (1, 10)}
    assert metrics["recall@10"] >= metrics["recall@1"] > 0.3
    with pytest.raises(ValueError):
        RetrievalEvaluator({}, {"d": "x"}, {})


def test_sts_evaluator() -> None:
    pairs = [("a cat sat", "a cat sat down"), ("a cat sat", "stock markets fell"), ("the sun", "the sun is hot")]
    out = STSEvaluator(pairs, [0.9, 0.0, 0.7])(HashingEncoder(dim=512))
    assert out["spearman"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        STSEvaluator(pairs, [1.0])


def test_load_beir(tmp_path) -> None:
    (tmp_path / "qrels").mkdir()
    (tmp_path / "corpus.jsonl").write_text(
        json.dumps({"_id": "d1", "title": "Cats", "text": "cats eat fish"})
        + "\n"
        + json.dumps({"_id": "d2", "title": "", "text": "the sun is hot"})
        + "\n"
    )
    (tmp_path / "queries.jsonl").write_text(
        json.dumps({"_id": "q1", "text": "what do cats eat"}) + "\n" + json.dumps({"_id": "q9", "text": "unjudged"})
    )
    (tmp_path / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n")
    queries, corpus, qrels = load_beir(tmp_path)
    assert queries == {"q1": "what do cats eat"}
    assert corpus["d1"] == "Cats\ncats eat fish" and corpus["d2"] == "the sun is hot"
    assert qrels == {"q1": {"d1": 1.0}}
    (tmp_path / "qrels" / "dev.tsv").write_text("q1\td2\t1\n")  # headerless
    assert load_beir(tmp_path, "dev")[2] == {"q1": {"d2": 1.0}}
    (tmp_path / "qrels" / "bad.tsv").write_text("q-id\tc-id\tscore\nq1\td2\thigh\n")
    with pytest.raises(ValueError, match="bad score"):
        load_beir(tmp_path, "bad")
    metrics = RetrievalEvaluator(queries, corpus, qrels, ks=[1])(HashingEncoder(dim=256))
    assert metrics["recall@1"] == 1.0
