from __future__ import annotations

import math

import numpy as np
import pytest

from clmkit import BM25Retriever, HashingEncoder, HybridRetriever
from clmkit.encoders.base import Encoder


class ControlledEncoder(Encoder):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[list[str]] = []

    @property
    def native_dim(self) -> int:
        return 2

    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        self.calls.append(texts)
        if "FAIL" in texts:
            raise RuntimeError("encoding failed")
        vectors = {"query": [1, 0], "semantic": [1, 0], "query query": [0, 1]}
        return np.asarray([vectors.get(text, [0.6, 0.8]) for text in texts], dtype=np.float32)


def test_bm25_hand_calculated_scores_and_global_statistics() -> None:
    retriever = BM25Retriever(k1=1.5, b=0.75)
    retriever.add(["apple apple pear", "pear", ""], ids=["a", "b", "c"], metadata=[{"owner": "a"}, {}, {}])
    # N=3, df(apple)=1, length(a)=3, average length=4/3, tf=2.
    expected = math.log(1 + 2.5 / 1.5) * 2 * 2.5 / (2 + 1.5 * (0.25 + 0.75 * 3 / (4 / 3)))
    hits = retriever.search("APPLE apple", k=3)
    assert [hit.id for hit in hits] == ["a"]
    assert hits[0].score == pytest.approx(expected)
    assert retriever.search("apple", filter={"owner": "a"})[0].score == pytest.approx(expected)


def test_rrf_hand_calculated_and_deduplicated() -> None:
    retriever = HybridRetriever(ControlledEncoder(), candidate_depth=2, rrf_k=10)
    retriever.add(["semantic", "query query", "other"], ids=["s", "l", "m"])
    # Dense shortlist s,m; lexical shortlist l. s and l tie at 1/11.
    hits = retriever.search("query", k=3)
    # k expands both shortlists to three: l also gets dense rank 3.
    assert [hit.id for hit in hits] == ["l", "s", "m"]
    assert hits[0].score == pytest.approx(1 / 11 + 1 / 13)
    assert hits[1].score == pytest.approx(1 / 11)
    assert hits[2].score == pytest.approx(1 / 12)
    assert len({hit.id for hit in hits}) == len(hits)
    limited = retriever.search("query", k=2)
    assert [hit.id for hit in limited] == ["l", "s"]
    assert limited[0].score == pytest.approx(1 / 11)


@pytest.mark.parametrize("hybrid", [False, True])
def test_constraints_sku_date_amount_owner_and_single_eligibility_pass(hybrid: bool) -> None:
    retriever = HybridRetriever(ControlledEncoder(), candidate_depth=1) if hybrid else BM25Retriever()
    retriever.add(
        ["SKU-X refund"] * 5,
        ids=["forbidden-owner", "wrong-sku", "old", "expensive", "allowed"],
        metadata=[
            {"owner": "bob", "sku": "SKU-X", "date": "2026-09-01", "amount": 99},
            {"owner": "alice", "sku": "SKU-Y", "date": "2026-09-01", "amount": 99},
            {"owner": "alice", "sku": "SKU-X", "date": "2025-09-01", "amount": 99},
            {"owner": "alice", "sku": "SKU-X", "date": "2026-09-01", "amount": 999},
            {"owner": "alice", "sku": "SKU-X", "date": "2026-09-01", "amount": 99},
        ],
    )
    calls = []

    def eligible(meta):
        calls.append(meta)
        return (
            meta["owner"] == "alice"
            and meta["sku"] == "SKU-X"
            and meta["date"] >= "2026-01-01"
            and meta["amount"] <= 100
        )

    batch = retriever.search(["SKU-X refund", "SKU-X"], k=4, filter=eligible)
    assert len(calls) == 5
    assert [[hit.id for hit in row] for row in batch] == [["allowed"], ["allowed"]]
    assert retriever.search("SKU-X", filter={"owner": "nobody"}) == []


@pytest.mark.parametrize("hybrid", [False, True])
def test_ties_do_not_depend_on_insertion_order(hybrid: bool) -> None:
    def ranked(ids):
        retriever = HybridRetriever(ControlledEncoder(), candidate_depth=1) if hybrid else BM25Retriever()
        retriever.add(["same"] * 3, ids=ids)
        return [hit.id for hit in retriever.search("same", k=2)]

    assert ranked(["z", "a", "m"]) == ranked(["m", "z", "a"]) == ["a", "m"]


@pytest.mark.parametrize("hybrid", [False, True])
def test_lifecycle_refreshes_both_sources_and_copies_metadata(hybrid: bool) -> None:
    retriever = HybridRetriever(HashingEncoder(dim=32)) if hybrid else BM25Retriever()
    meta = {"owner": "alice", "nested": [1]}
    retriever.add(["apple", "pear"], ids=["a", "b"], metadata=[meta, {"owner": "bob"}])
    meta["nested"].append(2)
    document = retriever.get("a")
    document.text = "unindexed mutation"
    document.metadata["nested"].append(3)
    hit = retriever.search("apple", filter={"owner": "alice"})[0]
    hit.metadata["nested"].append(4)
    assert retriever.get("a").text == "apple"
    assert retriever.get("a").metadata["nested"] == [1]
    retriever.update("a", "pear", metadata={"owner": "bob"})
    assert retriever.search("apple", filter={"owner": "alice"}) == []
    assert {hit.id for hit in retriever.search("pear", k=2, filter={"owner": "bob"})} == {"a", "b"}
    retriever.update("a", "pear pear")
    assert retriever.get("a").metadata == {"owner": "bob"}
    assert retriever.delete(["a", "a", "missing"]) == 1
    assert "a" not in retriever and retriever.get("a") is None
    assert [hit.id for hit in retriever.search("pear", k=5)] == ["b"]
    assert retriever.delete(["b"]) == 1
    assert retriever.search("pear") == []
    retriever.add(["apple"], ids=["a"])
    assert retriever.search("apple")[0].id == "a"


def test_bm25_delete_recomputes_document_frequency() -> None:
    retriever = BM25Retriever()
    retriever.add(["apple", "apple", "pear"], ids=["a", "b", "c"])
    retriever.delete(["b", "c"])
    assert retriever.search("apple")[0].score == pytest.approx(math.log(1 + 0.5 / 1.5))


def test_hybrid_failed_mutations_keep_previous_complete_corpus() -> None:
    encoder = ControlledEncoder()
    retriever = HybridRetriever(encoder)
    retriever.add(["semantic", "query query"], ids=["s", "l"])
    before = retriever.search("query", k=2)
    with pytest.raises(RuntimeError, match="encoding failed"):
        retriever.update("s", "FAIL")
    with pytest.raises(RuntimeError, match="encoding failed"):
        retriever.add(["FAIL"], ids=["new"])
    assert retriever.search("query", k=2) == before
    assert "new" not in retriever
    retriever.update("s", "other")
    assert encoder.calls[-1] == ["other"]  # unchanged l was not re-encoded
    encoder.query_template = "changed {text}"
    with pytest.raises(ValueError, match="rebuild"):
        retriever.search("query")


@pytest.mark.parametrize("hybrid", [False, True])
def test_empty_batch_validation_and_failed_adds(hybrid: bool) -> None:
    retriever = HybridRetriever(HashingEncoder(dim=32)) if hybrid else BM25Retriever()
    assert retriever.search([]) == []
    assert retriever.search(["x", "y"]) == [[], []]
    assert retriever.add([]) == []
    retriever.add([""], ids=["empty"])
    assert retriever.search([]) == []
    for text, ids in [(["x", "y"], ["dup", "dup"]), (["x"], ["empty"]), (["x"], [])]:
        with pytest.raises(ValueError):
            retriever.add(text, ids=ids)
    with pytest.raises(TypeError):
        retriever.add("single")
    with pytest.raises(TypeError):
        retriever.add([None])
    with pytest.raises(TypeError):
        retriever.add(["x"], ids=[1])
    with pytest.raises(TypeError):
        retriever.add(["x"], metadata=[{"bad": object()}])
    with pytest.raises(KeyError):
        retriever.update("missing", "x")
    for k in [0, -1, True, 1.5]:
        with pytest.raises(ValueError, match="k"):
            retriever.search("x", k=k)
    with pytest.raises(TypeError):
        retriever.search([None])
    assert len(retriever) == 1


def test_lexical_empty_and_unknown_terms() -> None:
    retriever = BM25Retriever()
    retriever.add(["", "known"], ids=["blank", "text"])
    assert retriever.search(["", "???", "unknown"]) == [[], [], []]


@pytest.mark.parametrize("kwargs", [{"k1": 0}, {"k1": float("inf")}, {"b": -1}, {"b": float("nan")}])
def test_invalid_bm25_parameters(kwargs) -> None:
    with pytest.raises(ValueError):
        BM25Retriever(**kwargs)


@pytest.mark.parametrize("kwargs", [{"candidate_depth": 0}, {"rrf_k": -1}, {"rrf_k": float("nan")}])
def test_invalid_fusion_parameters(kwargs) -> None:
    with pytest.raises(ValueError):
        HybridRetriever(HashingEncoder(dim=32), **kwargs)
