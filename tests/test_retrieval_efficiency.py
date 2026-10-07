"""Do not spend model work on requests with a known empty or invalid result."""

from __future__ import annotations

import pytest

from clmkit import HashingEncoder, Retriever


class CountingEncoder(HashingEncoder):
    def __init__(self):
        super().__init__(dim=64)
        self.calls = 0

    def _encode(self, texts, batch_size):
        self.calls += 1
        return super()._encode(texts, batch_size)


def test_empty_store_and_excluded_scope_skip_model_work():
    encoder = CountingEncoder()
    retriever = Retriever(encoder)
    assert retriever.search("refund") == []
    assert retriever.search(["refund", "delivery"]) == [[], []]
    assert encoder.calls == 0
    retriever.add(["refund policy"], ids=["policy"], metadata=[{"tenant": "north"}])
    encoder.calls = 0
    assert retriever.search("refund", filter={"tenant": "south"}) == []
    assert retriever.search(["refund", "delivery"], filter=lambda m: m["tenant"] == "south") == [[], []]
    assert encoder.calls == 0
    assert retriever.search("refund", filter={"tenant": "north"})[0].id == "policy"
    assert encoder.calls == 1


@pytest.mark.parametrize("ids", [["existing"], ["repeat", "repeat"], [""]])
def test_invalid_add_ids_fail_before_model_work(ids):
    encoder = CountingEncoder()
    retriever = Retriever(encoder)
    retriever.add(["original"], ids=["existing"])
    encoder.calls = 0
    with pytest.raises((TypeError, ValueError)):
        retriever.add(["unnecessary embedding"] * len(ids), ids=ids)
    assert encoder.calls == 0
    assert len(retriever) == len(retriever.index) == 1
    assert retriever.get("existing").text == "original"


def test_empty_fast_path_keeps_validation():
    retriever = Retriever(CountingEncoder())
    with pytest.raises(ValueError, match="k"):
        retriever.search("query", k=0)
    with pytest.raises(ValueError, match="k"):
        retriever.search("query", k=-1)
    with pytest.raises(TypeError, match="strings"):
        retriever.search([123])
    with pytest.raises(ValueError, match="no reranker"):
        retriever.search("query", rerank=True)
    assert retriever.search([]) == []
