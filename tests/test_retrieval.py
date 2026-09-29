from __future__ import annotations

import pytest

from clmkit import HashingEncoder, NumpyIndex, Retriever, chunk_text
from clmkit.rerank import EncoderReranker, load_reranker
from clmkit.retrieval import EncoderMismatchWarning

DOCS = [
    ("Paris is the capital of France and home of the Eiffel Tower.", {"lang": "en", "topic": "geo"}),
    ("Berlin is the capital of Germany.", {"lang": "en", "topic": "geo"}),
    ("Photosynthesis converts light into chemical energy in plants.", {"lang": "en", "topic": "bio"}),
    ("La tour Eiffel est à Paris.", {"lang": "fr", "topic": "geo"}),
]


@pytest.fixture
def retriever(hashing: HashingEncoder) -> Retriever:
    r = Retriever(hashing)
    r.add([d for d, _ in DOCS], ids=[f"d{i}" for i in range(len(DOCS))], metadata=[m for _, m in DOCS])
    return r


def test_search_basic_and_batch(retriever: Retriever) -> None:
    hits = retriever.search("what is the capital of Germany", k=2)
    assert hits[0].id == "d1" and hits[0].metadata["topic"] == "geo" and hits[0].text
    batch = retriever.search(["capital of Germany", "how do plants use light"], k=1)
    assert [b[0].id for b in batch] == ["d1", "d2"]
    assert hits[0].to_dict()["id"] == "d1"


def test_filters(retriever: Retriever) -> None:
    assert {h.id for h in retriever.search("Eiffel Tower Paris", k=5, filter={"lang": "fr"})} == {"d3"}
    hits = retriever.search("capital", k=5, filter=lambda m: m["topic"] == "bio")
    assert [h.id for h in hits] == ["d2"]
    assert retriever.search("capital", k=5, min_score=0.99) == []


def test_add_validation_and_delete(retriever: Retriever, hashing: HashingEncoder) -> None:
    with pytest.raises(TypeError, match="sequence"):
        retriever.add("a single string")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="same length"):
        retriever.add(["a", "b"], ids=["x"])
    with pytest.raises(ValueError, match="already in index"):
        retriever.add(["dup"], ids=["d0"])
    with pytest.raises(TypeError):
        retriever.add(["x"], metadata=[{"bad": object()}])
    assert len(retriever) == 4  # failed adds changed nothing
    assert retriever.add([]) == []
    auto = retriever.add(["new doc"])
    assert len(auto[0]) == 32 and auto[0] in retriever
    assert retriever.delete(["d0", "nope"]) == 1
    assert "d0" not in retriever and retriever.get("d0") is None
    with pytest.raises(ValueError, match="dim"):
        Retriever(hashing, NumpyIndex(8))


def test_persistence_and_encoder_mismatch(tmp_path, retriever: Retriever) -> None:
    retriever.save(tmp_path / "r", extra_meta={"encoder_spec": {"model": "hashing"}})
    assert Retriever.read_meta(tmp_path / "r")["encoder_spec"] == {"model": "hashing"}
    loaded = Retriever.load(tmp_path / "r", HashingEncoder(dim=256))
    assert len(loaded) == 4
    assert loaded.search("capital of Germany", k=1)[0].id == "d1"
    assert loaded.get("d3").metadata == {"lang": "fr", "topic": "geo"}
    with pytest.warns(EncoderMismatchWarning):
        Retriever.load(tmp_path / "r", _other_encoder_same_dim())
    with pytest.raises(ValueError, match="built with encoder"):
        Retriever.load(tmp_path / "r", _other_encoder_same_dim(), strict=True)


def _other_encoder_same_dim() -> HashingEncoder:
    enc = HashingEncoder(dim=256)
    enc.name = "some-other-model"
    return enc


def test_reranking_pipeline(retriever: Retriever) -> None:
    retriever.reranker = EncoderReranker(HashingEncoder(dim=512))
    hits = retriever.search("capital of France", k=2, rerank_candidates=4)
    assert len(hits) == 2
    assert all("retrieval_score" in h.metadata for h in hits)
    assert hits[0].score >= hits[1].score
    plain = retriever.search("capital of France", k=2, rerank=False)
    assert all("retrieval_score" not in h.metadata for h in plain)
    retriever.reranker = None
    with pytest.raises(ValueError, match="no reranker"):
        retriever.search("x", rerank=True)


def test_reranker_requires_text() -> None:
    from clmkit.types import SearchHit

    rr = EncoderReranker(HashingEncoder(dim=64))
    assert rr.rerank("q", []) == []
    with pytest.raises(ValueError, match="without text"):
        rr.rerank("q", [SearchHit("a", 1.0)])
    assert isinstance(load_reranker("encoder:hashing", dim=32), EncoderReranker)
    assert load_reranker(rr) is rr


def test_chunk_text() -> None:
    assert chunk_text("") == []
    assert chunk_text("short") == ["short"]
    text = "\n\n".join(f"Paragraph {i}. " + "Sentence here. " * 10 for i in range(5))
    chunks = chunk_text(text, max_chars=200, overlap=30)
    assert len(chunks) > 3 and all(len(c) <= 200 for c in chunks)
    words = set(text.split())
    assert words == set(" ".join(chunks).split())  # nothing lost
    long_word = "x" * 450
    assert [len(c) for c in chunk_text(long_word, 200, 0)] == [200, 200, 50]
    with pytest.raises(ValueError):
        chunk_text("abc", 10, 10)
    with pytest.raises(ValueError):
        chunk_text("abc", 0)
