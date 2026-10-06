"""Regression checks for persistence and evaluation boundary contracts."""

from __future__ import annotations

import json

import numpy as np
import pytest

from clmkit import HashingEncoder, NumpyIndex, Retriever
from clmkit.agents import SemanticMemory
from clmkit.data import ContrastiveExample
from clmkit.encoders.base import Encoder
from clmkit.eval import RetrievalEvaluator, evaluate_run, pearson, spearman
from clmkit.retrieval import EncoderMismatchWarning


class TaskEncoder(Encoder):
    name = "task-table"

    @property
    def native_dim(self) -> int:
        return 2

    def _encode(self, texts, batch_size):
        table = {
            "left:q": [1, 0],
            "right:q": [0, 1],
            "q": [1, 0],
            ":q": [1, 0],
            "a": [1, 0],
            "b": [0, 1],
            "recent": [0.6, 0.8],
            "negative": [-1, 0],
        }
        return np.array([table[t] for t in texts], dtype=np.float32)


@pytest.mark.parametrize("instructions", [("left", "right"), ("left", "left"), (None, None)])
def test_evaluator_preserves_example_instructions(instructions):
    positives = ["b" if inst == "right" else "a" for inst in instructions]
    examples = [
        ContrastiveExample("q", positive, instruction=inst)
        for inst, positive in zip(instructions, positives, strict=True)
    ]
    evaluator = RetrievalEvaluator.from_examples(examples, ks=[1], metrics=["recall"])
    assert evaluator(TaskEncoder(query_template="{instruction}:{text}")) == {"recall@1": 1.0}


@pytest.mark.parametrize("override", [None, "left", ""])
def test_evaluator_explicit_instruction_overrides_examples(override):
    examples = [ContrastiveExample("q", "a", negatives=["b"], instruction="right")]
    evaluator = RetrievalEvaluator.from_examples(examples, instruction=override, ks=[1], metrics=["recall"])
    assert evaluator(TaskEncoder(query_template="{instruction}:{text}")) == {"recall@1": 1.0}


def test_memory_roundtrip_settings_and_overrides(tmp_path):
    now = [0.0]
    encoder = TaskEncoder()
    memory = SemanticMemory(encoder, recency_half_life=1, candidate_multiplier=7, clock=lambda: now[0])
    memory.remember("a", id="old")
    now[0] = 10
    memory.remember("recent", id="new")
    memory.save(tmp_path)
    loaded = SemanticMemory.load(tmp_path, encoder, clock=lambda: now[0])
    assert loaded.recency_half_life == 1 and loaded.candidate_multiplier == 7
    assert loaded.recall("q", k=1) == memory.recall("q", k=1)
    overridden = SemanticMemory.load(tmp_path, encoder, recency_half_life=None, candidate_multiplier=2)
    assert overridden.recency_half_life is None and overridden.candidate_multiplier == 2
    assert overridden.recall("q", k=1)[0].id == "old"
    meta = Retriever.read_meta(tmp_path)
    meta.pop("memory")
    (tmp_path / "retriever.json").write_text(json.dumps(meta))
    assert SemanticMemory.load(tmp_path, encoder).recency_half_life is None


def test_memory_candidate_budget_is_approximate_and_negative_scores_approach_zero():
    now = [0.0]
    memory = SemanticMemory(TaskEncoder(), recency_half_life=1, clock=lambda: now[0])
    for i in range(4):
        memory.remember("a", id=f"old{i}")
    now[0] = 10
    memory.remember("recent", id="new")
    assert memory.recall("q", k=1)[0].id.startswith("old")
    memory.candidate_multiplier = 5
    assert memory.recall("q", k=1)[0].id == "new"
    memory.remember("negative", id="negative")
    before = memory.recall("q", k=1, filter=lambda _: False)
    assert before == []
    now[0] = 11
    negative = next(h for h in memory.recall("q", k=6) if h.id == "negative")
    assert negative.score == pytest.approx(-0.5)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_memory_rejects_invalid_decay(bad):
    with pytest.raises(ValueError):
        SemanticMemory(TaskEncoder(), recency_half_life=bad)


def test_retriever_config_identity_and_legacy_compatibility(tmp_path):
    encoder = HashingEncoder(dim=32)
    retriever = Retriever(encoder, encoder_identity="immutable-model-v1")
    retriever.add(["ABC"])
    retriever.save(tmp_path)
    assert len(Retriever.load(tmp_path, encoder, strict=True, encoder_identity="immutable-model-v1")) == 1
    for changed, identity in [
        (HashingEncoder(dim=32, lowercase=False), "immutable-model-v1"),
        (encoder, "immutable-model-v2"),
        (encoder, None),
    ]:
        with pytest.raises(ValueError, match="built with encoder"):
            Retriever.load(tmp_path, changed, strict=True, encoder_identity=identity)
    with pytest.warns(EncoderMismatchWarning):
        Retriever.load(tmp_path, encoder, encoder_identity="override")
    meta = Retriever.read_meta(tmp_path)
    meta["encoder"] = f"{encoder.name}:{encoder.dim}"
    meta.pop("fingerprint_version")
    meta.pop("encoder_identity")
    (tmp_path / "retriever.json").write_text(json.dumps(meta))
    with pytest.warns(EncoderMismatchWarning, match="legacy"):
        assert len(Retriever.load(tmp_path, encoder, strict=True)) == 1


@pytest.mark.parametrize("corruption", ["mismatch", "duplicate", "count", "bad_metadata"])
def test_retriever_rejects_inconsistent_store(tmp_path, corruption):
    encoder = HashingEncoder(dim=32)
    retriever = Retriever(encoder)
    retriever.add(["a", "b"], ids=["a", "b"])
    retriever.save(tmp_path)
    path = tmp_path / "documents.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if corruption in ("mismatch", "duplicate"):
        records[1]["id"] = "c" if corruption == "mismatch" else "a"
    elif corruption == "bad_metadata":
        records[0]["metadata"] = []
    else:
        meta = Retriever.read_meta(tmp_path)
        meta["count"] = 1
        (tmp_path / "retriever.json").write_text(json.dumps(meta))
    path.write_text("\n".join(json.dumps(record) for record in records))
    with pytest.raises(ValueError, match="corrupt"):
        Retriever.load(tmp_path, encoder)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_numpy_rejects_invalid_query_chunk(bad):
    with pytest.raises(ValueError):
        NumpyIndex(2, query_chunk=bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("metric", ["cosine", "dot"])
def test_numpy_rejects_nonfinite_query_and_stored_vectors(tmp_path, bad, metric):
    index = NumpyIndex(2, metric=metric, query_chunk=2)
    index.add(["a"], np.array([[1, 0]]))
    index.save(tmp_path)
    assert NumpyIndex.load(tmp_path).query_chunk == 2
    with pytest.raises(ValueError, match="NaN or inf"):
        index.search(np.array([bad, 1]))
    np.save(tmp_path / "vectors.npy", np.array([[bad, 0]]))
    with pytest.raises(ValueError, match="NaN or inf"):
        NumpyIndex.load(tmp_path)


@pytest.mark.parametrize("ids", [["a", "a"], ["a", ""], ["a", 2]])
def test_numpy_rejects_invalid_saved_ids(tmp_path, ids):
    index = NumpyIndex(2)
    index.add(["a", "b"], np.eye(2))
    index.save(tmp_path)
    (tmp_path / "ids.json").write_text(json.dumps(ids))
    with pytest.raises(ValueError, match="corrupt"):
        NumpyIndex.load(tmp_path)


@pytest.mark.parametrize("factory", ["flat", "hnsw"])
def test_faiss_persists_budget_and_rejects_mismatched_native_ids(tmp_path, factory):
    pytest.importorskip("faiss")
    from clmkit.index.faiss_index import FaissIndex

    index = FaissIndex(2, factory=factory, filter_overfetch=17)
    index.add(["a", "b"], np.eye(2))
    index.save(tmp_path)
    restored = FaissIndex.load(tmp_path)
    assert restored.filter_overfetch == 17
    assert restored.search(np.array([1, 0]), k=1)[0][0][0] == "a"
    (tmp_path / "ids.json").write_text(json.dumps({"1": "a", "2": "b"}))
    with pytest.raises(ValueError, match="native ids/dimension mismatch"):
        FaissIndex.load(tmp_path)


def test_metrics_deduplicate_and_validate():
    assert evaluate_run({"q": ["a", "a", "b"]}, {"q": {"a": 1, "b": 1}}, ks=[2]) == {
        "ndcg@2": 1.0,
        "mrr@2": 1.0,
        "recall@2": 1.0,
        "map@2": 1.0,
    }
    for ks in ([0], [-1], [True], [1.5], []):
        with pytest.raises(ValueError):
            evaluate_run({}, {}, ks=ks)
    with pytest.raises(ValueError, match="finite"):
        evaluate_run({}, {"q": {"a": float("nan")}})
    for correlation in (pearson, spearman):
        with pytest.raises(ValueError, match="finite"):
            correlation([1, float("nan")], [1, 2])


@pytest.mark.parametrize("backend", ["numpy", "faiss"])
def test_index_format_version_validation_and_legacy_default(tmp_path, backend):
    from clmkit.index.base import load_index

    if backend == "faiss":
        pytest.importorskip("faiss")
        from clmkit.index.faiss_index import FaissIndex

        index = FaissIndex(2)
    else:
        index = NumpyIndex(2)
    index.add(["a"], np.array([[1, 0]]))
    index.save(tmp_path)
    path = tmp_path / "index.json"
    meta = json.loads(path.read_text())
    for version in (0, 2, "1", None, True, 1.0):
        meta["format_version"] = version
        path.write_text(json.dumps(meta))
        for loader in (type(index).load, load_index):
            with pytest.raises(ValueError, match="unsupported index format_version"):
                loader(tmp_path)
    meta.pop("format_version")
    path.write_text(json.dumps(meta))
    assert load_index(tmp_path).search(np.array([1, 0]), 1) == [[("a", 1.0)]]
    meta["format_version"] = 1
    path.write_text(json.dumps(meta))
    assert type(index).load(tmp_path).ids() == ["a"]
