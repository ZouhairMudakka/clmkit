"""Independent deterministic audit probes. Failures express proposed safety contracts.

Run directly: does not use repository conftest or pytest plugin autoload.
No network, downloads, torch imports, or writes outside this artifact directory.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from clmkit import HashingEncoder, NumpyIndex, Retriever
from clmkit.agents import Route, SemanticMemory, SemanticRouter
from clmkit.data import ContrastiveExample, iter_batches, mine_hard_negatives
from clmkit.encoders.base import Encoder
from clmkit.eval import RetrievalEvaluator, evaluate_run, spearman

ROOT = Path(__file__).resolve().parent


class TableEncoder(Encoder):
    name = "audit-table"

    def __init__(self, table, **kwargs):
        super().__init__(**kwargs)
        self.table = table

    @property
    def native_dim(self):
        return 2

    def _encode(self, texts, batch_size):
        return np.array([self.table[t] for t in texts], dtype=np.float32)


class CoreAudit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT)
        self.path = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_positive_randomized_filtered_search_and_roundtrip(self):
        rng = np.random.default_rng(74392)
        for metric in ("cosine", "dot"):
            for trial in range(12):
                vectors = rng.normal(size=(37, 9)).astype(np.float32)
                queries = rng.normal(size=(5, 9)).astype(np.float32)
                ids = [f"d{i}" for i in range(37)]
                removed = set(rng.choice(ids, 7, replace=False))
                allowed = set(rng.choice(ids, 8, replace=False)) - removed
                index = NumpyIndex(9, metric, query_chunk=2)
                index.add(ids, vectors)
                self.assertEqual(index.remove(removed), 7)
                if metric == "cosine":
                    # Independent double-precision cosine oracle.
                    v = vectors.astype(np.float64)
                    q = queries.astype(np.float64)
                    v /= np.sqrt((v * v).sum(axis=1, keepdims=True))
                    q /= np.sqrt((q * q).sum(axis=1, keepdims=True))
                else:
                    v, q = vectors.astype(np.float64), queries.astype(np.float64)
                result = index.search(queries, k=4, allowed_ids=allowed | {"absent"})
                for row, query in zip(result, q, strict=True):
                    expected = sorted(((id_, float(query @ v[i])) for i, id_ in enumerate(ids)
                                       if id_ in allowed), key=lambda x: x[1], reverse=True)[:4]
                    self.assertEqual([i for i, _ in row], [i for i, _ in expected])
                    np.testing.assert_allclose([s for _, s in row], [s for _, s in expected], atol=2e-6)
                folder = self.path / f"{metric}{trial}"
                index.save(folder)
                restored = NumpyIndex.load(folder).search(queries, 4, allowed_ids=allowed)
                for before, after in zip(result, restored, strict=True):
                    self.assertEqual([i for i, _ in before], [i for i, _ in after])
                    np.testing.assert_allclose([s for _, s in before], [s for _, s in after], atol=2e-6)

    def test_positive_duplicate_add_is_atomic(self):
        retriever = Retriever(HashingEncoder(dim=32))
        retriever.add(["first"], ids=["one"])
        before = retriever.search("first")
        with self.assertRaises(ValueError):
            retriever.add(["second", "third"], ids=["two", "one"])
        self.assertEqual(retriever.index.ids(), ["one"])
        self.assertEqual(retriever.search("first"), before)

    def test_positive_selective_filter_before_topk(self):
        retriever = Retriever(HashingEncoder(dim=64))
        retriever.add(["target"] * 80 + ["unrelated"], ids=[str(i) for i in range(81)],
                      metadata=[{"tenant": "other"}] * 80 + [{"tenant": "mine"}])
        hits = retriever.search("target", k=1, filter={"tenant": "mine"})
        self.assertEqual([h.id for h in hits], ["80"])
        self.assertEqual(retriever.search("target", filter={"tenant": "missing"}), [])

    def test_positive_batch_preserves_every_example(self):
        for size in (1, 3, 9):
            examples = [ContrastiveExample(f"q{i % 4}", f"p{i % 3}", score=i) for i in range(41)]
            batches = list(iter_batches(examples, size, seed=438))
            self.assertCountEqual([id(e) for batch in batches for e in batch], [id(e) for e in examples])
            for batch in batches:
                self.assertLessEqual(len(batch), size)
                self.assertEqual(len({e.query for e in batch}), len(batch))
                self.assertEqual(len({e.positive for e in batch}), len(batch))

    def test_positive_metrics_hand_oracle(self):
        result = evaluate_run({"q": ["b", "a", "x"]}, {"q": {"a": 3, "b": 1}, "missing": {"z": 1}},
                              ks=[2], metrics=["recall", "precision", "ndcg", "map", "mrr"])
        self.assertAlmostEqual(result["ndcg@2"], (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3)) / 2)
        for key in ("recall@2", "precision@2", "map@2", "mrr@2"):
            self.assertEqual(result[key], 0.5)
        self.assertAlmostEqual(spearman([1, 2, 2, 4], [4, 2, 2, 1]), -1)

    def test_positive_router_aggregations_and_abstention(self):
        encoder = TableEncoder({"q": [1, 0], "a": [1, 0], "b": [0, 1], "c": [-1, 0]})
        for aggregation, expected in (("max", 1), ("mean", 0.5), ("centroid", 1 / math.sqrt(2))):
            router = SemanticRouter(encoder, [Route("r", ["a", "b"]), Route("s", ["c"])],
                                    aggregation=aggregation, threshold=expected + 0.01)
            self.assertAlmostEqual(router.scores("q")["r"], expected, places=6)
            self.assertIsNone(router.route("q"))

    def test_query_nonfinite_rejected(self):
        for metric in ("cosine", "dot"):
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(metric=metric, value=value):
                    index = NumpyIndex(2, metric)
                    index.add(["a"], np.array([[1, 0]]))
                    with self.assertRaises(ValueError):
                        index.search(np.array([value, 1]))

    def test_loaded_nonfinite_vectors_rejected(self):
        index = NumpyIndex(2)
        index.add(["a"], np.array([[1, 0]]))
        index.save(self.path)
        np.save(self.path / "vectors.npy", np.array([[np.nan, 0]], dtype=np.float32))
        with self.assertRaises(ValueError):
            NumpyIndex.load(self.path)

    def test_loaded_duplicate_ids_rejected(self):
        index = NumpyIndex(2)
        index.add(["a", "b"], np.eye(2))
        index.save(self.path)
        (self.path / "ids.json").write_text('["a", "a"]', encoding="utf-8")
        with self.assertRaises(ValueError):
            NumpyIndex.load(self.path)

    def test_retriever_loaded_id_set_mismatch_rejected(self):
        encoder = HashingEncoder(dim=32)
        retriever = Retriever(encoder)
        retriever.add(["alpha", "beta"], ids=["a", "b"])
        retriever.save(self.path)
        records = [json.loads(line) for line in (self.path / "documents.jsonl").read_text().splitlines()]
        records[1]["id"] = "c"
        (self.path / "documents.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        with self.assertRaises(ValueError):
            Retriever.load(self.path, encoder, strict=True)

    def test_strict_fingerprint_detects_changed_encoder_config(self):
        first = HashingEncoder(dim=32, lowercase=True)
        changed = HashingEncoder(dim=32, lowercase=False)
        self.assertFalse(np.array_equal(first.encode("ABC"), changed.encode("ABC")))
        retriever = Retriever(first)
        retriever.add(["ABC"])
        retriever.save(self.path)
        with self.assertRaises(ValueError):
            Retriever.load(self.path, changed, strict=True)

    def test_fingerprint_detects_changed_document_prompt(self):
        first = HashingEncoder(dim=32)
        changed = HashingEncoder(dim=32, document_template="important {text}")
        self.assertFalse(np.array_equal(first.encode("ABC"), changed.encode("ABC")))
        self.assertNotEqual(first.fingerprint(), changed.fingerprint())

    def test_memory_decay_candidate_budget_controls_approximation(self):
        table = {"q": [1, 0], **{f"old{i}": [1, 0] for i in range(4)}, "new": [0.6, 0.8]}
        now = [0.0]
        memory = SemanticMemory(TableEncoder(table), recency_half_life=1, clock=lambda: now[0])
        for i in range(4):
            memory.remember(f"old{i}", id=f"old{i}")
        now[0] = 10.0
        memory.remember("new", id="new")
        # F07 permits bounded candidate reranking for alpha. The default budget
        # does not promise global decayed top-k; a comprehensive pool does here.
        self.assertTrue(memory.recall("q", k=1)[0].id.startswith("old"))
        memory.candidate_multiplier = 5
        self.assertEqual(memory.recall("q", k=1)[0].id, "new")

    def test_memory_persistence_preserves_decay(self):
        encoder = TableEncoder({"q": [1, 0], "old": [1, 0], "new": [0.6, 0.8]})
        now = [0.0]
        memory = SemanticMemory(encoder, recency_half_life=1, clock=lambda: now[0])
        memory.remember("old", id="old")
        now[0] = 10.0
        memory.remember("new", id="new")
        memory.save(self.path)
        restored = SemanticMemory.load(self.path, encoder, clock=lambda: now[0])
        self.assertEqual(restored.recall("q", k=1)[0].id, memory.recall("q", k=1)[0].id)

    def test_metrics_duplicate_docs_cannot_exceed_one(self):
        metrics = evaluate_run({"q": ["a", "a"]}, {"q": {"a": 1}}, ks=[2], metrics=["ndcg", "map"])
        for name, score in metrics.items():
            self.assertLessEqual(score, 1, f"{name}={score}")

    def test_metrics_nonpositive_cutoff_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_run({"q": ["a"]}, {"q": {"a": 1}}, ks=[-1])

    def test_training_nonfinite_score_rejected(self):
        with self.assertRaises(ValueError):
            ContrastiveExample("q", "p", score=float("nan"))

    def test_zero_requested_negatives_stays_zero(self):
        encoder = TableEncoder({"q": [1, 0], "p": [0, 1], "n": [1, 0]})
        result = mine_hard_negatives(encoder, [ContrastiveExample("q", "p")], ["n", "p"],
                                     num_negatives=0, max_relative_score=None)
        self.assertEqual(result[0].negatives, [])

    def test_example_evaluation_preserves_per_query_instructions(self):
        examples = [ContrastiveExample("q", "a", instruction="left"),
                    ContrastiveExample("q", "b", instruction="right")]
        encoder = TableEncoder({"left:q": [1, 0], "right:q": [0, 1], "q": [1, 0], "a": [1, 0], "b": [0, 1]},
                               query_template="{instruction}:{text}")
        evaluator = RetrievalEvaluator.from_examples(examples, ks=[1], metrics=["recall"])
        self.assertEqual(evaluator(encoder)["recall@1"], 1.0)

    def test_numpy_negative_query_chunk_rejected(self):
        with self.assertRaises(ValueError):
            NumpyIndex(2, query_chunk=-1)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CoreAudit)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    payload = {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
               "torch_imported": "torch" in sys.modules, "python": sys.executable,
               "failure_details": [{"test": str(test), "traceback": tb} for test, tb in result.failures],
               "error_details": [{"test": str(test), "traceback": tb} for test, tb in result.errors]}
    (ROOT / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    sys.exit(not result.wasSuccessful())
