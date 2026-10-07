"""Intent batching and bounded mining contracts, with no model downloads."""

from __future__ import annotations

import copy
import random

import numpy as np
import pytest

from clmkit.data import ContrastiveExample, iter_batches, load_examples, mine_hard_negatives, save_examples


def test_label_roundtrip_and_legacy_schema(tmp_path):
    old = ContrastiveExample("q", "p", ["n"], "instruction", 0.5)
    labelled = ContrastiveExample("q2", "p2", label="intent")
    assert "label" not in old.to_dict()
    assert load_examples(save_examples(tmp_path / "pairs.jsonl", [old, labelled])) == [old, labelled]
    for label in ("", " ", 3):
        with pytest.raises(ValueError, match="label"):
            ContrastiveExample("q", "p", label=label)


@pytest.mark.parametrize("seed", [0, 42, 1729])
@pytest.mark.parametrize("batch_size", [1, 3, 8])
def test_label_batches_match_greedy_oracle_and_cover_all_examples(seed, batch_size):
    examples = [ContrastiveExample(f"q{i % 11}", f"p{i % 13}", label=str(i % 5)) for i in range(60)]
    pending = list(examples)
    random.Random(seed).shuffle(pending)
    expected = []
    while pending:
        batch, deferred, labels, texts = [], [], set(), set()
        for ex in pending:
            if len(batch) < batch_size and ex.label not in labels and not {ex.query, ex.positive} & texts:
                batch.append(ex)
                labels.add(ex.label)
                texts.update((ex.query, ex.positive))
            else:
                deferred.append(ex)
        expected.append(batch)
        pending = deferred
    actual = list(iter_batches(examples, batch_size, seed=seed, avoid_same_label=True, avoid_duplicates=False))
    assert actual == expected
    assert sorted(id(ex) for batch in actual for ex in batch) == sorted(map(id, examples))


def test_label_batching_is_opt_in_and_requires_complete_labels():
    examples = [ContrastiveExample("q1", "p1", label="a"), ContrastiveExample("q2", "p2", label="a")]
    assert list(iter_batches(examples, 2, shuffle=False)) == [examples]
    assert list(iter_batches(examples, 2, shuffle=False, avoid_same_label=True)) == [[ex] for ex in examples]
    examples.append(ContrastiveExample("q3", "p3"))
    with pytest.raises(ValueError, match="every example"):
        list(iter_batches(examples, 2, avoid_same_label=True))


class TableEncoder:
    def __init__(self, vectors):
        self.vectors = vectors
        self.products = []

    def encode(self, texts, **kwargs):
        owner = self

        class ObservedArray(np.ndarray):
            def __matmul__(self, other):
                owner.products.append((self.shape, other.shape))
                return np.asarray(self) @ np.asarray(other)

        return np.asarray([self.vectors[t] for t in texts], dtype=np.float32).view(ObservedArray)


@pytest.mark.parametrize(("query_block", "corpus_block"), [(1, 1), (2, 3), (128, 4096)])
@pytest.mark.parametrize("labelled", [False, True])
@pytest.mark.parametrize(("skip_top", "ratio"), [(0, None), (1, 0.95), (3, None)])
def test_block_mining_matches_full_matrix_stable_oracle(query_block, corpus_block, labelled, skip_top, ratio):
    # Integer vectors make exact score ties intentional and reproducible.
    rng = np.random.default_rng(72)
    corpus = [f"d{i}" for i in range(17)]
    labels = {text: str(i % 4) for i, text in enumerate(corpus)}
    vectors = {text: rng.integers(-2, 3, size=5) for text in corpus}
    examples = []
    for i in range(5):
        vectors[f"q{i}"] = rng.integers(-2, 3, size=5)
        examples.append(
            ContrastiveExample(f"q{i}", corpus[i], [corpus[(i + 1) % 17]], label=str(i % 4) if labelled else None)
        )
    vectors["d16"] = vectors["d15"].copy()
    encoder = TableEncoder(vectors)
    actual = mine_hard_negatives(
        encoder,
        examples,
        [*corpus, corpus[0]],
        num_negatives=2,
        skip_top=skip_top,
        max_relative_score=ratio,
        query_block_size=query_block,
        corpus_block_size=corpus_block,
        corpus_labels=[labels[t] for t in [*corpus, corpus[0]]] if labelled else None,
    )
    sims = np.array([vectors[e.query] for e in examples]) @ np.array([vectors[t] for t in corpus]).T
    for i, ex in enumerate(examples):
        ranked = sorted(range(len(corpus)), key=lambda j: (-sims[i, j], j))[: skip_top + 7]
        expected = []
        for j in ranked[skip_top:]:
            text = corpus[j]
            if text == ex.positive or text in ex.negatives or (labelled and labels[text] == ex.label):
                continue
            if ratio is not None and sims[i, j] > ratio * np.dot(vectors[ex.query], vectors[ex.positive]):
                continue
            expected.append(text)
        assert actual[i].negatives == ex.negatives + expected[:2]
        assert actual[i].label == ex.label
    assert encoder.products
    assert all(q[0] <= query_block and d[1] <= corpus_block for q, d in encoder.products)


@pytest.mark.parametrize(
    ("examples", "corpus", "labels", "error"),
    [
        ([ContrastiveExample("q", "p", label="a")], ["p"], None, "aligned"),
        ([ContrastiveExample("q", "p")], ["p"], ["a"], "every example"),
        ([ContrastiveExample("q", "p", label="a")], ["p", "p"], ["a", "b"], "conflicting"),
        ([ContrastiveExample("q", "p", label="a")], ["p"], ["b"], "conflicting"),
        ([ContrastiveExample("q", "p", ["unknown"], label="a")], ["p"], ["a"], "existing negatives"),
        ([ContrastiveExample("q", "p", ["n"], label="a")], ["p", "n"], ["a", "a"], "existing negatives"),
    ],
)
def test_label_mining_rejects_ambiguous_labels_before_encoding(examples, corpus, labels, error):
    with pytest.raises(ValueError, match=error):
        mine_hard_negatives(TableEncoder({}), examples, corpus, corpus_labels=labels)


def test_mining_equal_scores_follow_first_corpus_order_across_blocks():
    corpus = ["z", "a", "m", "z", "b"]
    encoder = TableEncoder({text: [1.0, 0.0] for text in [*corpus, "q", "p"]})
    result = mine_hard_negatives(
        encoder,
        [ContrastiveExample("q", "p")],
        corpus,
        num_negatives=2,
        skip_top=1,
        max_relative_score=None,
        corpus_block_size=1,
    )
    assert result[0].negatives == ["a", "m"]


def test_zero_mining_retains_label_without_encoding():
    example = ContrastiveExample("q", "p", ["n"], label="intent")
    result = mine_hard_negatives(TableEncoder({}), [example], [], num_negatives=0)
    assert result == [example]
    assert result[0].negatives is not example.negatives


@pytest.mark.parametrize("kwarg", ["query_block_size", "corpus_block_size", "batch_size"])
def test_mining_rejects_zero_block_sizes(kwarg):
    with pytest.raises(ValueError, match="must be >= 1"):
        mine_hard_negatives(TableEncoder({}), [ContrastiveExample("q", "p")], ["p"], **{kwarg: 0})


def _toy_encoder(torch):
    class ToyEncoder:
        device = "cpu"

        def __init__(self):
            self.model = torch.nn.Sequential(torch.nn.Embedding(64, 8), torch.nn.Linear(8, 6))

        def forward(self, texts, kind="document", instruction=None):
            return self.model(torch.tensor([sum(t.encode()) % 64 for t in texts]))

    return ToyEncoder()


@pytest.mark.torch
def test_label_batches_full_and_gradcache_match_loss_and_gradients():
    torch = pytest.importorskip("torch")
    from clmkit.training import ContrastiveTrainer, TrainConfig

    examples = [ContrastiveExample(f"q{i}", f"p{i}", label=str(i % 3)) for i in range(9)]
    full_encoder = _toy_encoder(torch)
    chunked_encoder = copy.deepcopy(full_encoder)
    base = dict(batch_size=6, avoid_same_label=True, max_negatives=0)
    full = ContrastiveTrainer(full_encoder, TrainConfig(**base), examples)
    chunked = ContrastiveTrainer(chunked_encoder, TrainConfig(**base, mini_batch_size=1), examples)
    assert full.total_steps == chunked.total_steps == 3
    for batch in full._epoch_batches(0):
        assert len(batch) == len({ex.label for ex in batch}) == 3
        assert full.training_step(batch) == pytest.approx(chunked.training_step(batch), rel=1e-5)
    for p, q in zip(full_encoder.model.parameters(), chunked_encoder.model.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-5)


@pytest.mark.torch
@pytest.mark.parametrize("mini_batch_size", [None, 1])
def test_label_training_rejects_direct_batch_conflicts_and_explicit_negatives(mini_batch_size):
    torch = pytest.importorskip("torch")
    from clmkit.training import ContrastiveTrainer, TrainConfig

    encoder = _toy_encoder(torch)
    cfg = TrainConfig(avoid_same_label=True, mini_batch_size=mini_batch_size)
    examples = [ContrastiveExample("q1", "p1", ["n"], label="a")]
    with pytest.raises(ValueError, match="explicit negatives"):
        ContrastiveTrainer(encoder, cfg, examples)
    cfg.max_negatives = 0
    trainer = ContrastiveTrainer(encoder, cfg, examples)
    assert trainer._prepare(examples)[2] == ["p1"]
    for extra, error in (
        (ContrastiveExample("q2", "p2", label="a"), "distinct labels"),
        (ContrastiveExample("p1", "p2", label="b"), "distinct texts"),
        (ContrastiveExample("q2", "p2"), "every example"),
    ):
        with pytest.raises(ValueError, match=error):
            trainer.training_step([*examples, extra])
        assert all(p.grad is None for p in encoder.model.parameters())
