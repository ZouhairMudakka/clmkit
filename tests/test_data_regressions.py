from __future__ import annotations

import random

import pytest

from clmkit import HashingEncoder
from clmkit.data import ContrastiveExample, iter_batches, mine_hard_negatives


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_example_rejects_nonfinite_score(score):
    with pytest.raises(ValueError, match="score must be finite"):
        ContrastiveExample("q", "p", score=score)


def test_zero_negative_mining_preserves_examples_without_encoding(monkeypatch):
    encoder = HashingEncoder(dim=8)

    def unexpected_encode(*args, **kwargs):
        pytest.fail("zero-negative mining should not encode")

    monkeypatch.setattr(encoder, "encode", unexpected_encode)
    example = ContrastiveExample("q", "p", ["existing"], instruction="search", score=0.5)
    result = mine_hard_negatives(encoder, [example], ["candidate"], num_negatives=0)
    assert result == [example]
    assert result[0].negatives is not example.negatives


@pytest.mark.parametrize("kwargs", [{"num_negatives": -1}, {"skip_top": -1}])
def test_negative_mining_rejects_negative_counts(kwargs):
    with pytest.raises(ValueError, match="must be >= 0"):
        mine_hard_negatives(HashingEncoder(dim=8), [ContrastiveExample("q", "p")], ["candidate"], **kwargs)


def _reference_batches(examples, batch_size, *, shuffle, seed, drop_last):
    """Simple stable greedy oracle, independent of occupied-position bookkeeping."""
    pending = list(examples)
    if shuffle:
        random.Random(seed).shuffle(pending)
    result = []
    while pending:
        batch, deferred, seen = [], [], set()
        for example in pending:
            if len(batch) < batch_size and example.query not in seen and example.positive not in seen:
                batch.append(example)
                seen.update((example.query, example.positive))
            else:
                deferred.append(example)
        if len(batch) < batch_size and drop_last:
            break
        result.append(batch)
        pending = deferred
    return result


@pytest.mark.parametrize("batch_size", [1, 2, 3, 8, 64])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("drop_last", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 17, 999])
def test_batching_matches_stable_greedy_order(batch_size, shuffle, drop_last, seed):
    rng = random.Random(seed)
    # Shared positives, cross-role collisions, and query==positive all occur.
    examples = [ContrastiveExample(str(rng.randrange(12)), str(rng.randrange(12))) for _ in range(100)]
    kwargs = {"shuffle": shuffle, "seed": seed, "drop_last": drop_last}
    expected = _reference_batches(examples, batch_size, **kwargs)
    actual = list(iter_batches(examples, batch_size, **kwargs))
    assert [[id(ex) for ex in batch] for batch in actual] == [[id(ex) for ex in batch] for batch in expected]


def test_drop_last_stops_at_first_incomplete_batch():
    # The second batch could be full, but historical drop_last stops at the first
    # incomplete batch. Preserve that contract when assigning future batches.
    examples = [ContrastiveExample("x", "y"), ContrastiveExample("x", "a"), ContrastiveExample("y", "b")]
    assert list(iter_batches(examples, 2, shuffle=False, drop_last=True)) == []
    assert list(iter_batches(examples, 2, shuffle=False)) == [[examples[0]], examples[1:]]


def test_batching_preserves_all_duplicate_examples_and_empty_input():
    examples = [ContrastiveExample(f"q{i}", "shared") for i in range(4000)]
    assert list(iter_batches(examples, 64, shuffle=False)) == [[ex] for ex in examples]
    assert list(iter_batches([], 64)) == []


def test_batching_yields_completed_batch_before_reading_later_examples():
    class ObservedExamples(list):
        def __getitem__(self, index):
            if index >= 2:
                pytest.fail("first complete batch should be yielded immediately")
            return super().__getitem__(index)

    examples = ObservedExamples([ContrastiveExample(f"q{i}", f"p{i}") for i in range(10)])
    first = next(iter_batches(examples, 2, shuffle=False))
    assert [ex.query for ex in first] == ["q0", "q1"]
