from __future__ import annotations

import pytest

from clmkit import HashingEncoder
from clmkit.data import ContrastiveExample, mine_hard_negatives


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
