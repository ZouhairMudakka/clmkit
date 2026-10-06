"""Training boundary regressions with a local torch model and no model downloads."""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from clmkit.data import ContrastiveExample  # noqa: E402
from clmkit.losses import CoSENTLoss, MatryoshkaLoss  # noqa: E402
from clmkit.training import ContrastiveTrainer, TrainConfig  # noqa: E402

pytestmark = pytest.mark.torch


class ToyEncoder:
    device = "cpu"

    def __init__(self):
        self.model = torch.nn.Sequential(torch.nn.Embedding(64, 8), torch.nn.Linear(8, 6))
        self.documents = []

    def forward(self, texts, kind="document", instruction=None):
        if kind == "document":
            self.documents.extend(texts)
        return self.model(torch.tensor([sum(t.encode()) % 64 for t in texts]))

    def save_pretrained(self, path, *, merge_adapter=False):
        Path(path).mkdir(parents=True, exist_ok=True)
        return Path(path)


@pytest.mark.parametrize("seed", [0, 7, 42])
@pytest.mark.parametrize("pattern", ["query", "positive", "mixed"])
def test_epochs_cover_every_example_and_schedule_all_batches(tmp_path, seed, pattern):
    examples = [
        ContrastiveExample(
            "same" if pattern == "query" else f"q{i % 3}" if pattern == "mixed" else f"q{i}",
            "same" if pattern == "positive" else f"p{i}",
            [f"unique-negative-{i}"],
        )
        for i in range(7)
    ]
    encoder = ToyEncoder()
    trainer = ContrastiveTrainer(
        encoder,
        TrainConfig(output_dir=str(tmp_path), epochs=3, batch_size=4, seed=seed, warmup_ratio=0, log_every=1),
        examples,
    )
    result = trainer.train()
    counts = Counter(encoder.documents)
    assert all(counts[f"unique-negative-{i}"] == 3 for i in range(7))
    assert result.global_step == trainer.total_steps == len(result.history)
    if pattern != "mixed":
        assert result.global_step == 21
    assert result.history[-1]["lr"] == 0
    assert all(record["lr"] > 0 for record in result.history[:-1])


@pytest.mark.parametrize("max_steps", [2, 8])
def test_explicit_max_steps_stops_mid_epoch_or_repeats_epochs(tmp_path, max_steps):
    encoder = ToyEncoder()
    examples = [ContrastiveExample("same", f"p{i}", [f"n{i}"]) for i in range(5)]
    trainer = ContrastiveTrainer(
        encoder, TrainConfig(output_dir=str(tmp_path), epochs=0, batch_size=4, max_steps=max_steps), examples
    )
    assert trainer.train().global_step == trainer.total_steps == max_steps
    assert len(encoder.documents) == 2 * max_steps


class InfiniteBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.new_tensor(1.0)

    @staticmethod
    def backward(ctx, upstream):
        return upstream.new_tensor(float("inf"))


def _assert_state_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_state_equal(actual[key], expected[key])
    elif isinstance(actual, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            _assert_state_equal(a, b)
    else:
        assert actual == expected


@pytest.mark.parametrize("mini_batch_size", [None, 1])
@pytest.mark.parametrize("failure", ["loss", "backward", "parameter_gradient"])
@pytest.mark.parametrize("successful_steps", [0, 1])
def test_numerical_failure_preserves_model_optimizer_scheduler(tmp_path, mini_batch_size, failure, successful_steps):
    encoder = ToyEncoder()
    trainer = None

    def loss_fn(q, p, negatives=None, scores=None):
        value = q.square().mean() + p.square().mean()
        if trainer.global_step < successful_steps:
            return value
        if failure == "loss":
            return value * float("nan")
        if failure == "backward":
            return InfiniteBackward.apply(value)
        return value

    trainer = ContrastiveTrainer(
        encoder,
        TrainConfig(
            output_dir=str(tmp_path), max_steps=successful_steps + 1, mini_batch_size=mini_batch_size, log_every=1
        ),
        [ContrastiveExample(f"q{i}", f"p{i}") for i in range(3)],
        loss_fn=loss_fn,
    )
    if failure == "parameter_gradient":
        next(encoder.model.parameters()).register_hook(
            lambda grad: grad * float("inf") if trainer.global_step >= successful_steps else grad
        )

    def snapshot():
        return copy.deepcopy(
            (encoder.model.state_dict(), trainer.optimizer.state_dict(), trainer.scheduler.state_dict())
        )

    before = snapshot()

    def after_success(record):
        nonlocal before
        before = snapshot()

    trainer.callbacks.append(after_success)
    with pytest.raises(FloatingPointError, match="non-finite"):
        trainer.train()
    for actual, expected in zip(snapshot(), before, strict=True):
        _assert_state_equal(actual, expected)
    assert trainer.global_step == successful_steps
    assert all(p.grad is None for p in encoder.model.parameters())
    assert not (tmp_path / "final").exists()


def test_weighted_matryoshka_preserves_pairs_and_is_order_invariant():
    def dimension_loss(q, p, negatives=None, scores=None):
        return q.new_tensor(float(q.shape[-1]))

    q = torch.ones(2, 4)
    assert MatryoshkaLoss(dimension_loss, [2, 4], [1, 3])(q, q).item() == 3.5
    assert MatryoshkaLoss(dimension_loss, [4, 2], [3, 1])(q, q).item() == 3.5
    assert MatryoshkaLoss(dimension_loss, [2, 4], [0, 1])(q, q).item() == 4


@pytest.mark.parametrize("dims", [[0], [-1], [2.5], [True], [2, 2]])
def test_matryoshka_rejects_invalid_dimensions(dims):
    with pytest.raises(ValueError, match="dims"):
        MatryoshkaLoss(lambda q, p, n=None, scores=None: q.sum(), dims)


@pytest.mark.parametrize("weights", [[0, 0], [-1, 2], [float("nan"), 1], [float("inf"), 1], [1e308, 1e308]])
def test_matryoshka_rejects_invalid_weights(weights):
    with pytest.raises(ValueError, match="weights"):
        MatryoshkaLoss(lambda q, p, n=None, scores=None: q.sum(), [2, 4], weights)


@pytest.mark.parametrize("short_tensor", ["query", "positive", "negative"])
def test_matryoshka_checks_every_embedding_width(short_tensor):
    q = torch.ones(2, 2 if short_tensor == "query" else 4)
    p = torch.ones(2, 2 if short_tensor == "positive" else 4)
    n = torch.ones(2, 1, 2 if short_tensor == "negative" else 4)
    with pytest.raises(ValueError, match="exceeds"):
        MatryoshkaLoss(lambda q, p, n=None, scores=None: q.sum(), [4])(q, p, n)


@pytest.mark.parametrize("scores", [[0, float("nan")], [0, float("inf")], [[0], [1]]])
def test_cosent_rejects_nonfinite_or_misshaped_labels(scores):
    q = torch.ones(2, 4)
    with pytest.raises(ValueError, match="scores"):
        CoSENTLoss()(q, q, scores=torch.tensor(scores))
