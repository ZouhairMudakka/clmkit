from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from clmkit.losses import (  # noqa: E402
    CoSENTLoss,
    InfoNCELoss,
    MatryoshkaLoss,
    TripletLoss,
    build_loss,
    info_nce,
)
from clmkit.pooling import first_token_pool, get_pooler, last_token_pool, mean_pool  # noqa: E402

pytestmark = pytest.mark.torch


def _emb(*shape: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


# ------------------------------------------------------------------ pooling --
def test_last_token_pool_is_padding_side_agnostic() -> None:
    h = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    right = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    left = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]])
    assert torch.equal(last_token_pool(h, right), torch.stack([h[0, 3], h[1, 1]]))
    assert torch.equal(last_token_pool(h, left), torch.stack([h[0, 3], h[1, 3]]))
    assert torch.equal(first_token_pool(h, left), torch.stack([h[0, 0], h[1, 2]]))
    assert torch.allclose(mean_pool(h, right)[1], h[1, :2].mean(0))
    assert get_pooler("mean") is mean_pool
    with pytest.raises(ValueError):
        get_pooler("max")


# ------------------------------------------------------------------- losses --
def test_info_nce_matches_manual_cross_entropy() -> None:
    q, p = _emb(4, 8), _emb(4, 8, seed=1)
    t = 0.1
    logits = F.normalize(q, dim=-1) @ F.normalize(p, dim=-1).T / t
    expected = F.cross_entropy(logits, torch.arange(4))
    assert torch.allclose(info_nce(q, p, temperature=t), expected, atol=1e-6)


def test_info_nce_with_hard_negatives_shared_across_batch() -> None:
    q, p, n = _emb(3, 8), _emb(3, 8, seed=1), _emb(3, 2, 8, seed=2)
    qn, pn, nn = F.normalize(q, dim=-1), F.normalize(p, dim=-1), F.normalize(n, dim=-1)
    logits = torch.cat([qn @ pn.T, qn @ nn.reshape(6, 8).T], dim=1) / 0.05
    expected = F.cross_entropy(logits, torch.arange(3))
    assert torch.allclose(info_nce(q, p, n), expected, atol=1e-5)
    # without in-batch negatives: [pos, own negatives] only
    own = torch.cat([(qn * pn).sum(-1, keepdim=True), torch.einsum("bd,bnd->bn", qn, nn)], 1) / 0.05
    assert torch.allclose(
        info_nce(q, p, n, in_batch_negatives=False), F.cross_entropy(own, torch.zeros(3, dtype=torch.long)), atol=1e-5
    )
    with pytest.raises(ValueError, match="negatives must be"):
        info_nce(q, p, _emb(2, 2, 8))
    with pytest.raises(ValueError, match="temperature"):
        info_nce(q, p, temperature=0)


def test_info_nce_symmetric_and_perfect_alignment() -> None:
    q = torch.eye(4)
    assert info_nce(q, q.clone(), temperature=0.01) < 1e-6
    q2, p2 = _emb(4, 8), _emb(4, 8, seed=3)
    sym = info_nce(q2, p2, symmetric=True)
    fwd, bwd = info_nce(q2, p2), info_nce(p2, q2)
    assert torch.allclose(sym, (fwd + bwd) / 2, atol=1e-6)


def test_false_negative_masking_removes_suspicious_candidates() -> None:
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    p = torch.tensor([[0.6, 0.8], [0.0, 1.0]])  # query 0's positive is weak...
    n = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]])  # ...and its "negative" is identical to it
    plain = info_nce(q, p, n, temperature=0.1)
    masked = info_nce(q, p, n, temperature=0.1, false_negative_margin=0.1)
    assert masked < plain
    assert torch.isfinite(masked)


def test_cosent_orders_pairs() -> None:
    a = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    b_good = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # cos: 1, 0
    b_bad = torch.tensor([[0.0, 1.0], [1.0, 0.0]])  # cos: 0, 1
    scores = torch.tensor([1.0, 0.0])
    loss = CoSENTLoss(scale=20.0)
    assert loss(a, b_good, scores=scores) < 1e-6
    assert loss(a, b_bad, scores=scores) == pytest.approx(math.log1p(math.exp(20.0)), rel=1e-4)
    with pytest.raises(ValueError, match="scores"):
        loss(a, b_good)


def test_triplet() -> None:
    q = torch.tensor([[1.0, 0.0]])
    assert TripletLoss(0.2)(q, q, torch.tensor([[[0.0, 1.0]]])) == 0
    assert TripletLoss(0.2)(q, torch.tensor([[0.0, 1.0]]), q[:, None, :]) == pytest.approx(1.2)
    with pytest.raises(ValueError):
        TripletLoss()(q, q)


def test_matryoshka_wraps_and_validates() -> None:
    q, p = _emb(4, 16), _emb(4, 16, seed=1)
    inner = InfoNCELoss(0.05)
    mrl = MatryoshkaLoss(inner, [16, 8, 4])
    expected = (inner(q, p) + inner(q[:, :8], p[:, :8]) + inner(q[:, :4], p[:, :4])) / 3
    assert torch.allclose(mrl(q, p), expected, atol=1e-6)
    with pytest.raises(ValueError, match="exceeds"):
        MatryoshkaLoss(inner, [32])(q, p)
    with pytest.raises(ValueError):
        MatryoshkaLoss(inner, [])
    with pytest.raises(ValueError):
        MatryoshkaLoss(inner, [8, 4], weights=[1.0])
    assert isinstance(build_loss("infonce", matryoshka_dims=[8], temperature=0.02), MatryoshkaLoss)
    assert isinstance(build_loss("mnrl"), InfoNCELoss)


def test_gradients_flow() -> None:
    q = _emb(4, 8).requires_grad_()
    p = _emb(4, 8, seed=1).requires_grad_()
    info_nce(q, p, false_negative_margin=0.1, symmetric=True).backward()
    assert q.grad is not None and torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0
