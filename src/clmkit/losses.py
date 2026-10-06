"""Contrastive objectives (PyTorch).

All losses share one call signature so the trainer can swap them freely::

    loss = loss_fn(query, positive, negatives=None, scores=None)

* ``query``, ``positive``: ``(B, D)`` embeddings (un-normalised is fine; losses normalise).
* ``negatives``: optional ``(B, N, D)`` hard negatives per query.
* ``scores``: optional ``(B,)`` graded similarity labels (CoSENT).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from torch import Tensor


class ContrastiveLoss(Protocol):
    def __call__(
        self, query: Tensor, positive: Tensor, negatives: Tensor | None = None, scores: Tensor | None = None
    ) -> Tensor: ...


def info_nce(
    query: Tensor,
    positive: Tensor,
    negatives: Tensor | None = None,
    *,
    temperature: float = 0.05,
    in_batch_negatives: bool = True,
    symmetric: bool = False,
    false_negative_margin: float | None = None,
) -> Tensor:
    """InfoNCE / multiple-negatives ranking loss.

    Each query must pick its positive out of a candidate set made of:

    * the other queries' positives (``in_batch_negatives=True``),
    * hard negatives - with in-batch negatives on, *every* query's hard negatives
      are shared across the batch (more negatives for free).

    ``false_negative_margin`` (as in the Qwen3-Embedding paper): candidates whose
    similarity exceeds the positive's by more than the margin are masked out, since
    they are probably unlabelled positives rather than true negatives.

    ``symmetric=True`` adds the document->query direction (CLIP-style).
    """
    import torch
    import torch.nn.functional as F

    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    q = F.normalize(query, dim=-1)
    p = F.normalize(positive, dim=-1)
    bsz = q.shape[0]
    pos_scores = (q * p).sum(-1)

    if in_batch_negatives:
        logits = q @ p.T
        targets = torch.arange(bsz, device=q.device)
    else:
        logits = pos_scores[:, None]
        targets = torch.zeros(bsz, dtype=torch.long, device=q.device)

    if negatives is not None and negatives.numel() > 0:
        n = F.normalize(negatives, dim=-1)
        if n.dim() != 3 or n.shape[0] != bsz:
            raise ValueError(f"negatives must be (B, N, D) with B={bsz}, got {tuple(n.shape)}")
        if in_batch_negatives:
            neg_logits = q @ n.reshape(-1, n.shape[-1]).T  # (B, B*N)
        else:
            neg_logits = torch.einsum("bd,bnd->bn", q, n)  # (B, N)
        logits = torch.cat([logits, neg_logits], dim=1)

    if false_negative_margin is not None:
        suspicious = logits > (pos_scores[:, None] + false_negative_margin)
        suspicious[torch.arange(bsz, device=q.device), targets] = False
        logits = logits.masked_fill(suspicious, float("-inf"))

    loss = F.cross_entropy(logits / temperature, targets)
    if symmetric and in_batch_negatives:
        loss = (loss + F.cross_entropy((p @ q.T) / temperature, torch.arange(bsz, device=q.device))) / 2
    return loss


class InfoNCELoss:
    """Configurable :func:`info_nce` (the default training objective)."""

    def __init__(
        self,
        temperature: float = 0.05,
        *,
        in_batch_negatives: bool = True,
        symmetric: bool = False,
        false_negative_margin: float | None = None,
    ) -> None:
        self.temperature = temperature
        self.in_batch_negatives = in_batch_negatives
        self.symmetric = symmetric
        self.false_negative_margin = false_negative_margin

    def __call__(
        self, query: Tensor, positive: Tensor, negatives: Tensor | None = None, scores: Tensor | None = None
    ) -> Tensor:
        return info_nce(
            query,
            positive,
            negatives,
            temperature=self.temperature,
            in_batch_negatives=self.in_batch_negatives,
            symmetric=self.symmetric,
            false_negative_margin=self.false_negative_margin,
        )


class CoSENTLoss:
    """CoSENT: rank pairs by cosine consistently with graded labels (STS-style data).

    ``loss = log(1 + sum_{s_i > s_j} exp(scale * (cos_j - cos_i)))``
    """

    def __init__(self, scale: float = 20.0) -> None:
        self.scale = scale

    def __call__(
        self, query: Tensor, positive: Tensor, negatives: Tensor | None = None, scores: Tensor | None = None
    ) -> Tensor:
        import torch
        import torch.nn.functional as F

        if scores is None:
            raise ValueError("CoSENT needs graded `scores` labels for every pair")
        if scores.shape != (query.shape[0],) or not torch.isfinite(scores).all():
            raise ValueError("CoSENT scores must be a finite label vector of shape (B,)")
        cos = (F.normalize(query, dim=-1) * F.normalize(positive, dim=-1)).sum(-1) * self.scale
        diff = cos[None, :] - cos[:, None]  # diff[i, j] = cos_j - cos_i
        mask = scores[:, None] > scores[None, :]  # only pairs where i should rank above j
        diff = diff.masked_fill(~mask, float("-inf"))
        zero = torch.zeros(1, device=diff.device, dtype=diff.dtype)
        return torch.logsumexp(torch.cat([zero, diff.flatten()]), dim=0)


class TripletLoss:
    """Cosine-distance triplet loss with a margin, using the first hard negative per query."""

    def __init__(self, margin: float = 0.2) -> None:
        self.margin = margin

    def __call__(
        self, query: Tensor, positive: Tensor, negatives: Tensor | None = None, scores: Tensor | None = None
    ) -> Tensor:
        import torch.nn.functional as F

        if negatives is None or negatives.numel() == 0:
            raise ValueError("TripletLoss needs at least one hard negative per example")
        q = F.normalize(query, dim=-1)
        pos = (q * F.normalize(positive, dim=-1)).sum(-1)
        neg = (q * F.normalize(negatives[:, 0], dim=-1)).sum(-1)
        return F.relu(neg - pos + self.margin).mean()


class MatryoshkaLoss:
    """Apply ``inner`` at several truncated dimensions so embeddings stay useful when cut
    (Matryoshka Representation Learning). Qwen3-Embedding supports MRL natively.

    Dimensions must be distinct positive integers. Weights follow the supplied
    dimension order and must be finite and nonnegative, with a positive total.
    """

    def __init__(self, inner: ContrastiveLoss, dims: Sequence[int], weights: Sequence[float] | None = None) -> None:
        if not dims:
            raise ValueError("dims must be non-empty")
        if any(isinstance(d, bool) or not isinstance(d, Integral) or d <= 0 for d in dims):
            raise ValueError("dims must be positive integers")
        if len(set(dims)) != len(dims):
            raise ValueError("dims must be distinct")
        self.inner = inner
        self.dims = list(dims)
        self.weights = [float(w) for w in weights] if weights is not None else [1.0] * len(self.dims)
        if len(self.weights) != len(self.dims):
            raise ValueError("weights must match dims")
        if any(not math.isfinite(w) or w < 0 for w in self.weights):
            raise ValueError("weights must be finite and nonnegative")
        if not math.isfinite(sum(self.weights)) or sum(self.weights) <= 0:
            raise ValueError("weights must have a finite positive total")

    def __call__(
        self, query: Tensor, positive: Tensor, negatives: Tensor | None = None, scores: Tensor | None = None
    ) -> Tensor:
        tensors = [query, positive] + ([negatives] if negatives is not None else [])
        full = min(tensor.shape[-1] for tensor in tensors)
        terms = []
        for dim, w in zip(self.dims, self.weights, strict=True):
            if dim > full:
                raise ValueError(f"matryoshka dim {dim} exceeds embedding size {full}")
            neg = negatives[..., :dim] if negatives is not None else None
            terms.append(w * self.inner(query[..., :dim], positive[..., :dim], neg, scores))
        return sum(terms[1:], terms[0]) / sum(self.weights)


def build_loss(
    name: str = "infonce", *, matryoshka_dims: Sequence[int] | None = None, **kwargs: object
) -> ContrastiveLoss:
    """Build a loss from the :data:`~clmkit.registry.LOSSES` registry, optionally MRL-wrapped."""
    from clmkit.registry import LOSSES

    loss: ContrastiveLoss = LOSSES.build(name, **kwargs)
    if matryoshka_dims:
        loss = MatryoshkaLoss(loss, matryoshka_dims)
    return loss
