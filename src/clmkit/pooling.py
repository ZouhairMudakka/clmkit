"""Pooling strategies that turn token states ``(B, T, H)`` into sentence vectors ``(B, H)``.

All functions are padding-side agnostic: they locate real tokens from the
attention mask instead of assuming left or right padding.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def last_token_pool(hidden: Tensor, mask: Tensor) -> Tensor:
    """Hidden state of the last non-padding token (decoder-only embedders, e.g. Qwen3-Embedding)."""
    import torch

    seq_len = mask.shape[1]
    # Index of the last 1 in each row, regardless of padding side.
    last = seq_len - 1 - mask.flip(dims=[1]).int().argmax(dim=1)
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


def first_token_pool(hidden: Tensor, mask: Tensor) -> Tensor:
    """Hidden state of the first non-padding token (``[CLS]`` pooling for BERT-style encoders)."""
    import torch

    first = mask.int().argmax(dim=1)
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), first]


def mean_pool(hidden: Tensor, mask: Tensor) -> Tensor:
    """Average of non-padding token states."""
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)


POOLERS: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
    "last_token": last_token_pool,
    "cls": first_token_pool,
    "mean": mean_pool,
}


def get_pooler(name: str) -> Callable[[Tensor, Tensor], Tensor]:
    try:
        return POOLERS[name]
    except KeyError:
        raise ValueError(f"unknown pooling {name!r}; choose from {sorted(POOLERS)}") from None
