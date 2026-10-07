"""Small, dependency-free helpers shared across clmkit."""

from __future__ import annotations

import importlib
import logging
import os
import random
import re
from collections.abc import Iterator, Sequence
from types import ModuleType
from typing import Any, TypeVar

import numpy as np

T = TypeVar("T")

logger = logging.getLogger("clmkit")

# Maps an importable module to the pip extra that provides it, for friendly errors.
_EXTRAS = {
    "torch": "hf",
    "transformers": "hf",
    "peft": "train",
    "fastapi": "serve",
    "uvicorn": "serve",
    "faiss": "faiss",
    "yaml": "yaml",
    "sklearn": "sklearn",
    "mcp": "mcp",
}


class MissingDependencyError(ImportError):
    """Raised when an optional backend is used without its extra installed."""


def require(module: str) -> ModuleType:
    """Import an optional dependency or raise an actionable error.

    >>> np_mod = require("numpy")
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        root = module.split(".")[0]
        extra = _EXTRAS.get(root)
        hint = (
            f' From a clmkit checkout run: python -m pip install ".[{extra}]".'
            " For Git/release-wheel installs, use the same version with the extra;"
            " see https://github.com/ZouhairMudakka/clmkit#install."
            if extra
            else ""
        )
        raise MissingDependencyError(f"clmkit needs the optional dependency '{root}'.{hint}") from exc


def is_available(module: str) -> bool:
    """Return True if ``module`` can be imported (without importing heavy submodules twice)."""
    try:
        importlib.import_module(module)
    except ImportError:
        return False
    return True


def batched(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    """Yield consecutive slices of ``items`` of length ``size`` (the last may be shorter)."""
    if size < 1:
        raise ValueError(f"batch size must be >= 1, got {size}")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation that is safe for all-zero rows."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        with np.errstate(over="ignore"):
            norm = float(np.linalg.norm(x))
        if np.isinf(norm) and np.isfinite(x).all():
            norm = float(np.linalg.norm(x.astype(np.float64)))
            return np.divide(x, max(norm, eps), out=np.empty_like(x), dtype=np.float64)
        return np.divide(x, max(norm, eps), out=np.empty_like(x))
    with np.errstate(over="ignore"):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
    overflow = np.isinf(norms[:, 0])
    if overflow.any():
        # Keep the usual float32 path cheap; only exceptional rows need a wider
        # accumulator. Finite float32 components can overflow when squared.
        norms = norms.astype(np.float64)
        norms[overflow] = np.linalg.norm(x[overflow].astype(np.float64), axis=1, keepdims=True)
    return np.divide(x, np.maximum(norms, eps), out=np.empty_like(x))


def truncate_dims(x: np.ndarray, dim: int | None) -> np.ndarray:
    """Matryoshka-style truncation to the first ``dim`` components (no renormalisation)."""
    if dim is None:
        return x
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    if dim > x.shape[-1]:
        raise ValueError(f"requested dim {dim} exceeds embedding size {x.shape[-1]}")
    return x[..., :dim]


def set_seed(seed: int) -> None:
    """Seed python, numpy and (if installed) torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if is_available("torch"):
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - no GPU in CI
            torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None = None) -> str:
    """Resolve ``"auto"``/``None`` to the best available torch device string."""
    if device not in (None, "auto"):
        return str(device)
    torch = require("torch")
    if torch.cuda.is_available():  # pragma: no cover - no GPU in CI
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():  # pragma: no cover
        return "mps"
    return "cpu"


def version_tuple(version: str) -> tuple[int, ...]:
    """Parse the numeric prefix of a version string: ``"4.51.3.dev0" -> (4, 51, 3)``."""
    return tuple(int(p) for p in re.findall(r"\d+", version)[:3])


def parse_value(raw: str) -> Any:
    """Parse a CLI ``key=value`` value: JSON literal when possible, else the raw string."""
    import json

    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw
