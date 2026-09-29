"""Vector indexes. ``numpy`` is built in; ``faiss`` is optional."""

from clmkit.index.base import VectorIndex, load_index
from clmkit.index.numpy_index import NumpyIndex

__all__ = ["NumpyIndex", "VectorIndex", "load_index"]
