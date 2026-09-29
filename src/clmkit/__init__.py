"""clmkit - a skeleton framework for Contrastive Language Models.

The top-level namespace only imports numpy-based modules; torch/transformers
backends load lazily when you build them (``load_encoder("Qwen/Qwen3-Embedding-8B")``).
"""

from clmkit.encoders import Encoder, HashingEncoder, load_encoder
from clmkit.index import NumpyIndex, VectorIndex, load_index
from clmkit.registry import ENCODERS, INDEXES, LOSSES, RERANKERS, Registry
from clmkit.retrieval import Retriever
from clmkit.text import chunk_text
from clmkit.types import SearchHit

__version__ = "0.1.0"

__all__ = [
    "ENCODERS",
    "INDEXES",
    "LOSSES",
    "RERANKERS",
    "Encoder",
    "HashingEncoder",
    "NumpyIndex",
    "Registry",
    "Retriever",
    "SearchHit",
    "VectorIndex",
    "__version__",
    "chunk_text",
    "load_encoder",
    "load_index",
]
