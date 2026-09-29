"""LangChain adapter: ``ClmkitEmbeddings`` implements ``langchain_core.embeddings.Embeddings``.

Works without LangChain installed (duck-typed), and subclasses the real base class
when ``langchain-core`` is available so ``isinstance`` checks pass::

    from clmkit.integrations.langchain import ClmkitEmbeddings
    emb = ClmkitEmbeddings("Qwen/Qwen3-Embedding-0.6B")
    vectorstore = FAISS.from_texts(texts, emb)   # any LangChain vector store
"""

from __future__ import annotations

from typing import Any

from clmkit.encoders import Encoder, load_encoder

try:  # pragma: no cover - depends on the environment
    from langchain_core.embeddings import Embeddings as _Base
except ImportError:  # pragma: no cover
    _Base = object


class ClmkitEmbeddings(_Base):
    def __init__(
        self,
        encoder: Encoder | str | dict[str, Any] = "hashing",
        *,
        instruction: str | None = None,
        batch_size: int = 32,
        **encoder_kwargs: Any,
    ) -> None:
        self.encoder = load_encoder(encoder, **encoder_kwargs)
        self.instruction = instruction
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.encoder.encode(list(texts), kind="document", batch_size=self.batch_size).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.encoder.encode(text, kind="query", instruction=self.instruction).tolist()
