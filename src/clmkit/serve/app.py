"""REST API (``pip install "clmkit[serve]"``).

Endpoints:

* ``POST /v1/embeddings`` - **OpenAI-compatible**, so existing SDKs/tools work unchanged.
  Extra optional fields: ``input_type`` (``"query"``/``"document"``) and ``instruction``.
* ``POST /v1/search`` - semantic search over the loaded retriever.
* ``POST /v1/documents`` / ``DELETE /v1/documents`` - only when ``allow_writes=True``.
* ``POST /v1/rerank`` - rerank candidate documents (if a reranker is configured).
* ``GET /health``, ``GET /v1/models``.

Security defaults: optional bearer-token auth (constant-time compare), request size
limits, writes disabled unless explicitly enabled, and binding to 127.0.0.1 in the CLI.
"""

# NOTE: no `from __future__ import annotations` here - FastAPI must resolve the
# request-model annotations of the endpoints, which are local to create_app().
import base64
import hmac
import threading
from typing import TYPE_CHECKING, Any, Literal

from clmkit.utils import require

if TYPE_CHECKING:
    from fastapi import FastAPI

    from clmkit.encoders.base import Encoder
    from clmkit.rerank import Reranker
    from clmkit.retrieval import Retriever


def create_app(
    encoder: "Encoder",
    *,
    retriever: "Retriever | None" = None,
    reranker: "Reranker | None" = None,
    api_key: str | None = None,
    allow_writes: bool = False,
    max_batch: int = 256,
    max_chars: int = 32_768,
    max_k: int = 100,
    model_name: str | None = None,
) -> "FastAPI":
    fastapi = require("fastapi")
    from fastapi import Depends, HTTPException, Request
    from pydantic import BaseModel, Field

    if retriever is not None and retriever.encoder is not encoder:
        raise ValueError("retriever must use the same encoder instance as the app")
    model_id = model_name or encoder.name
    lock = threading.Lock()  # torch modules are not guaranteed thread-safe for concurrent forward
    app = fastapi.FastAPI(title="clmkit", version=_version(), docs_url="/docs", redoc_url=None)

    def auth(request: Request) -> None:
        if api_key is None:
            return
        header = request.headers.get("authorization", "")
        token = header[7:] if header.lower().startswith("bearer ") else ""
        if not hmac.compare_digest(token.encode(), api_key.encode()):
            raise HTTPException(
                status_code=401, detail="invalid or missing API key", headers={"WWW-Authenticate": "Bearer"}
            )

    def check_texts(texts: list[str]) -> None:
        if not texts:
            raise HTTPException(422, "input must not be empty")
        if len(texts) > max_batch:
            raise HTTPException(413, f"at most {max_batch} inputs per request")
        if any(len(t) > max_chars for t in texts):
            raise HTTPException(413, f"each input must be at most {max_chars} characters")

    class EmbeddingRequest(BaseModel):
        input: str | list[str]
        model: str | None = None
        encoding_format: Literal["float", "base64"] = "float"
        dimensions: int | None = Field(default=None, ge=1)
        input_type: Literal["query", "document"] = "document"
        instruction: str | None = Field(default=None, max_length=2000)
        user: str | None = None

    class SearchRequest(BaseModel):
        query: str = Field(min_length=1, max_length=max_chars)
        k: int = Field(default=5, ge=1, le=max_k)
        filter: dict[str, Any] | None = None
        instruction: str | None = Field(default=None, max_length=2000)
        rerank: bool | None = None

    class AddRequest(BaseModel):
        texts: list[str]
        ids: list[str] | None = None
        metadata: list[dict[str, Any]] | None = None

    class DeleteRequest(BaseModel):
        ids: list[str] = Field(min_length=1, max_length=10_000)

    class RerankRequest(BaseModel):
        query: str = Field(min_length=1, max_length=max_chars)
        documents: list[str]
        top_n: int | None = Field(default=None, ge=1)
        instruction: str | None = Field(default=None, max_length=2000)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": model_id,
            "dim": encoder.dim,
            "documents": len(retriever) if retriever is not None else None,
            "reranker": getattr(reranker, "name", None),
        }

    @app.get("/v1/models", dependencies=[Depends(auth)])
    def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "clmkit"}]}

    @app.post("/v1/embeddings", dependencies=[Depends(auth)])
    def embeddings(req: EmbeddingRequest) -> dict[str, Any]:
        texts = [req.input] if isinstance(req.input, str) else req.input
        check_texts(texts)
        if req.dimensions is not None and req.dimensions > encoder.native_dim:
            raise HTTPException(422, f"dimensions must be <= {encoder.native_dim}")
        with lock:
            vecs = encoder.encode(texts, kind=req.input_type, instruction=req.instruction, dim=req.dimensions)
        data = []
        for i, v in enumerate(vecs):
            emb: Any = (
                base64.b64encode(v.astype("<f4").tobytes()).decode("ascii")
                if req.encoding_format == "base64"
                else v.tolist()
            )
            data.append({"object": "embedding", "index": i, "embedding": emb})
        approx_tokens = sum(max(1, len(t) // 4) for t in texts)  # rough; no tokenizer dependency
        return {
            "object": "list",
            "data": data,
            "model": model_id,
            "usage": {"prompt_tokens": approx_tokens, "total_tokens": approx_tokens},
        }

    def need_retriever() -> "Retriever":
        if retriever is None:
            raise HTTPException(404, "no index loaded (start the server with --index)")
        return retriever

    @app.post("/v1/search", dependencies=[Depends(auth)])
    def search(req: SearchRequest) -> dict[str, Any]:
        r = need_retriever()
        if req.rerank and r.reranker is None:
            raise HTTPException(422, "no reranker configured")
        with lock:
            hits = r.search(req.query, k=req.k, filter=req.filter, instruction=req.instruction, rerank=req.rerank)
        return {"object": "list", "data": [h.to_dict() for h in hits]}

    @app.post("/v1/documents", dependencies=[Depends(auth)])
    def add_documents(req: AddRequest) -> dict[str, Any]:
        if not allow_writes:
            raise HTTPException(403, "writes are disabled (start the server with --allow-writes)")
        r = need_retriever()
        check_texts(req.texts)
        try:
            with lock:
                ids = r.add(req.texts, ids=req.ids, metadata=req.metadata)
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"ids": ids, "count": len(r)}

    @app.delete("/v1/documents", dependencies=[Depends(auth)])
    def delete_documents(req: DeleteRequest) -> dict[str, Any]:
        if not allow_writes:
            raise HTTPException(403, "writes are disabled (start the server with --allow-writes)")
        r = need_retriever()
        with lock:
            removed = r.delete(req.ids)
        return {"deleted": removed, "count": len(r)}

    @app.post("/v1/rerank", dependencies=[Depends(auth)])
    def rerank(req: RerankRequest) -> dict[str, Any]:
        if reranker is None:
            raise HTTPException(404, "no reranker configured (start the server with --reranker)")
        check_texts(req.documents)
        with lock:
            scores = reranker.score(req.query, req.documents, instruction=req.instruction)
        order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        if req.top_n:
            order = order[: req.top_n]
        return {
            "model": getattr(reranker, "name", "reranker"),
            "results": [{"index": i, "relevance_score": float(scores[i]), "document": req.documents[i]} for i in order],
        }

    return app


def _version() -> str:
    from clmkit import __version__

    return __version__
