from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from clmkit import HashingEncoder, Retriever
from clmkit.rerank import EncoderReranker

# ------------------------------------------------------------------- REST ---
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from clmkit.serve.app import create_app  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    enc = HashingEncoder(dim=64)
    r = Retriever(enc)
    r.add(["cats eat fish", "the sun is a star"], ids=["cat", "sun"], metadata=[{"t": "animal"}, {"t": "space"}])
    app = create_app(
        enc,
        retriever=r,
        reranker=EncoderReranker(HashingEncoder(dim=128)),
        allow_writes=True,
        max_batch=4,
        max_chars=100,
    )
    return TestClient(app)


def test_openai_compatible_embeddings(client: TestClient) -> None:
    res = client.post("/v1/embeddings", json={"input": ["hello", "world"], "model": "x"})
    assert res.status_code == 200
    body = res.json()
    assert body["object"] == "list" and len(body["data"]) == 2
    assert len(body["data"][0]["embedding"]) == 64 and body["usage"]["total_tokens"] >= 2
    one = client.post("/v1/embeddings", json={"input": "hello", "dimensions": 16}).json()
    assert len(one["data"][0]["embedding"]) == 16
    b64 = client.post("/v1/embeddings", json={"input": "hello", "encoding_format": "base64"}).json()
    vec = np.frombuffer(base64.b64decode(b64["data"][0]["embedding"]), dtype="<f4")
    np.testing.assert_allclose(vec, body["data"][0]["embedding"], rtol=1e-6)


def test_limits_and_validation(client: TestClient) -> None:
    assert client.post("/v1/embeddings", json={"input": ["a"] * 5}).status_code == 413
    assert client.post("/v1/embeddings", json={"input": "x" * 101}).status_code == 413
    assert client.post("/v1/embeddings", json={"input": []}).status_code == 422
    assert client.post("/v1/embeddings", json={"input": "a", "dimensions": 9999}).status_code == 422
    assert client.post("/v1/embeddings", json={"input": "a", "input_type": "bogus"}).status_code == 422
    assert client.post("/v1/search", json={"query": "x", "k": 1000}).status_code == 422


def test_search_documents_rerank(client: TestClient) -> None:
    hits = client.post("/v1/search", json={"query": "what do cats eat", "k": 1}).json()["data"]
    assert hits[0]["id"] == "cat" and hits[0]["metadata"] == {"t": "animal"}
    assert client.post("/v1/search", json={"query": "cats", "filter": {"t": "space"}}).json()["data"][0]["id"] == "sun"
    added = client.post("/v1/documents", json={"texts": ["penguins swim"], "ids": ["p"]}).json()
    assert added == {"ids": ["p"], "count": 3}
    assert client.post("/v1/documents", json={"texts": ["dup"], "ids": ["p"]}).status_code == 422
    assert client.request("DELETE", "/v1/documents", json={"ids": ["p"]}).json() == {"deleted": 1, "count": 2}
    rr = client.post("/v1/rerank", json={"query": "cats", "documents": ["the sun", "cats eat fish"], "top_n": 1})
    assert rr.json()["results"][0]["index"] == 1
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["documents"] == 2 and health["dim"] == 64


def test_auth_and_write_protection() -> None:
    enc = HashingEncoder(dim=32)
    client = TestClient(create_app(enc, retriever=Retriever(enc), api_key="s3cret"))
    assert client.post("/v1/embeddings", json={"input": "x"}).status_code == 401
    bad = client.post("/v1/embeddings", json={"input": "x"}, headers={"Authorization": "Bearer nope"})
    assert bad.status_code == 401
    ok = client.post("/v1/embeddings", json={"input": "x"}, headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200
    assert client.get("/health").status_code == 200  # liveness stays public
    headers = {"Authorization": "Bearer s3cret"}
    assert client.post("/v1/documents", json={"texts": ["x"]}, headers=headers).status_code == 403
    assert client.post("/v1/rerank", json={"query": "q", "documents": ["d"]}, headers=headers).status_code == 404
    assert client.get("/v1/models", headers=headers).json()["data"][0]["id"] == enc.name


def test_no_index_and_mismatched_retriever() -> None:
    enc = HashingEncoder(dim=32)
    client = TestClient(create_app(enc))
    assert client.post("/v1/search", json={"query": "x"}).status_code == 404
    with pytest.raises(ValueError, match="same encoder"):
        create_app(enc, retriever=Retriever(HashingEncoder(dim=32)))


# -------------------------------------------------------------------- MCP ---
def _mcp_request(server, method: str, params):  # type: ignore[no-untyped-def]
    """Invoke a registered handler directly, for both MCP SDK generations."""
    from mcp import types

    if hasattr(server, "get_request_handler"):  # mcp >= 2: handler(ctx, params) -> result
        return server.get_request_handler(method).handler(None, params)
    req_cls = {"tools/list": types.ListToolsRequest, "tools/call": types.CallToolRequest}[method]
    req = req_cls(method=method, params=params) if params is not None else req_cls(method=method)

    async def unwrap():  # type: ignore[no-untyped-def]
        return (await server.request_handlers[req_cls](req)).root  # mcp 1.x wraps in ServerResult

    return unwrap()


def test_mcp_server_lists_and_calls_tools() -> None:
    pytest.importorskip("mcp")
    import anyio
    from mcp import types

    from clmkit.agents import retriever_tools
    from clmkit.serve.mcp_server import build_server

    r = Retriever(HashingEncoder(dim=32))
    r.add(["cats eat fish"], ids=["cat"])
    server = build_server(retriever_tools(r))

    async def go() -> tuple:
        tools = await _mcp_request(server, "tools/list", None)
        ok = await _mcp_request(
            server, "tools/call", types.CallToolRequestParams(name="knowledge_search", arguments={"query": "cats"})
        )
        bad = await _mcp_request(
            server, "tools/call", types.CallToolRequestParams(name="knowledge_search", arguments={})
        )
        return tools, ok, bad

    tools, ok, bad = anyio.run(go)
    assert [t.name for t in tools.tools] == ["knowledge_search"]
    assert tools.tools[0].model_dump(by_alias=True)["inputSchema"]["required"] == ["query"]
    assert json.loads(ok.content[0].text)[0]["id"] == "cat"
    assert not ok.model_dump(by_alias=True)["isError"]
    assert "query" in bad.content[0].text  # our validator or the SDK's own schema check (mcp 1.x)
    assert bad.model_dump(by_alias=True)["isError"] is True
