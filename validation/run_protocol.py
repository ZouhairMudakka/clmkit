"""Independent loopback-only protocol probes. No model or package downloads."""
import contextlib
import importlib.metadata as metadata
import json
import os
import platform
import socket
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from clmkit import HashingEncoder, Retriever
from clmkit.agents import retriever_tools
from clmkit.encoders.openai_compat import OpenAICompatibleEncoder, RemoteEncoderError
from clmkit.serve.app import create_app

HERE = Path(__file__).resolve().parent
RESULTS = []


def case(name, fn):
    started = time.perf_counter()
    try:
        result = dict(name=name, status="passed", detail=fn())
    except Exception as exc:
        result = dict(name=name, status="failed", error=repr(exc), traceback=traceback.format_exc())
    result["seconds"] = time.perf_counter() - started
    RESULTS.append(result)
    print(json.dumps(result), flush=True)


@contextlib.contextmanager
def local_app(app):
    import uvicorn
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            if time.monotonic() > deadline:
                raise TimeoutError("local ASGI startup")
            time.sleep(.02)
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        server.should_exit = True
        thread.join(10)
        sock.close()
        assert not thread.is_alive(), "ASGI cleanup"


def rest_probes():
    import httpx
    import openai
    enc = HashingEncoder(dim=16)
    r = Retriever(enc)
    r.add(["cats eat fish", "sun is a star"], ids=["cat", "sun"])
    with local_app(create_app(enc, retriever=r, api_key="fixture", max_batch=3, max_chars=30)) as url:
        with httpx.Client(base_url=url, timeout=5, trust_env=False) as client:
            auth = {"Authorization": "Bearer fixture"}
            def sdk_default():
                with openai.OpenAI(api_key="fixture", base_url=url + "/v1") as sdk:
                    assert len(sdk.embeddings.create(model=enc.name, input="cats").data[0].embedding) == 16
            case("sdk_default_constructor_and_embeddings", sdk_default)
            for encoding in [None, "float", "base64"]:
                def sdk_probe(encoding=encoding):
                    with openai.OpenAI(api_key="fixture", base_url=url + "/v1", http_client=httpx.Client(trust_env=False), max_retries=0) as sdk:
                        kwargs = {} if encoding is None else {"encoding_format": encoding}
                        result = sdk.embeddings.create(model=enc.name, input=["cats", "sun"], dimensions=8, **kwargs)
                        assert [x.index for x in result.data] == [0, 1]
                        if encoding == "base64":
                            import base64
                            vectors = [np.frombuffer(base64.b64decode(x.embedding), dtype="<f4") for x in result.data]
                        else:
                            vectors = [x.embedding for x in result.data]
                        np.testing.assert_allclose(vectors, enc.encode(["cats", "sun"], dim=8), rtol=1e-6, atol=1e-7)
                        return {"encoding": encoding, "dimension": len(vectors[0])}
                case(f"sdk_explicit_client_encoding_{encoding}", sdk_probe)
            for name, method, path, body in [
                ("models", "GET", "/v1/models", None),
                ("embeddings", "POST", "/v1/embeddings", {"input": "cats"}),
                ("search", "POST", "/v1/search", {"query": "cats"}),
                ("add", "POST", "/v1/documents", {"texts": ["cats"]}),
                ("delete", "DELETE", "/v1/documents", {"ids": ["cat"]}),
                ("rerank", "POST", "/v1/rerank", {"query": "cats", "documents": ["cats"]}),
            ]:
                def auth_probe(method=method, path=path, body=body):
                    statuses = [client.request(method, path, json=body, headers=h).status_code for h in [{}, {"Authorization": "Bearer wrong"}]]
                    assert statuses == [401, 401], statuses
                    return statuses
                case(f"auth_{name}", auth_probe)
            for method, body in [("POST", {"texts": ["dogs"]}), ("DELETE", {"ids": ["cat"]})]:
                def write_probe(method=method, body=body):
                    assert client.request(method, "/v1/documents", json=body, headers=auth).status_code == 403
                    assert len(r) == 2
                case(f"writes_disabled_{method}", write_probe)
            malformed = [
                ("empty_list", {"input": []}, 422), ("numeric_input", {"input": [1, 2]}, 422),
                ("null_input", {"input": None}, 422), ("dimension_zero", {"input": "a", "dimensions": 0}, 422),
                ("dimension_negative", {"input": "a", "dimensions": -1}, 422),
                ("dimension_exceeds_native", {"input": "a", "dimensions": 17}, 422),
                ("dimension_fraction", {"input": "a", "dimensions": 1.5}, 422),
                ("encoding_invalid", {"input": "a", "encoding_format": "bytes"}, 422),
                ("batch_limit", {"input": ["a"] * 4}, 413), ("text_limit", {"input": "a" * 31}, 413),
            ]
            for name, body, expected in malformed:
                def invalid(body=body, expected=expected):
                    response = client.post("/v1/embeddings", json=body, headers=auth)
                    assert response.status_code == expected, (response.status_code, response.text)
                    return response.status_code
                case(f"rest_{name}", invalid)
    with local_app(create_app(enc, retriever=r, allow_writes=True)) as url:
        with httpx.Client(base_url=url, trust_env=False, timeout=5) as client:
            def roundtrip():
                assert client.post("/v1/documents", json={"texts": ["dogs bark"], "ids": ["dog"]}).status_code == 200
                assert client.post("/v1/search", json={"query": "dogs bark", "k": 1}).json()["data"][0]["id"] == "dog"
                assert client.request("DELETE", "/v1/documents", json={"ids": ["dog"]}).json()["deleted"] == 1
                assert len(r) == 2
            case("writes_enabled_roundtrip", roundtrip)


@contextlib.contextmanager
def fixture_server(responses):
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            calls.append({"body": json.loads(self.rfile.read(int(self.headers["Content-Length"]))), "auth": self.headers.get("Authorization")})
            status, body, headers = responses[min(len(calls)-1, len(responses)-1)]
            if isinstance(body, dict):
                body = json.dumps(body).encode()
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def remote_probes():
    good = {"data": [{"index": 1, "embedding": [0., 2.]}, {"index": 0, "embedding": [3., 0.]}]}
    def ordering():
        with fixture_server([(200, good, {})]) as (url, calls):
            enc = OpenAICompatibleEncoder("fixture", base_url=url, native_dim=2, api_key="fixture-key")
            np.testing.assert_array_equal(enc.encode(["first", "second"], normalize=False), [[3., 0.], [0., 2.]])
            assert calls[0]["auth"] == "Bearer fixture-key"
    case("remote_reorders_indexed_response", ordering)
    for status in [408, 409, 429, 500, 502, 503, 504]:
        def retry(status=status):
            with fixture_server([(status, b"temporary", {}), (200, good, {})]) as (url, calls):
                enc = OpenAICompatibleEncoder("fixture", base_url=url, native_dim=2, backoff=0, max_retries=1)
                assert enc.encode(["a", "b"]).shape == (2, 2)
                assert len(calls) == 2
        case(f"remote_retry_{status}", retry)
    for status, expected in [(401, 1), (429, 3), (503, 3)]:
        def exhausted(status=status, expected=expected):
            with fixture_server([(status, b"fixture-error", {})]) as (url, calls):
                enc = OpenAICompatibleEncoder("fixture", base_url=url, native_dim=2, backoff=0, max_retries=2)
                try:
                    enc.encode(["a", "b"])
                except RemoteEncoderError as exc:
                    assert str(status) in str(exc)
                else:
                    raise AssertionError("expected RemoteEncoderError")
                assert len(calls) == expected, len(calls)
        case(f"remote_error_bound_{status}", exhausted)
    def redirects():
        with fixture_server([(200, good, {})]) as (target, target_calls):
            for status in [301, 302, 303, 307, 308]:
                with fixture_server([(status, b"", {"Location": target + "/embeddings"})]) as (url, calls):
                    enc = OpenAICompatibleEncoder("fixture", base_url=url, native_dim=2, api_key="fixture-secret", backoff=0)
                    try:
                        enc.encode(["a", "b"])
                    except RemoteEncoderError as exc:
                        assert "redirect" in str(exc)
                    else:
                        raise AssertionError("redirect followed")
                    assert len(calls) == 1
            assert not target_calls
    case("remote_redirects_refused_no_key_forward", redirects)
    malformed = {
        "invalid_json": b"not-json", "null_json": b"null", "missing_data": {},
        "duplicate_indices": {"data": [{"index": 0, "embedding": [1, 0]}, {"index": 0, "embedding": [0, 1]}]},
        "out_of_range_indices": {"data": [{"index": 5, "embedding": [1, 0]}, {"index": 6, "embedding": [0, 1]}]},
        "nan_vector": {"data": [{"index": 0, "embedding": [float("nan"), 0]}, {"index": 1, "embedding": [0, 1]}]},
        "infinity_vector": {"data": [{"index": 0, "embedding": [float("inf"), 0]}, {"index": 1, "embedding": [0, 1]}]},
        "wrong_dim": {"data": [{"index": 0, "embedding": [1]}, {"index": 1, "embedding": [2]}]},
    }
    for name, body in malformed.items():
        def invalid(body=body):
            with fixture_server([(200, body, {})]) as (url, calls):
                enc = OpenAICompatibleEncoder("fixture", base_url=url, native_dim=2, max_retries=0)
                try:
                    out = enc.encode(["a", "b"])
                except RemoteEncoderError:
                    return "rejected as RemoteEncoderError"
                else:
                    raise AssertionError(f"malformed remote output accepted: {out.tolist()}")
        case(f"remote_rejects_{name}", invalid)


def mcp_child():
    from clmkit.serve.mcp_server import run_stdio
    r = Retriever(HashingEncoder(dim=32))
    r.add(["cats eat fish"], ids=["cat"])
    run_stdio(retriever_tools(r))


def mcp_probe():
    import anyio
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    async def run():
        params = StdioServerParameters(command=sys.executable, args=[str(Path(__file__).resolve()), "--mcp-child"], env={**os.environ, "PYTHONUNBUFFERED": "1"})
        with anyio.fail_after(30):
            with (HERE / "mcp-stderr.log").open("w", encoding="utf-8") as errlog:
                async with stdio_client(params, errlog=errlog) as streams:
                    async with ClientSession(*streams) as session:
                        init = await session.initialize()
                        listed = await session.list_tools()
                        assert [t.name for t in listed.tools] == ["knowledge_search"]
                        found = await session.call_tool("knowledge_search", {"query": "cats"})
                        assert not found.model_dump(by_alias=True)["isError"]
                        assert json.loads(found.content[0].text)[0]["id"] == "cat"
                        for name, args in [("knowledge_search", {}), ("knowledge_search", {"query": "cats", "k": True}), ("knowledge_add", {"text": "bad"})]:
                            rejected = await session.call_tool(name, args)
                            assert rejected.model_dump(by_alias=True)["isError"]
                        return {"protocol_version": init.model_dump(by_alias=True)["protocolVersion"], "tools": [t.name for t in listed.tools]}
    return anyio.run(run)


def main():
    import openai
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), command=f'{sys.executable} {Path(__file__).resolve()}', python=sys.version, platform=platform.platform(), environment=os.environ.get("CLMKIT_VALIDATION_ENV", "see runner provenance; package versions recorded"), versions={p: metadata.version(p) for p in ["openai", "httpx", "fastapi", "starlette", "uvicorn", "mcp", "numpy", "anyio"]})
    manifest["loaded_openai"] = {"version": openai.__version__, "path": openai.__file__}
    print(json.dumps(manifest), flush=True)
    rest_probes()
    remote_probes()
    case("mcp_actual_stdio_initialize_list_call_errors", mcp_probe)
    manifest.update(finished_utc=datetime.now(timezone.utc).isoformat(), results=RESULTS, passed=sum(x["status"] == "passed" for x in RESULTS), failed=sum(x["status"] == "failed" for x in RESULTS), torch_imported="torch" in sys.modules)
    (HERE / "results.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ["passed", "failed", "torch_imported"]}), flush=True)
    return bool(manifest["failed"])


if __name__ == "__main__":
    if "--mcp-child" in sys.argv:
        mcp_child()
    else:
        raise SystemExit(main())
