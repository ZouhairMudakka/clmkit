"""Small real-package integrations and owned-loopback transport probes."""
import asyncio
import contextlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from clmkit import HashingEncoder
from clmkit.encoders.openai_compat import OpenAICompatibleEncoder, RemoteEncoderError
from run_protocol import case, RESULTS, fixture_server

HERE = Path(__file__).resolve().parent


def langchain_cases():
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import InMemoryVectorStore
    from clmkit.integrations.langchain import ClmkitEmbeddings
    encoder = HashingEncoder(dim=128)
    adapter = ClmkitEmbeddings(encoder)

    def base():
        assert isinstance(adapter, Embeddings)
        return metadata.version("langchain-core")

    def direct():
        texts = ["cats eat fish", "stars orbit galaxies"]
        np.testing.assert_allclose(adapter.embed_documents(texts), encoder.encode(texts))
        np.testing.assert_allclose(adapter.embed_query(texts[0]), encoder.encode(texts[0], kind="query"))

    def asynchronous():
        async def check():
            docs = await adapter.aembed_documents(["cats", "stars"])
            query = await adapter.aembed_query("cats")
            np.testing.assert_allclose(docs, adapter.embed_documents(["cats", "stars"]))
            np.testing.assert_allclose(query, adapter.embed_query("cats"))
        asyncio.run(check())

    def vectorstore():
        store = InMemoryVectorStore(adapter)
        store.add_texts(["cats eat fish", "stars orbit galaxies"], ids=["cat", "star"])
        hits = store.similarity_search("cats eat fish", k=1)
        assert hits[0].page_content == "cats eat fish"
        store.delete(["cat"])
        assert store.similarity_search("cats eat fish", k=1)[0].page_content == "stars orbit galaxies"

    for name, fn in [("actual_langchain_base", base), ("actual_langchain_vectors", direct),
                     ("actual_langchain_async", asynchronous), ("actual_langchain_vectorstore", vectorstore)]:
        case(name, fn)


@contextlib.contextmanager
def delayed_server(delays):
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            index = len(calls)
            calls.append(time.monotonic())
            time.sleep(delays[min(index, len(delays)-1)])
            body = json.dumps({"data": [{"index": 0, "embedding": [1, 0]}]}).encode()
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)
        assert not thread.is_alive()


def timeout_retry():
    with delayed_server([0.6, 0]) as (url, calls):
        enc = OpenAICompatibleEncoder("fixture", base_url=url, native_dim=2, timeout=0.2, max_retries=1, backoff=0)
        np.testing.assert_allclose(enc.encode(["cats"]), [[1, 0]])
        assert len(calls) == 2, calls


def timeout_exhaustion():
    with delayed_server([0.6]) as (url, calls):
        enc = OpenAICompatibleEncoder("fixture", base_url=url, native_dim=2, timeout=0.2, max_retries=1, backoff=0)
        start = time.monotonic()
        try:
            enc.encode(["cats"])
        except RemoteEncoderError:
            assert time.monotonic() - start < 5
            assert len(calls) == 2, calls
        else:
            raise AssertionError("timeout was accepted")


def malformed_indices(indices):
    body = {"data": [{"index": i, "embedding": v} for i, v in zip(indices, [[1, 0], [0, 1]])]}
    with fixture_server([(200, body, {})]) as (url, calls):
        enc = OpenAICompatibleEncoder("fixture", base_url=url, native_dim=2, max_retries=0)
        try:
            enc.encode(["cat", "star"])
        except RemoteEncoderError:
            return "rejected"
        else:
            raise AssertionError("invalid indices accepted")


if __name__ == "__main__":
    langchain_cases()
    case("timeout_retries_then_succeeds", timeout_retry)
    case("timeout_exhaustion_is_bounded", timeout_exhaustion)
    for name, indices in [("boolean", [False, True]), ("float", [0.0, 1.0]),
                          ("string", ["0", "1"]), ("negative", [-1, 0])]:
        case("rejects_" + name + "_indices", lambda indices=indices: malformed_indices(indices))
    result = {"python": sys.version, "versions": {p: metadata.version(p) for p in
              ["openai", "httpx", "langchain-core", "mcp", "numpy"]},
              "environment": os.environ.get("CLMKIT_VALIDATION_ENV", "see runner provenance; package versions recorded"),
              "results": RESULTS, "passed": sum(r["status"] == "passed" for r in RESULTS),
              "failed": sum(r["status"] == "failed" for r in RESULTS), "torch_imported": "torch" in sys.modules}
    (HERE / "additional-results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["passed", "failed", "torch_imported"]}))
    raise SystemExit(bool(result["failed"]))
