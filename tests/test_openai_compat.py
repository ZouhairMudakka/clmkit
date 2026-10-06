from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

import numpy as np
import pytest

from clmkit.encoders.openai_compat import OpenAICompatibleEncoder, RemoteEncoderError


class _Handler(BaseHTTPRequestHandler):
    calls: ClassVar[list[dict]] = []
    fail_first: ClassVar[int] = 0
    mode: ClassVar[str] = "ok"

    def log_message(self, *args: object) -> None:  # silence
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).calls.append({"body": body, "auth": self.headers.get("Authorization")})
        if type(self).fail_first > 0:
            type(self).fail_first -= 1
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b"busy")
            return
        if type(self).mode == "bad":
            payload = {"unexpected": True}
        else:
            dim = body.get("dimensions") or 6
            # reverse order on purpose: the client must sort by index
            payload = {
                "data": [{"index": i, "embedding": [float(i + 1)] * dim} for i in reversed(range(len(body["input"])))]
            }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def server() -> Iterator[str]:
    _Handler.calls = []
    _Handler.fail_first = 0
    _Handler.mode = "ok"
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    httpd.shutdown()


def test_batches_order_auth_and_templates(server: str) -> None:
    enc = OpenAICompatibleEncoder("Qwen/Qwen3-Embedding-8B", base_url=server, api_key="sk-test", max_batch=2)
    out = enc.encode(["a", "b", "c"], normalize=False)
    assert out.shape == (3, 6)
    np.testing.assert_array_equal(out[:, 0], [1, 2, 1])  # two requests: [a, b], [c]
    assert len(_Handler.calls) == 2
    assert all(c["auth"] == "Bearer sk-test" for c in _Handler.calls)
    # Qwen3 preset -> query instruction applied client-side, documents untouched
    enc.encode("where is paris", kind="query")
    sent = _Handler.calls[-1]["body"]["input"][0]
    assert sent.startswith("Instruct: ") and sent.endswith("\nQuery:where is paris")
    assert "sk-test" not in repr(enc)


def test_dimensions_and_dim_discovery(server: str) -> None:
    enc = OpenAICompatibleEncoder("m", base_url=server, dimensions=4)
    assert enc.native_dim == 4
    assert enc.encode("x").shape == (4,)
    assert _Handler.calls[-1]["body"]["dimensions"] == 4
    probe = OpenAICompatibleEncoder("m", base_url=server)
    assert probe.native_dim == 6  # discovered with one probe request


def test_retries_then_succeeds(server: str) -> None:
    _Handler.fail_first = 2
    enc = OpenAICompatibleEncoder("m", base_url=server, native_dim=6, backoff=0.01)
    assert enc.encode("x").shape == (6,)
    assert len(_Handler.calls) == 3


def test_gives_up_and_reports(server: str) -> None:
    _Handler.fail_first = 10
    enc = OpenAICompatibleEncoder("m", base_url=server, native_dim=6, max_retries=1, backoff=0.01)
    with pytest.raises(RemoteEncoderError, match="503"):
        enc.encode("x")


def test_malformed_payload(server: str) -> None:
    _Handler.mode = "bad"
    enc = OpenAICompatibleEncoder("m", base_url=server, native_dim=6)
    with pytest.raises(RemoteEncoderError, match="malformed"):
        enc.encode("x")


@pytest.mark.parametrize("indices", [[0, 0], [0, 2], [-1, 0], [False, True], [0.0, 1.0], ["0", "1"]])
def test_rejects_misaligned_response_indices(monkeypatch: pytest.MonkeyPatch, indices: list) -> None:
    enc = OpenAICompatibleEncoder("fixture", native_dim=2)
    body = {"data": [{"index": index, "embedding": [1, 0]} for index in indices]}
    monkeypatch.setattr(enc, "_post", lambda payload: body)
    with pytest.raises(RemoteEncoderError, match="malformed"):
        enc.encode(["first", "second"])


@pytest.mark.parametrize("vector", [[float("nan"), 0], [0, float("inf")], [], [1, 2, 3]])
def test_rejects_invalid_response_vectors(monkeypatch: pytest.MonkeyPatch, vector: list) -> None:
    enc = OpenAICompatibleEncoder("fixture", native_dim=2)
    monkeypatch.setattr(enc, "_post", lambda payload: {"data": [{"index": 0, "embedding": vector}]})
    with pytest.raises(RemoteEncoderError):
        enc.encode("text")


def test_discovered_dimension_must_hold_across_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    enc = OpenAICompatibleEncoder("fixture", max_batch=1)
    responses = iter(
        [
            {"data": [{"index": 0, "embedding": [1, 0]}]},
            {"data": [{"index": 0, "embedding": [1, 0, 0]}]},
        ]
    )
    monkeypatch.setattr(enc, "_post", lambda payload: next(responses))
    with pytest.raises(RemoteEncoderError, match="dimension"):
        enc.encode(["first", "second"])


def test_unreachable_and_bad_url() -> None:
    enc = OpenAICompatibleEncoder("m", base_url="http://127.0.0.1:9/v1", native_dim=2, max_retries=0, timeout=2)
    with pytest.raises(RemoteEncoderError, match="unreachable"):
        enc.encode("x")
    with pytest.raises(ValueError, match="http"):
        OpenAICompatibleEncoder("m", base_url="file:///etc/passwd")


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch, server: str) -> None:
    monkeypatch.setenv("MY_KEY", "from-env")
    enc = OpenAICompatibleEncoder("m", base_url=server, api_key_env="MY_KEY", native_dim=6)
    enc.encode("x")
    assert _Handler.calls[-1]["auth"] == "Bearer from-env"


class _Redirector(BaseHTTPRequestHandler):
    """Redirects every request to a second server that records what it receives."""

    target = ""

    def log_message(self, *args: object) -> None:
        pass

    def do_POST(self) -> None:
        self.send_response(307)
        self.send_header("Location", type(self).target)
        self.end_headers()


def test_redirects_are_refused_so_the_api_key_cannot_leak(server: str) -> None:
    _Redirector.target = server + "/embeddings"  # a *different* host:port than the one we trust
    httpd = HTTPServer(("127.0.0.1", 0), _Redirector)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        enc = OpenAICompatibleEncoder(
            "m", base_url=f"http://127.0.0.1:{httpd.server_address[1]}/v1", api_key="sk-secret", native_dim=6
        )
        with pytest.raises(RemoteEncoderError, match="redirect"):
            enc.encode("x")
        assert _Handler.calls == []  # the redirect target never saw the request (or the key)
    finally:
        httpd.shutdown()
