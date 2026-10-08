"""Offline HTTP transport and process mocks; no model or network downloads."""

from __future__ import annotations

import copy
import math
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validation import benchmark_serving as benchmark


def body():
    return {
        "object": "list",
        "model": benchmark.MODEL_ID,
        "data": [{"object": "embedding", "index": 0, "embedding": [1 / math.sqrt(384)] * 384}],
        "usage": {"total_tokens": 10},
    }


def test_cost_summary_keeps_failures_and_unstarted_requests_in_counts():
    records = [{"seconds": latency, "error": "ReadTimeout" if latency == 3 else None} for latency in (1, 2, 3, 4)]
    result = benchmark.summarize_requests(records, concurrency=4, requested=5, elapsed=10)
    assert (result["completed"], result["successful"], result["failed"], result["not_started"]) == (4, 3, 1, 1)
    assert result["all_request_p50_seconds"] == 2.5
    assert result["all_request_p95_seconds"] == pytest.approx(3.85)
    assert result["successful_request_p95_seconds"] == pytest.approx(3.8)
    assert result["successful_requests_per_second"] == 0.3
    assert result["error_counts"] == {"ReadTimeout": 1}


@pytest.mark.parametrize("corruption", ["dimension", "nan", "model", "index", "changed", "schema"])
def test_validation_rejects_corrupt_or_concurrently_changed_vectors(corruption):
    reference = benchmark.validate_embedding(body())
    response = copy.deepcopy(body())
    if corruption == "dimension":
        response["data"][0]["embedding"].pop()
    elif corruption == "nan":
        response["data"][0]["embedding"][0] = float("nan")
    elif corruption == "changed":
        response["data"][0]["embedding"][0] = 0.9
    elif corruption == "index":
        response["data"][0]["index"] = 2
    elif corruption == "model":
        response["model"] = "other"
    else:
        del response["data"]
    with pytest.raises(ValueError):
        benchmark.validate_embedding(response, reference)


@pytest.mark.parametrize("concurrency", [1, 4, 8, 16])
def test_closed_loop_transport_counts_http_and_correctness_failures_and_closes_clients(concurrency):
    httpx = pytest.importorskip("httpx")
    lock = threading.Lock()
    count, clients = 0, []

    def handler(request):
        nonlocal count
        assert request.headers["Authorization"] == "Bearer synthetic"
        with lock:
            count += 1
            index = count
        response = body()
        if index == 2:
            return httpx.Response(503, json={"error": "synthetic"})
        if index == 7:
            response["data"][0]["embedding"][0] = float("inf")
            # JSON normally prohibits infinity: deliberately malformed wire content.
            return httpx.Response(200, content=b'{"object":"invalid"}', headers={"content-type": "application/json"})
        return httpx.Response(200, json=response)

    def factory(**kwargs):
        client = httpx.Client(**kwargs, transport=httpx.MockTransport(handler))
        clients.append(client)
        return client

    result = benchmark.measure(
        "http://127.0.0.1:1234",
        "synthetic",
        benchmark.validate_embedding(body()),
        concurrency=concurrency,
        requests=16,
        deadline=time.monotonic() + 30,
        client_factory=factory,
    )
    assert result["completed"] == count == 16
    assert result["successful"] == 14
    assert result["error_counts"] == {"HTTPStatusError": 1, "ValueError": 1}
    assert len(clients) == concurrency and all(client.is_closed for client in clients)


def test_expired_deadline_does_not_start_requests():
    httpx = pytest.importorskip("httpx")

    def unexpected(request):
        pytest.fail("An expired benchmark must not send requests")

    result = benchmark.measure(
        "http://127.0.0.1:1234",
        "synthetic",
        benchmark.validate_embedding(body()),
        concurrency=4,
        requests=16,
        deadline=time.monotonic() - 1,
        client_factory=lambda **kw: httpx.Client(**kw, transport=httpx.MockTransport(unexpected)),
    )
    assert result["not_started"] == 16 and result["completed"] == 0
    assert result["all_request_p95_seconds"] is None


@pytest.mark.parametrize("state", ["exited", "graceful", "stuck"])
def test_cleanup_only_signals_the_owned_child_and_escalates_if_needed(state):
    calls = []

    class Child:
        def poll(self):
            return 0 if state == "exited" else None

        def terminate(self):
            calls.append("terminate")

        def kill(self):
            calls.append("kill")

        def wait(self, timeout):
            calls.append("wait")
            if state == "stuck" and "kill" not in calls:
                raise subprocess.TimeoutExpired("owned-child", timeout)

    benchmark.stop_child(Child())
    assert (
        calls
        == ({"exited": [], "graceful": ["terminate", "wait"], "stuck": ["terminate", "wait", "kill", "wait"]}[state])
    )


def test_remote_guard_prevents_local_process_or_download_activity(monkeypatch, tmp_path):
    monkeypatch.delenv("CODESPACES", raising=False)
    monkeypatch.setattr(benchmark.subprocess, "Popen", lambda *a, **kw: pytest.fail("must not spawn locally"))
    with pytest.raises(RuntimeError, match="authorized Linux Codespace"):
        benchmark.run_benchmark(tmp_path / "never.json")
    assert not (tmp_path / "never.json").exists()


def test_server_constructor_pins_revision_offline_loading_and_loopback_socket(monkeypatch, tmp_path):
    calls = {}
    # This constructor test exercises no real serving dependencies. In particular,
    # NumPy-only installed-wheel jobs do not install AnyIO transitively.
    fake_anyio = ModuleType("anyio")
    fake_to_thread = ModuleType("anyio.to_thread")
    fake_to_thread.current_default_thread_limiter = lambda: SimpleNamespace(total_tokens=4)
    fake_anyio.to_thread = fake_to_thread
    fake_torch = SimpleNamespace(
        set_num_threads=lambda n: calls.update(threads=n),
        set_num_interop_threads=lambda n: calls.update(interop=n),
        get_num_threads=lambda: 4,
        get_num_interop_threads=lambda: 1,
    )
    fake_encoder = SimpleNamespace(max_length=128, dim=384, fingerprint_config=lambda: {"pooling": "mean"})

    class Server:
        def __init__(self, config):
            calls["config"] = config

        def run(self, sockets):
            calls["sockets"] = sockets

    monkeypatch.setattr(benchmark, "require_codespaces", lambda: None)
    monkeypatch.setitem(sys.modules, "anyio", fake_anyio)
    monkeypatch.setitem(sys.modules, "anyio.to_thread", fake_to_thread)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules, "resource", SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda _: SimpleNamespace(ru_maxrss=2))
    )
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(Server=Server, Config=lambda app, **kw: kw))
    import clmkit
    import clmkit.serve.app

    monkeypatch.setattr(
        clmkit, "load_encoder", lambda model, **kw: calls.update(model=model, model_args=kw) or fake_encoder
    )
    monkeypatch.setattr(
        clmkit.serve.app,
        "create_app",
        lambda encoder, **kw: calls.update(app_args=kw) or SimpleNamespace(router=SimpleNamespace()),
    )
    bound = SimpleNamespace(getsockname=lambda: ("127.0.0.1", 9999))
    monkeypatch.setattr(benchmark.socket, "socket", lambda **kw: bound)
    monkeypatch.setenv("CLMKIT_SERVING_BENCHMARK_TOKEN", "test-secret")
    state = tmp_path / "state.json"
    benchmark.serve(42, state)
    assert calls["model_args"]["revision"] == benchmark.REVISION
    assert calls["model_args"]["model_kwargs"] == calls["model_args"]["tokenizer_kwargs"] == {"local_files_only": True}
    assert calls["model_args"]["max_length"] == 128
    assert calls["config"]["host"] == "127.0.0.1" and calls["config"]["workers"] == 1
    assert calls["app_args"]["allow_writes"] is False
    assert calls["sockets"] == [bound]
    assert "test-secret" not in state.read_text()
