"""Auxiliary closed-loop HTTP measurement; run only in the authorized CPU Codespace.

Run after all relevance/training jobs, for example under an external
``timeout 630s python validation/benchmark_serving.py --output serving.json``.
The 600-second internal deadline kills only this benchmark's own server child.
This measures one short synthetic query, not production capacity or an SLA.
Default closed-loop client concurrency levels are 1, 4, 8 and 16.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_ID = f"{MODEL}@{REVISION}"
QUERY = "How can I replace a lost payment card?"
DIMENSION = 384
CONCURRENCY_LEVELS = (1, 4, 8, 16)
THREAD_ENV = dict.fromkeys(
    (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ),
    "4",
)


def require_codespaces():
    if os.environ.get("CODESPACES", "").lower() != "true" or sys.platform != "linux":
        raise RuntimeError("Real serving measurements require the authorized Linux Codespace")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance():
    root = Path(__file__).resolve().parents[1]
    sources = {p.relative_to(root).as_posix(): digest(p) for p in sorted((root / "src" / "clmkit").rglob("*.py"))}
    return {
        "benchmark_sha256": digest(__file__),
        "core_source_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
        "product_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),  # noqa: S607
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "torch", "transformers", "fastapi", "starlette", "uvicorn", "httpx")
        },
    }


def validate_embedding(body, reference=None):
    try:
        rows = body["data"]
        if body["object"] != "list" or body["model"] != MODEL_ID or len(rows) != 1:
            raise ValueError("Invalid embedding response envelope")
        vector = rows[0]["embedding"]
        if rows[0]["object"] != "embedding" or rows[0]["index"] != 0 or len(vector) != DIMENSION:
            raise ValueError("Invalid embedding shape or index")
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in vector):
            raise ValueError("Nonfinite or nonnumeric embedding")
        if not isinstance(body["usage"]["total_tokens"], int) or body["usage"]["total_tokens"] < 1:
            raise ValueError("Invalid usage schema")
        if reference is not None and any(
            not math.isclose(x, y, rel_tol=1e-5, abs_tol=1e-6) for x, y in zip(vector, reference, strict=True)
        ):
            raise ValueError("Concurrent response differs from warm reference")
        return vector
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("Malformed embedding response") from error


def summarize_requests(records, *, concurrency, requested, elapsed):
    import numpy as np

    successes = [row["seconds"] for row in records if row["error"] is None]
    all_latencies = [row["seconds"] for row in records]
    errors = Counter(row["error"] for row in records if row["error"] is not None)

    def quantile(values, q):
        return float(np.quantile(values, q)) if values else None

    return {
        "concurrency": concurrency,
        "requested": requested,
        "completed": len(records),
        "not_started": requested - len(records),
        "successful": len(successes),
        "failed": len(records) - len(successes),
        "error_counts": dict(errors),
        "elapsed_seconds": elapsed,
        "successful_requests_per_second": len(successes) / elapsed if elapsed > 0 else None,
        "all_request_p50_seconds": quantile(all_latencies, 0.5),
        "all_request_p95_seconds": quantile(all_latencies, 0.95),
        "successful_request_p50_seconds": quantile(successes, 0.5),
        "successful_request_p95_seconds": quantile(successes, 0.95),
    }


def measure(base_url, token, reference, *, concurrency, requests, deadline, client_factory=None):
    """Each persistent client waits for its response before its next request."""
    import httpx

    client_factory = client_factory or httpx.Client
    barrier = threading.Barrier(concurrency)
    clients = [
        client_factory(base_url=base_url, headers={"Authorization": f"Bearer {token}"}, timeout=10.0, trust_env=False)
        for _ in range(concurrency)
    ]

    def worker(index):
        rows = []
        barrier.wait(timeout=10)
        for _ in range(index, requests, concurrency):
            if time.monotonic() >= deadline:
                break
            started = time.perf_counter()
            error = None
            try:
                response = clients[index].post("/v1/embeddings", json={"input": QUERY, "input_type": "query"})
                response.raise_for_status()
                validate_embedding(response.json(), reference)
            except Exception as exc:
                error = type(exc).__name__
            rows.append({"seconds": time.perf_counter() - started, "error": error})
        return rows

    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(worker, range(concurrency)))
        elapsed = time.perf_counter() - started
    finally:
        for client in clients:
            client.close()
    return summarize_requests(
        [row for batch in results for row in batch], concurrency=concurrency, requested=requests, elapsed=elapsed
    )


def stop_child(process):
    """Never search for, signal by name, or stop any process other than our child."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def serve(fd, state_path):
    require_codespaces()
    import resource

    import anyio.to_thread
    import torch
    import uvicorn

    from clmkit import load_encoder
    from clmkit.serve.app import create_app

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    encoder = load_encoder(
        MODEL,
        revision=REVISION,
        model_kwargs={"local_files_only": True},
        tokenizer_kwargs={"local_files_only": True},
        device="cpu",
        dtype="float32",
        max_length=128,
    )
    load_seconds = time.perf_counter() - started
    app = create_app(
        encoder,
        api_key=os.environ["CLMKIT_SERVING_BENCHMARK_TOKEN"],
        model_name=MODEL_ID,
        allow_writes=False,
        max_batch=1,
    )

    @contextlib.asynccontextmanager
    async def lifespan(app):
        anyio.to_thread.current_default_thread_limiter().total_tokens = 4
        yield

    app.router.lifespan_context = lifespan
    bound = socket.socket(fileno=fd)
    if bound.getsockname()[0] != "127.0.0.1":
        raise ValueError("Serving benchmark must bind only IPv4 loopback")
    state = {
        "model_load_seconds": load_seconds,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "http_worker_thread_limit": 4,
        "max_length": encoder.max_length,
        "embedding_dimension": encoder.dim,
        "encoder_configuration": encoder.fingerprint_config(),
        "thread_environment": {name: os.environ.get(name) for name in THREAD_ENV},
    }
    try:
        from threadpoolctl import threadpool_info

        state["threadpool_info"] = threadpool_info()
    except ImportError:
        state["threadpool_info"] = None
    state_path.write_text(json.dumps(state), encoding="utf-8")
    try:
        uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", workers=1, log_level="warning", access_log=False)).run(
            sockets=[bound]
        )
    finally:
        state["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        state_path.write_text(json.dumps(state), encoding="utf-8")


def run_benchmark(output, *, requests=48, max_seconds=600):
    require_codespaces()
    if not 1 <= requests <= 64 or not 1 <= max_seconds <= 600:
        raise ValueError("requests must be 1..64 and max_seconds must be 1..600")
    if output.exists():
        raise FileExistsError("Refusing to overwrite serving evidence")
    import httpx

    report = {
        "status": "running",
        "scope": "auxiliary deployment measurement, separate from relevance matrix",
        "model": MODEL,
        "revision": REVISION,
        "max_length": 128,
        "device": "cpu",
        "dtype": "float32",
        "server_processes": 1,
        "torch_threads": 4,
        "http_worker_thread_limit": 4,
        "server_thread_environment_before_import": THREAD_ENV,
        "parent_thread_environment": {name: os.environ.get(name) for name in THREAD_ENV},
        "bind": "127.0.0.1",
        "auth": "ephemeral synthetic bearer; unauthenticated request checked",
        "request_input": "one fixed short synthetic query; text intentionally omitted",
        "query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(),
        "concurrency_order": list(CONCURRENCY_LEVELS),
        "requests_per_level": requests,
        "internal_deadline_seconds": max_seconds,
        "request_timeout_seconds": 10,
        "provenance": provenance(),
        "measurements": [],
        "limitations": [
            "Closed-loop clients, one outstanding request per client; no open-loop arrival schedule.",
            "Fixed ascending concurrency order, one repetition, one short query; no production SLA.",
            "create_app serializes encoder forward calls with its shared lock.",
            "Client and server share the Codespace; underlying host resources may be shared.",
            "Run after all training and relevance jobs; do not overlap these workloads.",
            "Model is loaded from the retained local cache; no download time is measured.",
        ],
    }
    token = secrets.token_urlsafe(32)
    env = {
        **os.environ,
        **THREAD_ENV,
        "CLMKIT_SERVING_BENCHMARK_TOKEN": token,
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    deadline = time.monotonic() + max_seconds
    started = time.perf_counter()
    process = timer = None
    with tempfile.TemporaryDirectory(prefix="clmkit-serving-") as temporary, socket.socket() as bound:
        state_path = Path(temporary) / "state.json"
        bound.bind(("127.0.0.1", 0))
        bound.listen(128)
        port = bound.getsockname()[1]
        report["port"] = port

        def interrupted(signum, frame):
            raise TimeoutError("Benchmark interrupted by external deadline")

        old_sigterm = signal.signal(signal.SIGTERM, interrupted)
        try:
            process = subprocess.Popen(  # noqa: S603 - current interpreter and this fixed benchmark script only
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--serve-fd",
                    str(bound.fileno()),
                    "--state-path",
                    str(state_path),
                ],
                env=env,
                pass_fds=(bound.fileno(),),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            bound.close()  # the child owns its inherited listening socket now
            timer = threading.Timer(max_seconds, lambda: process.kill() if process.poll() is None else None)
            timer.start()
            base_url = f"http://127.0.0.1:{port}"
            with httpx.Client(base_url=base_url, timeout=2, trust_env=False) as client:
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("Server exited before becoming ready")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Server failed to become ready before deadline")
                    try:
                        response = client.get("/health")
                        if response.status_code == 200 and response.json().get("model") == MODEL_ID:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(0.1)
                report["startup_until_health_seconds"] = time.perf_counter() - started
                denied = client.post("/v1/embeddings", json={"input": QUERY})
                if denied.status_code != 401:
                    raise ValueError("Unauthenticated embedding request was not rejected")
                client.headers["Authorization"] = f"Bearer {token}"
                warm_started = time.perf_counter()
                reference = None
                for _ in range(4):
                    response = client.post("/v1/embeddings", json={"input": QUERY, "input_type": "query"})
                    response.raise_for_status()
                    reference = validate_embedding(response.json(), reference)
                report["warmup_requests"] = 4
                report["warmup_seconds"] = time.perf_counter() - warm_started
            for concurrency in CONCURRENCY_LEVELS:
                report["measurements"].append(
                    measure(base_url, token, reference, concurrency=concurrency, requests=requests, deadline=deadline)
                )
            report["status"] = (
                "passed" if all(row["successful"] == requests for row in report["measurements"]) else "failed"
            )
        except Exception as exc:
            report.update(status="failed", error_type=type(exc).__name__)
        finally:
            signal.signal(signal.SIGTERM, old_sigterm)
            if timer is not None:
                timer.cancel()
            if process is not None:
                stop_child(process)
                report["server_returncode"] = process.poll()
            if state_path.exists():
                report["server"] = json.loads(state_path.read_text())
    report["total_elapsed_seconds"] = time.perf_counter() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--requests", type=int, default=48)
    parser.add_argument("--max-seconds", type=int, default=600)
    parser.add_argument("--serve-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--state-path", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.serve_fd is not None:
        serve(args.serve_fd, args.state_path)
        return
    if args.output is None:
        parser.error("--output is required")
    report = run_benchmark(args.output, requests=args.requests, max_seconds=args.max_seconds)
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
