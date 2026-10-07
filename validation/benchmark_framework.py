"""Offline CPU microbenchmarks; run unchanged before/after a framework change.

Example: python validation/benchmark_framework.py --label before
For a paired comparison on PowerShell, save the trusted baseline first:
git show 19421a5:src/clmkit/data.py | Set-Content .tmp-framework-baseline.py
python validation/benchmark_framework.py --label paired --baseline .tmp-framework-baseline.py
Times are repeated medians, not test assertions. Peak bytes are Python/NumPy
allocations tracked during one extra invocation, excluding imports and fixtures.
These small synthetic workloads do not measure embedding-model quality or GPUs.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import runpy
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from clmkit import HashingEncoder
from clmkit.data import ContrastiveExample, iter_batches
from clmkit.eval import RetrievalEvaluator
from clmkit.index.numpy_index import NumpyIndex


def measure(fn, repeats=3):
    expected = fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        assert fn() == expected
        samples.append(time.perf_counter() - start)
    tracemalloc.start()
    assert fn() == expected
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 128 * 1024**2, "benchmark allocation budget exceeded"
    return {
        "median_seconds": statistics.median(samples),
        "samples_seconds": samples,
        "traced_peak_bytes": peak,
        "result": expected,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--baseline", type=Path, help="Trusted saved data.py for paired same-process comparisons")
    args = parser.parse_args()
    results = {}
    for n in (1000, 2000, 4000):
        examples = [ContrastiveExample(f"q{i}", "shared positive") for i in range(n)]
        results[f"shared_positive_{n}"] = measure(
            lambda examples=examples: sum(len(batch) for batch in iter_batches(examples, 64, shuffle=False))
        )
    examples = [ContrastiveExample(f"q{i}", f"p{i}") for i in range(20_000)]
    results["distinct_20000"] = measure(lambda: sum(len(batch) for batch in iter_batches(examples, 64, shuffle=False)))
    rng = np.random.default_rng(42)
    index = NumpyIndex(64, query_chunk=16)
    index.add([str(i) for i in range(2000)], rng.standard_normal((2000, 64), dtype=np.float32))
    queries = rng.standard_normal((32, 64), dtype=np.float32)
    results["numpy_2000x64_32queries_k10"] = measure(lambda: len(index.search(queries, 10)))
    allowed = [str(i) for i in range(0, 2000, 10)]
    results["numpy_filter_10percent"] = measure(lambda: len(index.search(queries, 10, allowed_ids=allowed)))
    corpus = {str(i): f"document topic {i}" for i in range(128)}
    evaluator = RetrievalEvaluator(
        {str(i): corpus[str(i)] for i in range(16)}, corpus, {str(i): {str(i): 1} for i in range(16)}, ks=[1, 10]
    )
    encoder = HashingEncoder(dim=64)
    results["hashing_evaluator_128docs_16queries"] = measure(lambda: evaluator(encoder))
    if args.baseline:
        baseline = runpy.run_path(str(args.baseline))["iter_batches"]
        paired = {}
        for name, count, duplicate_every in [("shared", 4000, 1), ("unique", 20000, 0), ("mixed", 20000, 10)]:
            examples = [
                ContrastiveExample(f"q{i}", "shared" if duplicate_every and i % duplicate_every == 0 else f"p{i}")
                for i in range(count)
            ]
            samples = {"before": [], "after": []}
            for repetition in range(5):
                methods = [("before", baseline), ("after", iter_batches)]
                if repetition % 2:
                    methods.reverse()
                for method, batcher in methods:
                    start = time.perf_counter()
                    assert sum(len(batch) for batch in batcher(examples, 64, shuffle=False)) == count
                    samples[method].append(time.perf_counter() - start)
            paired[name] = {
                method: {"median_seconds": statistics.median(times), "samples_seconds": times}
                for method, times in samples.items()
            }
        results["paired_same_process"] = paired
    output = {
        "label": args.label,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "fixture_seed": 42,
        "repeats": 3,
        "paired_repeats": 5 if args.baseline else 0,
        "memory_scope": "tracemalloc incremental allocations; excludes imports/fixtures/process RSS",
        "results": results,
    }
    path = Path("validation-results/framework-audit") / f"performance-{args.label}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
