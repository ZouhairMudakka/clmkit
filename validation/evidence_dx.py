"""Remote-only descriptive clmkit / Sentence Transformers workflow comparison.

Run with an outer 10-minute timeout. This is a bounded development fixture,
not a human productivity study or a relevance benchmark. Model weights and
prepared text stay in the Codespace; only the compact JSON report is exported.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import tempfile
import time
from pathlib import Path

import numpy as np

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PACKAGES = ("clmkit", "sentence-transformers", "transformers", "torch", "numpy", "safetensors")


def deterministic_sample(records: list[dict], count: int) -> list[dict]:
    """Hash-order IDs, then sort the chosen sample by ID for repeatability."""
    if count < 1:
        raise ValueError("sample count must be positive")
    selected = sorted(records, key=lambda row: (hashlib.sha256(row["id"].encode()).hexdigest(), row["id"]))[:count]
    return sorted(selected, key=lambda row: row["id"])


def normalized(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Expected a finite matrix")
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def exact_rank(
    documents: np.ndarray, queries: np.ndarray, ids: list[str], k: int, allowed: set[str] | None = None
) -> list[list[tuple[str, float]]]:
    """Independent cosine reference; every tie breaks on ID, including cutoff."""
    if len(ids) != len(documents) or len(set(ids)) != len(ids):
        raise ValueError("Vector IDs must be unique and match the matrix")
    if k < 1:
        raise ValueError("k must be positive")
    rows = [i for i, id_ in enumerate(ids) if allowed is None or id_ in allowed]
    scores = normalized(queries) @ normalized(documents).T
    return [
        [(ids[i], float(row[i])) for i in sorted(rows, key=lambda i: (-float(row[i]), ids[i]))[:k]] for row in scores
    ]


def rank_ids(rows: list) -> list[list[str]]:
    return [[item[0] for item in row] for row in rows]


def parity(left: np.ndarray, right: np.ndarray) -> dict:
    if left.shape != right.shape:
        return {"allclose": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    return {
        "allclose": bool(np.allclose(left, right, atol=2e-5, rtol=2e-4)),
        "max_absolute_difference": float(np.max(np.abs(left - right))) if left.size else 0.0,
        "atol": 2e-5,
        "rtol": 2e-4,
    }


def save_reference(path: Path, vectors: np.ndarray, documents: list[dict], config: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "vectors.npy", vectors, allow_pickle=False)
    (path / "manifest.json").write_text(json.dumps({"config": config, "documents": documents}), encoding="utf-8")


def load_reference(path: Path, expected_config: dict) -> tuple[np.ndarray, list[dict]]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest["config"] != expected_config:
        raise ValueError("Model or vector configuration mismatch; rebuild reference index")
    vectors = np.load(path / "vectors.npy", allow_pickle=False)
    documents = manifest["documents"]
    ids = [row["id"] for row in documents]
    if vectors.ndim != 2 or len(vectors) != len(ids) or len(set(ids)) != len(ids) or not np.isfinite(vectors).all():
        raise ValueError("Invalid reference snapshot")
    return vectors, documents


def run(
    data_dir: Path,
    *,
    revision: str,
    max_length: int = 128,
    document_count: int = 100,
    query_count: int = 20,
    repeats: int = 3,
    threads: int = 2,
) -> dict:
    if os.environ.get("CODESPACES", "").lower() != "true":
        raise RuntimeError("This model workflow runs only remotely with CODESPACES=true; no local downloads")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be an immutable 40-character model commit")
    if max_length not in {128, 256}:
        raise ValueError("max_length must be 128 or 256")
    if not (1 <= document_count <= 200 and 1 <= query_count <= 50 and 1 <= repeats <= 5 and 1 <= threads <= 4):
        raise ValueError("DX budget: <=200 documents, <=50 queries, <=5 repeats, <=4 threads")
    from validation.evidence_data import load_prepared

    prepared = load_prepared(data_dir, "dev")
    documents = deterministic_sample(prepared["corpus"], document_count)
    queries = deterministic_sample(prepared["queries"], query_count)
    if not documents or not queries:
        raise ValueError("DX comparison needs nonempty development queries and corpus")
    # This synthetic partition tests the API; it makes no dataset ownership claim.
    documents = [
        {**row, "metadata": {**row.get("metadata", {}), "dx_partition": i % 2}} for i, row in enumerate(documents)
    ]
    ids = [row["id"] for row in documents]
    texts = [row["text"] for row in documents]
    query_texts = [row["text"] for row in queries]
    metadata = [row["metadata"] for row in documents]
    config = {"model": MODEL, "revision": revision, "max_length": max_length, "pooling": "mean", "normalize": True}
    start = time.perf_counter()

    def check_budget() -> None:
        if time.perf_counter() - start > 540:
            raise TimeoutError("DX comparison exceeded the 540-second internal budget")

    import torch
    from sentence_transformers import SentenceTransformer

    from clmkit import Retriever, load_encoder

    torch.set_num_threads(threads)
    load_start = time.perf_counter()
    encoder = load_encoder(MODEL, revision=revision, device="cpu", dtype="float32", max_length=max_length)
    clmkit_load = time.perf_counter() - load_start
    check_budget()
    load_start = time.perf_counter()
    reference = SentenceTransformer(
        MODEL, revision=revision, device="cpu", trust_remote_code=False, model_kwargs={"use_safetensors": True}
    )
    reference.max_seq_length = max_length
    reference.float()
    reference.eval()
    direct_load = time.perf_counter() - load_start
    check_budget()
    k = min(10, len(ids))

    def reference_encode(items: list[str]) -> np.ndarray:
        return reference.encode(
            items, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )

    def clmkit_run():
        begin = time.perf_counter()
        retriever = Retriever(encoder, encoder_identity=revision)
        retriever.add(texts, ids=ids, metadata=metadata, batch_size=32)
        indexed = time.perf_counter()
        # Fetch all before ID sorting: existing NumpyIndex's partial selection
        # does not promise a deterministic tied cutoff. This cost is included.
        raw = retriever.search(query_texts, k=len(ids))
        ranks = [[(hit.id, hit.score) for hit in sorted(row, key=lambda hit: (-hit.score, hit.id))[:k]] for row in raw]
        end = time.perf_counter()
        return (
            retriever,
            ranks,
            {
                "document_encode_and_index_s": indexed - begin,
                "query_encode_and_search_s": end - indexed,
                "total_s": end - begin,
            },
        )

    def reference_run():
        begin = time.perf_counter()
        vectors = np.ascontiguousarray(reference_encode(texts), dtype=np.float32)
        indexed = time.perf_counter()
        query_vectors = reference_encode(query_texts)
        ranks = exact_rank(vectors, query_vectors, ids, k)
        end = time.perf_counter()
        return (
            (vectors, query_vectors),
            ranks,
            {
                "document_encode_and_index_s": indexed - begin,
                "query_encode_and_search_s": end - indexed,
                "total_s": end - begin,
            },
        )

    # Warm full representative input shape for each implementation before timing.
    clmkit_run()
    reference_run()
    check_budget()
    measurements: dict[str, list[dict]] = {"clmkit": [], "sentence_transformers_numpy": []}
    execution_order = []
    outputs = {}
    for repeat in range(repeats):
        order = ["clmkit", "sentence_transformers_numpy"]
        if repeat % 2:
            order.reverse()
        execution_order.append(order)
        for name in order:
            check_budget()
            result = clmkit_run() if name == "clmkit" else reference_run()
            outputs[name] = result
            measurements[name].append(result[2])

    retriever, clmkit_ranks, _ = outputs["clmkit"]
    (ref_documents, ref_queries), ref_ranks, _ = outputs["sentence_transformers_numpy"]
    clmkit_documents = retriever.index.get_vectors(ids)
    clmkit_queries = encoder.encode(query_texts, kind="query", batch_size=32)
    allowed = {row["id"] for row in documents if row["metadata"]["dx_partition"] == 0}
    filtered = retriever.search(query_texts, k=len(ids), filter={"dx_partition": 0})
    filtered_ranks = [
        [(hit.id, hit.score) for hit in sorted(row, key=lambda hit: (-hit.score, hit.id))[:k]] for row in filtered
    ]
    ref_filtered = exact_rank(ref_documents, ref_queries, ids, k, allowed)
    check_budget()
    with tempfile.TemporaryDirectory(prefix="clmkit-evidence-dx-") as temporary:
        root = Path(temporary)
        retriever.save(root / "clmkit")
        loaded = Retriever.load(root / "clmkit", encoder, strict=True, encoder_identity=revision)
        reloaded = loaded.search(query_texts, k=len(ids))
        reloaded_ids = [[hit.id for hit in sorted(row, key=lambda hit: (-hit.score, hit.id))[:k]] for row in reloaded]
        try:
            Retriever.load(root / "clmkit", encoder, strict=True, encoder_identity="deliberately-wrong-identity")
        except ValueError as exc:
            clmkit_mismatch = {"rejected": True, "error": str(exc).replace(str(root), "<temporary>")}
        else:
            clmkit_mismatch = {"rejected": False}
        save_reference(root / "direct", ref_documents, documents, config)
        restored_vectors, restored_documents = load_reference(root / "direct", config)
        restored_ids = [row["id"] for row in restored_documents]
        direct_reload_ranks = exact_rank(restored_vectors, ref_queries, restored_ids, k)
        try:
            load_reference(root / "direct", {**config, "revision": "deliberately-wrong-identity"})
        except ValueError as exc:
            direct_mismatch = {"rejected": True, "error": str(exc)}
        else:
            direct_mismatch = {"rejected": False}
    checks = {
        "document_vectors": parity(clmkit_documents, ref_documents),
        "query_vectors": parity(clmkit_queries, ref_queries),
        "top_k_rank_ids_equal": rank_ids(clmkit_ranks) == rank_ids(ref_ranks),
        "filter_rank_ids_equal": rank_ids(filtered_ranks) == rank_ids(ref_filtered),
        "filter_contains_only_allowed_ids": all(hit[0] in allowed for row in filtered_ranks for hit in row),
        "clmkit_reload_rank_ids_equal": reloaded_ids == rank_ids(clmkit_ranks),
        "direct_reload_rank_ids_equal": rank_ids(direct_reload_ranks) == rank_ids(ref_ranks),
        "clmkit_wrong_identity": clmkit_mismatch,
        "direct_wrong_identity": direct_mismatch,
    }
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed-as-distribution"
    distributions = sorted(
        {
            dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions() if dist.metadata["Name"]
        }.items()
    )
    return {
        "schema_version": "evidence-dx-v1",
        "claim_scope": "Descriptive executable workflow parity and timing; no human productivity or relevance claim.",
        "test_sealed": True,
        "dataset": prepared["manifest"]["dataset"],
        "split": "dev",
        "config": config,
        "document_ids": ids,
        "query_ids": [row["id"] for row in queries],
        "sampling": "First SHA256(ID)-ordered records, then ID sort; default 100 documents/20 queries.",
        "checks": checks,
        "rankings": {"clmkit": rank_ids(clmkit_ranks), "sentence_transformers_numpy": rank_ids(ref_ranks)},
        "measurements": measurements,
        "execution_order": execution_order,
        "timing_summary": {
            name: {
                "median_total_s": float(np.median([r["total_s"] for r in runs])),
                "min_total_s": min(r["total_s"] for r in runs),
                "max_total_s": max(r["total_s"] for r in runs),
            }
            for name, runs in measurements.items()
        },
        "load_seconds": {"clmkit_first": clmkit_load, "direct_second": direct_load},
        "load_timing_caveat": "Fixed load order; shared HF cache may favor second loader. Not a cold-start comparison.",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch_threads": torch.get_num_threads(),
            "versions": versions,
            "installed_distributions": dict(distributions),
        },
        "workflow": {
            "clmkit": [
                "Install clmkit[hf] at the recorded source revision",
                "load_encoder with model revision and max_length",
                "Retriever.add texts, IDs and metadata",
                "Retriever.search with metadata filter",
                "Retriever.save; Retriever.load(strict=True, encoder_identity=revision)",
            ],
            "direct": [
                "Install sentence-transformers and numpy at recorded versions",
                "SentenceTransformer with same model revision; set max_seq_length",
                "Encode and normalize documents and queries",
                "Store IDs and metadata; apply eligibility; matrix cosine; sort score then ID",
                "Save vectors plus document/config JSON; validate expected config on reload",
            ],
            "code_examples": {
                "clmkit": (
                    "encoder = load_encoder(MODEL, revision=REVISION, max_length=MAX_LENGTH, device='cpu')\n"
                    "r = Retriever(encoder, encoder_identity=REVISION)\n"
                    "r.add(texts, ids=ids, metadata=metadata)\n"
                    "hits = r.search(query, filter={'dx_partition': 0})"
                ),
                "direct": (
                    "model = SentenceTransformer(MODEL, revision=REVISION, device='cpu')\n"
                    "model.max_seq_length = MAX_LENGTH\n"
                    "docs = model.encode(texts, normalize_embeddings=True)\n"
                    "queries = model.encode(queries, normalize_embeddings=True)\n"
                    "ranks = exact_rank(docs, queries, ids, k=10, allowed=allowed_ids)"
                ),
            },
        },
        "limitations": [
            "One tiny deterministic development sample and runner; no independent participant study.",
            "Shared environment package counts do not establish isolated dependency footprints.",
            "Wrong-model test changes immutable identity/configuration; it does not download a second model.",
            "Both reload checks run in this process; fresh-process recovery is not established.",
            "Both snapshots are simple multi-file single-writer examples, not crash-safe database storage.",
            "Rank differences caused by close floating-point scores are reported, never silently dismissed.",
            "Full candidate fetch plus canonical tie sorting is included in clmkit timing.",
        ],
        "elapsed_s": time.perf_counter() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-length", type=int, default=128, choices=[128, 256])
    parser.add_argument("--documents", type=int, default=100)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    report = run(
        args.data_dir,
        revision=args.revision,
        max_length=args.max_length,
        document_count=args.documents,
        query_count=args.queries,
        repeats=args.repeats,
        threads=args.threads,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output), "checks": report["checks"]}, indent=2))


if __name__ == "__main__":
    main()
