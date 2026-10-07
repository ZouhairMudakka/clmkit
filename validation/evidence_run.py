"""Bounded CPU relevance experiments. Real downloads/runs belong in Codespaces.

Prepare data with evidence_data.py; lock the protocol; use development runs and
the pilot before sealing that protocol for a held-out test evaluation.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validation.evidence_data import load_prepared
from validation.evidence_metrics import ranking_metrics, rejection_metrics, select_rejection_threshold

MODELS = {
    "minilm": {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    },
    "reranker": {
        "model": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "revision": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
    },
    "qwen06": {"model": "Qwen/Qwen3-Embedding-0.6B", "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"},
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def runtime() -> dict:
    versions = {}
    for name in ("numpy", "torch", "transformers", "safetensors", "peft", "sentence-transformers"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            versions[name] = importlib.metadata.version(name)
    import resource

    git = shutil.which("git")
    if git is None:
        raise RuntimeError("Git is required to record source provenance")

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "versions": versions,
        "cpu_count": os.cpu_count(),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "disk_free_bytes": shutil.disk_usage(Path.cwd()).free,
        "source_commit": subprocess.check_output([git, "rev-parse", "HEAD"], text=True).strip(),  # noqa: S603
        "source_status": subprocess.check_output([git, "status", "--short"], text=True).strip(),  # noqa: S603
    }


def require_cloud() -> None:
    if os.environ.get("CODESPACES", "").lower() != "true":
        raise RuntimeError("Run real experiments only in the authorized Codespace")
    if shutil.disk_usage(Path.cwd()).free < 3 * 1024**3:
        raise RuntimeError("At least 3 GiB free disk is required before model loading")


def configure_cpu() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)


def lock_protocol(data_root: Path, output: Path) -> dict:
    manifests = {name: digest(data_root / name / "manifest.json") for name in ("banking77", "clinc150", "scifact")}
    protocol = {
        "schema": 1,
        "models": MODELS,
        "dataset_manifests": manifests,
        "max_length": {"banking77": 128, "clinc150": 128, "scifact": 256},
        "batch_size": 32,
        "top_k": 100,
        "rerank_depth": 20,
        "rrf_k": 60,
        "candidate_depth": 100,
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "training": {
            "seeds": [42, 1729, 2026],
            "max_steps": 200,
            "batch_size": 32,
            "max_length": 128,
            "learning_rate": 2e-5,
            "temperature": 0.05,
            "avoid_same_label": True,
            "max_negatives": 0,
        },
        "primary": {"banking77": "hit@1", "scifact": "ndcg@10"},
        "practical_adaptation_target": 0.02,
        "oos_false_acceptance_target": 0.05,
        "method_matrix": {
            "banking77": ["bm25", "dense", "hybrid", "rerank", "adapted_dense"],
            "clinc150": ["bm25", "dense"],
            "scifact": ["bm25", "dense", "hybrid", "rerank"],
        },
        "test_sealed": True,
        "selection": "fixed defaults; no test-informed parameter search",
        "limitations": [
            "BANKING77 relevance is same-intent proxy, not duplicate/resolution truth",
            "CLINC rejection applies only to the CLINC gallery and this model",
            "Public model pretraining exposure is unknown",
            "Rerank only top20; ranks21..100 retain their original retrieval order",
        ],
    }
    if output.exists():
        if json.loads(output.read_text()) != protocol:
            raise ValueError("Protocol already exists with different settings")
    else:
        write_json(output, protocol)
    return protocol


def select_queries(queries: list[dict], limit: int) -> list[dict]:
    if not limit or limit >= len(queries):
        return queries
    if limit < 1:
        raise ValueError("Query limit must be nonnegative")
    groups: dict[str, list[dict]] = defaultdict(list)
    for q in queries:
        groups[str(q.get("label", "all"))].append(q)
    for rows in groups.values():
        rows.sort(key=lambda q: hashlib.sha256(q["id"].encode()).hexdigest())
    selected = []
    depth = 0
    while len(selected) < limit:
        for label in sorted(groups):
            if depth < len(groups[label]):
                selected.append(groups[label][depth])
                if len(selected) == limit:
                    break
        depth += 1
    return selected


def checkpoint_identity(path: Path) -> str:
    if not list(path.rglob("*.safetensors")):
        raise ValueError("No safetensors weights in checkpoint")
    files = sorted(p for p in path.rglob("*") if p.is_file())
    return hashlib.sha256(
        json.dumps({str(p.relative_to(path)): digest(p) for p in files}, sort_keys=True).encode()
    ).hexdigest()


def prediction_identity(predictions: dict) -> str:
    value = hashlib.sha256()
    for qid, hits in sorted(predictions.items()):
        value.update(json.dumps([qid, hits], sort_keys=True, allow_nan=False).encode())
        value.update(b"\n")
    return value.hexdigest()


def source_identity() -> str:
    """Bind the gate to executable source, including uncommitted evidence code."""
    root = Path(__file__).resolve().parents[1]
    files = sorted([*(root / "src" / "clmkit").rglob("*.py"), *(root / "validation").glob("evidence*.py")])
    return hashlib.sha256(
        json.dumps({p.relative_to(root).as_posix(): digest(p) for p in files}, sort_keys=True).encode()
    ).hexdigest()


def candidate_coverage(queries: list[dict], qrels: dict, predictions: dict, depth: int) -> dict:
    values = []
    for query in queries:
        positives = {doc for doc, score in qrels[query["id"]].items() if score > 0}
        if positives:
            found = {hit["id"] for hit in predictions[query["id"]][:depth]} & positives
            values.append((bool(found), len(found) / len(positives)))
    return {
        "depth": depth,
        "rankable_queries": len(values),
        "hit_rate": float(np.mean([v[0] for v in values])) if values else None,
        "mean_recall": float(np.mean([v[1] for v in values])) if values else None,
    }


def load_model(protocol: dict, alias: str, dataset: str, checkpoint: Path | None = None):
    from clmkit import load_encoder

    spec = protocol["models"][alias]
    if checkpoint:
        encoder = load_encoder(str(checkpoint), device="cpu", dtype="float32")
        if encoder.max_length != protocol["max_length"][dataset]:
            raise ValueError("Checkpoint max_length differs from the registered comparison")
        identity = "sha256:" + checkpoint_identity(checkpoint)
    else:
        encoder = load_encoder(
            spec["model"],
            revision=spec["revision"],
            device="cpu",
            dtype="float32",
            max_length=protocol["max_length"][dataset],
        )
        identity = spec["model"] + "@" + spec["revision"]
    return encoder, identity


def train_pairs(corpus: list[dict], seed: int):
    from clmkit.data import ContrastiveExample

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in corpus:
        grouped[row["label"]].append(row)
    examples = []
    rng = random.Random(seed)  # noqa: S311 - deterministic experiment sampling, not cryptography
    for label, members in sorted(grouped.items()):
        members = sorted(members, key=lambda row: row["id"])
        rng.shuffle(members)
        for i, row in enumerate(members):
            positive = next(
                (
                    members[(i + offset) % len(members)]
                    for offset in range(1, len(members))
                    if members[(i + offset) % len(members)]["text"] != row["text"]
                ),
                None,
            )
            if positive:
                examples.append(ContrastiveExample(row["text"], positive["text"], label=label))
    if not examples:
        raise ValueError("No distinct same-label training pairs")
    return examples


def validate_run_request(args, protocol: dict, manifest_path: Path) -> None:
    if digest(manifest_path) != protocol["dataset_manifests"][args.dataset]:
        raise ValueError("Dataset manifest differs from the registered protocol")
    if args.split == "test":
        if not args.seal or args.query_limit:
            raise PermissionError("Test requires a seal and full query coverage")
        seal = json.loads(args.seal.read_text())
        if seal["protocol_sha256"] != digest(args.protocol):
            raise ValueError("Protocol changed since the test seal")
        if not seal.get("frozen_before_test") or seal.get("source_sha256") != source_identity():
            raise ValueError("Executable source changed or was not frozen before test")
        if args.model != "minilm":
            raise ValueError("Qwen is development-only until explicitly added to the sealed matrix")
        if args.checkpoint and checkpoint_identity(args.checkpoint) not in seal["checkpoint_identities"]:
            raise ValueError("Adapted checkpoint was not registered before test")
        if args.threshold_file and digest(args.threshold_file) not in seal["threshold_files"].values():
            raise ValueError("Rejection threshold file was not sealed")
        if args.dataset == "clinc150":
            if not args.threshold_file:
                raise ValueError("CLINC test needs its development threshold file before opening test data")
            threshold = json.loads(args.threshold_file.read_text())
            expected_identity = (
                None
                if args.method == "bm25"
                else protocol["models"][args.model]["model"] + "@" + protocol["models"][args.model]["revision"]
            )
            expected = {
                "dataset": args.dataset,
                "method": args.method,
                "encoder_identity": expected_identity,
                "protocol_sha256": digest(args.protocol),
                "manifest_sha256": digest(manifest_path),
                "query_limit": 0,
                "selection_split": "dev",
            }
            if any(threshold.get(key) != value for key, value in expected.items()):
                raise ValueError("Threshold is not the registered full-development threshold for this comparison")
    if args.dataset == "clinc150" and args.method not in {"bm25", "dense"}:
        raise ValueError("CLINC first-wave matrix contains BM25 and dense only")
    if args.checkpoint and (args.dataset != "banking77" or args.method != "dense"):
        raise ValueError("Registered adaptation comparison is BANKING77 dense retrieval")


def run_evaluation(args) -> dict:
    from clmkit import Retriever
    from clmkit.hybrid import BM25Retriever, HybridRetriever
    from clmkit.rerank import load_reranker
    from clmkit.types import SearchHit

    protocol = json.loads(args.protocol.read_text())
    validate_run_request(args, protocol, args.data_root / args.dataset / "manifest.json")
    data = load_prepared(args.data_root / args.dataset, args.split, allow_test=args.split == "test")
    queries = select_queries(data["queries"], args.query_limit)
    corpus = data["corpus"]
    if not queries or not corpus:
        raise ValueError("Evaluation requires nonempty queries and corpus")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite a result directory: {args.output}")
    args.output.mkdir(parents=True)
    start = time.perf_counter()
    encoder, identity = (
        (None, None) if args.method == "bm25" else load_model(protocol, args.model, args.dataset, args.checkpoint)
    )
    load_seconds = time.perf_counter() - start
    if args.method == "bm25":
        retriever = BM25Retriever(k1=protocol["bm25_k1"], b=protocol["bm25_b"])
    elif args.method == "hybrid":
        retriever = HybridRetriever(encoder, candidate_depth=protocol["candidate_depth"], rrf_k=protocol["rrf_k"])
    else:
        retriever = Retriever(encoder, encoder_identity=identity)
    start = time.perf_counter()
    retriever.add(
        [d["text"] for d in corpus],
        ids=[d["id"] for d in corpus],
        metadata=[{"label": d.get("label"), **d.get("metadata", {})} for d in corpus],
    )
    index_seconds = time.perf_counter() - start
    predictions = {}
    start = time.perf_counter()
    for offset in range(0, len(queries), protocol["batch_size"]):
        batch = queries[offset : offset + protocol["batch_size"]]
        rows = retriever.search([q["text"] for q in batch], k=protocol["top_k"])
        for q, hits in zip(batch, rows, strict=True):
            predictions[q["id"]] = [{"id": h.id, "score": float(h.score)} for h in hits]
        if offset % 512 == 0:
            print(
                json.dumps(
                    {
                        "stage": "retrieval",
                        "dataset": args.dataset,
                        "method": args.method,
                        "queries_completed": offset + len(batch),
                        "queries_total": len(queries),
                    }
                ),
                flush=True,
            )
    retrieval_seconds = time.perf_counter() - start
    qrels = {q["id"]: data["qrels"].get(q["id"], {}) for q in queries}
    before_rerank = ranking_metrics(queries, qrels, predictions, {d["id"] for d in corpus})
    first_stage_predictions_sha256 = prediction_identity(predictions)
    shortlist = candidate_coverage(queries, qrels, predictions, protocol["rerank_depth"])
    samples = select_queries(queries, min(30, len(queries)))
    # Single-query warm latency includes encoding and search, separate from batching throughput.
    retriever.search(samples[0]["text"], k=protocol["top_k"])
    latency_by_id = {}
    for q in samples:
        t = time.perf_counter()
        retriever.search(q["text"], k=protocol["top_k"])
        latency_by_id[q["id"]] = time.perf_counter() - t
    latencies = list(latency_by_id.values())
    rerank_seconds = 0.0
    reranker_load_seconds = 0.0
    combined_latencies = []
    if args.method == "rerank":
        del retriever, encoder
        gc.collect()
        load_start = time.perf_counter()
        reranker = load_reranker(
            protocol["models"]["reranker"]["model"],
            revision=protocol["models"]["reranker"]["revision"],
            device="cpu",
            max_length=256,
        )
        reranker_load_seconds = time.perf_counter() - load_start
        texts = {d["id"]: d["text"] for d in corpus}
        t = time.perf_counter()
        for i, query in enumerate(queries):
            row = predictions[query["id"]]
            head, tail = row[: protocol["rerank_depth"]], row[protocol["rerank_depth"] :]
            hits = [SearchHit(id=h["id"], score=h["score"], text=texts[h["id"]]) for h in head]
            query_start = time.perf_counter()
            scored = reranker.rerank(query["text"], hits, batch_size=16)
            if query["id"] in latency_by_id:
                combined_latencies.append(latency_by_id[query["id"]] + time.perf_counter() - query_start)
            predictions[query["id"]] = [{"id": h.id, "score": h.score, "stage": "rerank"} for h in scored] + tail
            if i % 100 == 0:
                print(json.dumps({"stage": "rerank", "completed": i + 1, "total": len(queries)}), flush=True)
        rerank_seconds = time.perf_counter() - t
    report = {
        "status": "passed",
        "dataset": args.dataset,
        "split": args.split,
        "method": args.method,
        "model": args.model if identity else None,
        "encoder_identity": identity,
        "protocol_sha256": digest(args.protocol),
        "manifest_sha256": digest(args.data_root / args.dataset / "manifest.json"),
        "query_limit": args.query_limit,
        "corpus_documents": len(corpus),
        "evaluation": ranking_metrics(queries, qrels, predictions, {d["id"] for d in corpus}),
        "timing": {
            "model_load_seconds": load_seconds,
            "index_build_seconds": index_seconds,
            "retrieval_seconds": retrieval_seconds,
            "reranking_seconds": rerank_seconds,
            "batched_queries_per_second": len(queries) / retrieval_seconds,
            "warm_retrieval_p50_seconds": float(np.median(latencies)),
            "warm_retrieval_p95_seconds": float(np.quantile(latencies, 0.95)),
            "latency_query_sample": len(samples),
            "reranker_in_latency_quantiles": False,
        },
        "runtime": runtime(),
        "query_ids": [q["id"] for q in queries],
        "source_sha256": source_identity(),
        "first_stage_predictions_sha256": first_stage_predictions_sha256,
        "limitations": protocol["limitations"],
    }
    if args.method == "rerank":
        report["first_stage_evaluation"] = before_rerank
        report["candidate_coverage"] = shortlist
        report["timing"].update(
            {
                "reranker_load_seconds": reranker_load_seconds,
                "combined_component_p50_seconds": float(np.median(combined_latencies)),
                "combined_component_p95_seconds": float(np.quantile(combined_latencies, 0.95)),
                "combined_latency_method": (
                    "sum of paired per-query retrieval and rerank durations measured separately; "
                    "excludes first-stage batching and model swap"
                ),
            }
        )
    if args.dataset == "clinc150":
        labels = {d["id"]: d["label"] for d in corpus}
        if args.split == "dev":
            threshold = select_rejection_threshold(
                queries, predictions, labels, protocol["oos_false_acceptance_target"]
            )
            threshold.update(
                {
                    "dataset": args.dataset,
                    "method": args.method,
                    "model": args.model,
                    "encoder_identity": identity,
                    "protocol_sha256": digest(args.protocol),
                    "manifest_sha256": report["manifest_sha256"],
                    "query_limit": args.query_limit,
                }
            )
            write_json(args.output / "threshold.json", threshold)
            report["rejection"] = threshold["selected"]
        else:
            if not args.threshold_file:
                raise ValueError("CLINC test needs its development threshold file")
            threshold = json.loads(args.threshold_file.read_text())
            for key in ("dataset", "method", "encoder_identity", "protocol_sha256", "manifest_sha256"):
                if threshold[key] != report[key]:
                    raise ValueError(f"Threshold belongs to a different {key}")
            if threshold["query_limit"]:
                raise ValueError("Test threshold must come from full development evaluation")
            report["rejection"] = rejection_metrics(queries, predictions, labels, threshold["selected"]["threshold"])
    with gzip.open(args.output / "predictions.jsonl.gz", "wt", encoding="utf-8") as handle:
        for q in queries:
            handle.write(json.dumps({"query_id": q["id"], "hits": predictions[q["id"]]}, allow_nan=False) + "\n")
    write_json(args.output / "result.json", report)
    print(
        json.dumps(
            {
                "status": "passed",
                "output": str(args.output),
                "metrics": report["evaluation"]["metrics"],
                "timing": report["timing"],
                "rejection": report.get("rejection"),
            }
        ),
        flush=True,
    )
    return report


def run_training(args) -> dict:
    from clmkit.training import ContrastiveTrainer, TrainConfig

    protocol = json.loads(args.protocol.read_text())
    data_path = args.data_root / "banking77"
    if digest(data_path / "manifest.json") != protocol["dataset_manifests"]["banking77"]:
        raise ValueError("Training manifest changed")
    data = load_prepared(data_path, "train")
    settings = protocol["training"]
    if args.seed not in settings["seeds"]:
        raise ValueError("Training seed is not registered")
    if args.output.exists():
        raise FileExistsError("Refusing to replace a training run")
    encoder, identity = load_model(protocol, "minilm", "banking77")
    examples = train_pairs(data["corpus"], args.seed)
    config = TrainConfig(
        output_dir=str(args.output),
        max_steps=args.pilot_steps or settings["max_steps"],
        epochs=20,
        batch_size=settings["batch_size"],
        learning_rate=settings["learning_rate"],
        loss_kwargs={"temperature": settings["temperature"]},
        seed=args.seed,
        max_negatives=0,
        avoid_same_label=True,
        precision="fp32",
        log_every=10,
    )
    started = time.perf_counter()
    result = ContrastiveTrainer(
        encoder, config, examples, callbacks=[lambda r: print(json.dumps(r), flush=True)]
    ).train()
    elapsed = time.perf_counter() - started
    report = {
        "status": "passed",
        "pilot": bool(args.pilot_steps),
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / result.global_step,
        "config": dataclasses.asdict(config),
        "result": dataclasses.asdict(result),
        "training_examples": len(examples),
        "base_identity": identity,
        "checkpoint_identity": checkpoint_identity(args.output / "final"),
        "source_sha256": source_identity(),
        "protocol_sha256": digest(args.protocol),
        "runtime": runtime(),
    }
    write_json(args.output / "training-result.json", report)
    print(
        json.dumps(
            {
                "status": "passed",
                "training_output": str(args.output),
                "steps": result.global_step,
                "seconds_per_step": report["seconds_per_step"],
            }
        ),
        flush=True,
    )
    return report


def seal_protocol(args) -> dict:
    protocol = json.loads(args.protocol.read_text())
    protocol_hash = digest(args.protocol)
    code_hash = source_identity()
    checkpoints = []
    checkpoint_by_seed = {}
    for seed in protocol["training"]["seeds"]:
        report_path = args.results_root / "training" / str(seed) / "training-result.json"
        report = json.loads(report_path.read_text())
        if (
            report["pilot"]
            or report["protocol_sha256"] != protocol_hash
            or report.get("status") != "passed"
            or report.get("source_sha256") != code_hash
            or report["config"]["seed"] != seed
            or report["result"]["global_step"] != protocol["training"]["max_steps"]
        ):
            raise ValueError("Cannot seal a pilot or a run from another protocol")
        if checkpoint_identity(report_path.parent / "final") != report["checkpoint_identity"]:
            raise ValueError("Training checkpoint changed since evaluation")
        checkpoints.append(report["checkpoint_identity"])
        checkpoint_by_seed[seed] = "sha256:" + report["checkpoint_identity"]
    thresholds = {}
    for method in ("bm25", "dense"):
        path = args.results_root / "dev" / "clinc150" / method / "threshold.json"
        data = json.loads(path.read_text())
        expected_identity = (
            None
            if method == "bm25"
            else protocol["models"]["minilm"]["model"] + "@" + protocol["models"]["minilm"]["revision"]
        )
        if (
            data["query_limit"]
            or data["protocol_sha256"] != protocol_hash
            or data.get("dataset") != "clinc150"
            or data.get("method") != method
            or data.get("selection_split") != "dev"
            or data.get("encoder_identity") != expected_identity
            or data.get("manifest_sha256") != protocol["dataset_manifests"]["clinc150"]
        ):
            raise ValueError("Threshold is not from full development evaluation")
        thresholds[str(path)] = digest(path)
    development_results = {}
    for dataset, methods in protocol["method_matrix"].items():
        manifest_path = args.data_root / dataset / "manifest.json"
        if digest(manifest_path) != protocol["dataset_manifests"][dataset]:
            raise ValueError("Dataset manifest changed before seal")
        expected_queries = json.loads(manifest_path.read_text())["split_ids"]["dev"]
        for method in methods:
            names = [f"adapted-{s}" for s in protocol["training"]["seeds"]] if method == "adapted_dense" else [method]
            for name in names:
                path = args.results_root / "dev" / dataset / name / "result.json"
                result = json.loads(path.read_text())
                identity = (
                    checkpoint_by_seed[int(name.split("-")[1])]
                    if method == "adapted_dense"
                    else None
                    if method == "bm25"
                    else protocol["models"]["minilm"]["model"] + "@" + protocol["models"]["minilm"]["revision"]
                )
                if (
                    result["query_limit"]
                    or result["protocol_sha256"] != protocol_hash
                    or result["status"] != "passed"
                    or result.get("source_sha256") != code_hash
                    or result.get("dataset") != dataset
                    or result.get("split") != "dev"
                    or result.get("method") != ("dense" if method == "adapted_dense" else method)
                    or result.get("model") != (None if method == "bm25" else "minilm")
                    or result.get("encoder_identity") != identity
                    or result.get("manifest_sha256") != protocol["dataset_manifests"][dataset]
                    or sorted(result.get("query_ids", [])) != sorted(expected_queries)
                    or result["evaluation"]["total_queries"] != len(expected_queries)
                ):
                    raise ValueError("Incomplete development matrix")
                development_results[str(path)] = digest(path)
                if method == "rerank":
                    baseline = json.loads((args.results_root / "dev" / dataset / "dense" / "result.json").read_text())
                    if result.get("first_stage_predictions_sha256") != baseline.get("first_stage_predictions_sha256"):
                        raise ValueError("Reranker must reuse identical dense first-stage candidate lists")
    seal = {
        "protocol_sha256": digest(args.protocol),
        "checkpoint_identities": checkpoints,
        "threshold_files": thresholds,
        "source_commit": runtime()["source_commit"],
        "source_sha256": code_hash,
        "development_results": development_results,
        "frozen_before_test": True,
    }
    if args.output.exists():
        raise FileExistsError("Seal already exists")
    write_json(args.output, seal)
    print(json.dumps(seal), flush=True)
    return seal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["lock", "run", "train", "seal", "preflight"])
    parser.add_argument("--data-root", type=Path, default=Path("/workspaces/evidence-data"))
    parser.add_argument("--protocol", type=Path, default=Path("/workspaces/evidence-results/protocol.json"))
    parser.add_argument("--results-root", type=Path, default=Path("/workspaces/evidence-results"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset", choices=["banking77", "clinc150", "scifact"], default="banking77")
    parser.add_argument("--method", choices=["bm25", "dense", "hybrid", "rerank"], default="dense")
    parser.add_argument("--model", choices=["minilm", "qwen06"], default="minilm")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pilot-steps", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--threshold-file", type=Path)
    parser.add_argument("--seal", type=Path)
    args = parser.parse_args()
    require_cloud()
    if args.command == "preflight":
        print(json.dumps(runtime(), indent=2))
        return
    if args.command == "lock":
        lock_protocol(args.data_root, args.protocol)
        print(json.dumps({"protocol": str(args.protocol), "sha256": digest(args.protocol)}))
        return
    if args.output is None:
        parser.error("--output is required")
    if args.command == "seal":
        seal_protocol(args)
        return
    configure_cpu()
    if args.command == "train":
        run_training(args)
    else:
        run_evaluation(args)


if __name__ == "__main__":
    main()
