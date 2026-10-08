"""Prepared auxiliary runner. Upload for review; execute only after the test suite passes.

No installation, downloads, retraining, existing-artifact replacement, or cleanup.
Private stdout/stderr logs may contain matched examples; receipt.json never does.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

SOURCE = Path("/workspaces/clmkit-evidence-aux")
PYTHON = Path("/workspaces/clmkit/.venv-validation/bin/python")
DATA = Path("/workspaces/evidence-data")
RESULTS = Path("/workspaces/evidence-results")
OUTPUT = RESULTS / "aux-oct8"
INDEXES = Path("/workspaces/intent-indexes/aux-oct8")
PROTOCOL = RESULTS / "protocol.json"
SEAL = RESULTS / "seal.json"
CHECKPOINT = RESULTS / "training/42/final"
THRESHOLD = RESULTS / "dev/clinc150/dense/threshold.json"
FROZEN_COMMIT = "e61cc76d6cf52f25422335d42eaf18dde44e3067"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
EXPECTED_GALLERIES = {"banking77": 8006, "clinc150": 15000}
TOTAL_SECONDS = 3600


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def git(*args):
    return subprocess.check_output(  # noqa: S603 - fixed read-only git calls below; no shell
        ["git", *args],  # noqa: S607 - standard Codespace git
        cwd=SOURCE,
        text=True,
        timeout=30,
    ).strip()


def checkpoint_identity():
    if not list(CHECKPOINT.rglob("*.safetensors")):
        raise ValueError("Existing seed-42 safetensors checkpoint is required")
    files = sorted(path for path in CHECKPOINT.rglob("*") if path.is_file())
    hashes = {str(path.relative_to(CHECKPOINT)): digest(path) for path in files}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def preflight():
    if os.environ.get("CODESPACES", "").lower() != "true" or os.name != "posix":
        raise RuntimeError("Run only in the authorized Linux Codespace")
    if Path.cwd().resolve() != SOURCE or not PYTHON.is_file():
        raise RuntimeError("Use the specified auxiliary checkout and retained validation interpreter")
    if OUTPUT.exists() or INDEXES.exists():
        raise FileExistsError("Auxiliary report/index directories already exist; preserve them and review manually")
    protocol, seal, suite = read_json(PROTOCOL), read_json(SEAL), read_json(RESULTS / "suite-test.json")
    protocol_hash = digest(PROTOCOL)
    if suite.get("phase") != "test" or suite.get("status") != "passed" or suite.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("The registered held-out test suite must finish successfully before auxiliary workloads")
    if (
        seal.get("schema") != 2
        or not seal.get("frozen_before_test")
        or seal.get("protocol_sha256") != protocol_hash
        or suite.get("source_sha256") != seal.get("source_sha256")
    ):
        raise ValueError("Test suite and frozen protocol seal disagree")
    expected_jobs = {}
    for dataset, methods in protocol["method_matrix"].items():
        for method in methods:
            names = (
                [f"adapted-{seed}" for seed in protocol["training"]["seeds"]] if method == "adapted_dense" else [method]
            )
            for name in names:
                expected_jobs[f"test-{dataset}-{name}"] = RESULTS / "test" / dataset / name / "result.json"
    jobs = suite.get("jobs", [])
    if len(jobs) != len(expected_jobs) or {job["name"] for job in jobs} != set(expected_jobs):
        raise ValueError("Held-out suite job matrix is incomplete")
    for job in jobs:
        path = expected_jobs[job["name"]]
        if job.get("status") not in {"passed", "reused"} or Path(job.get("result", "")).resolve() != path:
            raise ValueError("Held-out suite contains incomplete jobs")
        result = read_json(path)
        if (
            result.get("status") != "passed"
            or result.get("protocol_sha256") != protocol_hash
            or result.get("source_sha256") != seal["source_sha256"]
        ):
            raise ValueError("Held-out result does not match its completed suite")
    if protocol["models"]["minilm"] != {"model": MODEL, "revision": REVISION}:
        raise ValueError("Expected the approved pinned MiniLM revision")
    gallery_hashes = {}
    for dataset, count in EXPECTED_GALLERIES.items():
        manifest_path = DATA / dataset / "manifest.json"
        manifest = read_json(manifest_path)
        corpus_path = DATA / dataset / "corpus.jsonl"
        with corpus_path.open("rb") as handle:
            actual_count = sum(1 for line in handle if line.strip())
        if (
            digest(manifest_path) != protocol["dataset_manifests"][dataset]
            or manifest["processed_counts"]["corpus"] != count
            or actual_count != count
            or len(manifest["corpus_ids"]) != count
            or digest(corpus_path) != manifest["files"]["corpus.jsonl"]["sha256"]
            or protocol["max_length"][dataset] != 128
        ):
            raise ValueError("Gallery count/checksum or model max length differs from the approved recipe")
        gallery_hashes[dataset] = {
            "documents": count,
            "manifest_sha256": digest(manifest_path),
            "corpus_sha256": digest(corpus_path),
        }
    adapted_hash = checkpoint_identity()
    training = read_json(RESULTS / "training/42/training-result.json")
    if (
        training.get("status") != "passed"
        or training.get("pilot")
        or training["config"]["seed"] != 42
        or training["checkpoint_identity"] != adapted_hash
        or adapted_hash not in seal["checkpoint_identities"]
    ):
        raise ValueError("Seed-42 checkpoint differs from the existing successful registered run")
    threshold = read_json(THRESHOLD)
    if (
        digest(THRESHOLD) not in seal["threshold_files"].values()
        or threshold.get("dataset") != "clinc150"
        or threshold.get("method") != "dense"
        or threshold.get("selection_split") != "dev"
        or threshold.get("query_limit") != 0
        or threshold.get("protocol_sha256") != protocol_hash
        or threshold.get("encoder_identity") != f"{MODEL}@{REVISION}"
    ):
        raise ValueError("CLINC threshold is not the sealed full-development dense threshold")
    # Auxiliary recipes need not match source_identity() over evidence scripts.
    # Their product code must match the frozen core, and every used script is hashed.
    if git("rev-parse", "HEAD:src/clmkit") != git("rev-parse", f"{FROZEN_COMMIT}:src/clmkit"):
        raise ValueError("Auxiliary core product tree differs from the frozen evidence core")
    subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", "src/clmkit"],  # noqa: S607 - fixed read-only command
        cwd=SOURCE,
        check=True,
        timeout=30,
    )
    if git("ls-files", "--others", "--exclude-standard", "--", "src/clmkit"):
        raise ValueError("Untracked product files could change the auxiliary core")
    scripts = [
        SOURCE / "validation/benchmark_serving.py",
        SOURCE / "templates/intent_matching/intent_matching.py",
        *sorted((SOURCE / "validation").glob("evidence_*.py")),
    ]
    return {
        "source_commit": git("rev-parse", "HEAD"),
        "frozen_core_commit": FROZEN_COMMIT,
        "core_git_tree": git("rev-parse", "HEAD:src/clmkit"),
        "protocol_sha256": protocol_hash,
        "seal_sha256": digest(SEAL),
        "completed_test_suite_sha256": digest(RESULTS / "suite-test.json"),
        "script_sha256": {str(path.relative_to(SOURCE)): digest(path) for path in scripts},
        "orchestrator_sha256": digest(__file__),
        "galleries": gallery_hashes,
        "frozen_identity": f"{MODEL}@{REVISION}",
        "adapted_identity": f"sha256:{adapted_hash}",
        "clinc_threshold": threshold["selected"]["threshold"],
    }


def jobs():
    common = ["--data-root", str(DATA), "--protocol", str(PROTOCOL)]
    recipe = "templates/intent_matching/intent_matching.py"
    plans = [
        {
            "name": "dx",
            "timeout_seconds": 600,
            "report": OUTPUT / "dx.json",
            "command": [
                str(PYTHON),
                "-u",
                "validation/evidence_dx.py",
                "--data-dir",
                str(DATA / "banking77"),
                "--output",
                str(OUTPUT / "dx.json"),
                "--revision",
                REVISION,
                "--max-length",
                "128",
                "--documents",
                "100",
                "--queries",
                "20",
                "--repeats",
                "3",
                "--threads",
                "2",
            ],
        },
        {
            "name": "serving",
            "timeout_seconds": 630,
            "report": OUTPUT / "serving.json",
            "command": [
                str(PYTHON),
                "-u",
                "validation/benchmark_serving.py",
                "--output",
                str(OUTPUT / "serving.json"),
                "--requests",
                "48",
                "--max-seconds",
                "600",
            ],
        },
    ]
    for name, dataset, model, text in (
        ("banking77-frozen", "banking77", "minilm", "How can I replace my lost card?"),
        ("banking77-adapted-42", "banking77", "checkpoint", "How can I replace my lost card?"),
        ("clinc150-frozen", "clinc150", "minilm", "What is my account balance?"),
    ):
        options = [*common, "--dataset", dataset, "--model", model, "--index", str(INDEXES / name)]
        if model == "checkpoint":
            options += ["--checkpoint", str(CHECKPOINT)]
        plans.append(
            {
                "name": name + "-build",
                "timeout_seconds": 1200,
                "dataset": dataset,
                "model": model,
                "index": INDEXES / name,
                "command": [str(PYTHON), "-u", recipe, "build", *options],
            }
        )
        query_options = [*options, "--text", text]
        if dataset == "clinc150":
            query_options += ["--threshold", str(THRESHOLD), "--seal", str(SEAL)]
        plans.append(
            {
                "name": name + "-query",
                "timeout_seconds": 180,
                "dataset": dataset,
                "model": model,
                "requires": name + "-build",
                "index": INDEXES / name,
                "command": [str(PYTHON), "-u", recipe, "query", *query_options],
            }
        )
    return plans


def validate_result(job, stdout_path, provenance):
    if job["name"] == "dx":
        result = read_json(job["report"])
        checks = {
            key: (value.get("allclose", value.get("rejected")) if isinstance(value, dict) else value)
            for key, value in result["checks"].items()
        }
        if (
            not all(value is True for value in checks.values())
            or len(result["document_ids"]) != 100
            or len(result["query_ids"]) != 20
            or result["environment"]["torch_threads"] != 2
            or result["config"]["revision"] != REVISION
            or result["split"] != "dev"
        ):
            raise ValueError("DX parity/configuration check failed; preserve its report")
        return {
            "checks": checks,
            "document_ids": result["document_ids"],
            "query_ids": result["query_ids"],
            "timing_summary": result["timing_summary"],
            "elapsed_seconds": result["elapsed_s"],
        }
    if job["name"] == "serving":
        result = read_json(job["report"])
        if (
            result["status"] != "passed"
            or result["revision"] != REVISION
            or result["concurrency_order"] != [1, 4, 8, 16]
            or any(row["successful"] != 48 or row["failed"] for row in result["measurements"])
        ):
            raise ValueError("Serving correctness or completeness check failed; preserve its report")
        return {
            "status": result["status"],
            "measurements": result["measurements"],
            "server": result.get("server"),
            "total_elapsed_seconds": result["total_elapsed_seconds"],
        }
    result = read_json(stdout_path)
    expected_identity = provenance["adapted_identity" if job["model"] == "checkpoint" else "frozen_identity"]
    meta = read_json(job["index"] / "retriever.json")
    count = EXPECTED_GALLERIES[job["dataset"]]
    if (
        result.get("gallery_documents") != count
        or meta.get("count") != count
        or result.get("encoder_identity") != expected_identity
        or meta.get("encoder_identity") != expected_identity
    ):
        raise ValueError("Recipe output/snapshot gallery count or encoder identity mismatch")
    if job["name"].endswith("-build"):
        return {key: result[key] for key in ("index", "gallery_documents", "encoder_identity")}
    if result.get("action_executed") is not False or result.get("dataset") != job["dataset"]:
        raise ValueError("Query must be an inspection-only matching result")
    if job["dataset"] == "clinc150" and result.get("threshold") != provenance["clinc_threshold"]:
        raise ValueError("CLINC query used a different rejection threshold")
    matches = [{key: hit[key] for key in ("id", "label", "score")} for hit in result["matches"]]
    if not matches or any(not math.isfinite(hit["score"]) for hit in matches):
        raise ValueError("Recipe query returned no usable finite scored matches")
    return {
        **{
            key: result[key]
            for key in (
                "dataset",
                "gallery_documents",
                "encoder_identity",
                "status",
                "intent",
                "threshold",
                "action_executed",
            )
        },
        "matches": matches,
    }


def stop_group(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)  # only the new session created for this job
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
    except ProcessLookupError:
        process.wait(timeout=10)


def main():
    provenance = preflight()  # no workload or output creation before this gate
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "logs").mkdir()
    plans = jobs()
    env = {
        **os.environ,
        "PYTHONPATH": f"{SOURCE}/src:{SOURCE}:/workspaces/evidence-extra",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    env.update(
        dict.fromkeys(
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
    )
    receipt = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance,
        "scope": "Auxiliary deployment/workflow checks after completed held-out evaluation; no new quality claims",
        "total_budget_seconds": TOTAL_SECONDS,
        "environment": {
            key: env[key]
            for key in env
            if key
            in {
                "PYTHONPATH",
                "TOKENIZERS_PARALLELISM",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "HF_DATASETS_OFFLINE",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "BLIS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            }
        },
        "privacy": "Matched corpus text remains in private stdout logs; public matches retain only IDs/labels/scores",
        "jobs": [],
    }

    def persist():
        temporary = OUTPUT / "receipt.json.tmp"
        temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(OUTPUT / "receipt.json")

    def interrupted(signum, frame):
        raise InterruptedError("Auxiliary runner interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    started = time.monotonic()
    completed = set()
    persist()
    try:
        for job in plans:
            item = {
                "name": job["name"],
                "command": job["command"],
                "timeout_seconds": job["timeout_seconds"],
                "status": "pending",
                "exit_code": None,
            }
            receipt["jobs"].append(item)
            if job.get("requires") and job["requires"] not in completed:
                item.update(status="blocked", reason="Fresh build did not pass")
                persist()
                continue
            remaining = TOTAL_SECONDS - (time.monotonic() - started)
            if remaining < 10:
                item.update(status="not_started", reason="Total auxiliary time budget exhausted")
                persist()
                continue
            if job.get("report") and job["report"].exists():
                raise FileExistsError("Refusing to overwrite an existing auxiliary report")
            if job["name"].endswith("-build") and job["index"].exists():
                raise FileExistsError("Refusing to overwrite an existing intent index")
            stdout_path, stderr_path = [
                OUTPUT / "logs" / f"{job['name']}.{stream}.log" for stream in ("stdout", "stderr")
            ]
            item.update(status="running", stdout=str(stdout_path), stderr=str(stderr_path))
            persist()
            process = None
            begin = time.monotonic()
            try:
                with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
                    process = subprocess.Popen(  # noqa: S603 - fixed interpreter/scripts and reviewed argument lists
                        job["command"], cwd=SOURCE, env=env, stdout=stdout, stderr=stderr, start_new_session=True
                    )
                    item["pid"] = process.pid
                    persist()
                    item["exit_code"] = process.wait(timeout=min(job["timeout_seconds"], remaining))
                if item["exit_code"] != 0:
                    raise RuntimeError("Auxiliary subprocess failed; inspect preserved private logs")
                item["result"] = validate_result(job, stdout_path, provenance)
                item["status"] = "passed"
                completed.add(job["name"])
            except subprocess.TimeoutExpired:
                item.update(status="timeout", error_type="TimeoutExpired")
            except (KeyboardInterrupt, InterruptedError):
                item.update(status="interrupted", error_type="InterruptedError")
                raise
            except Exception as error:
                item.update(status="failed", error_type=type(error).__name__)
            finally:
                if process is not None:
                    stop_group(process)
                    item["exit_code"] = process.poll()
                item["elapsed_seconds"] = time.monotonic() - begin
                item["logs"] = {
                    path.name: {"sha256": digest(path), "bytes": path.stat().st_size}
                    for path in (stdout_path, stderr_path)
                    if path.exists()
                }
                if job.get("report") and job["report"].is_file():
                    item["report"] = {
                        "path": str(job["report"]),
                        "sha256": digest(job["report"]),
                        "bytes": job["report"].stat().st_size,
                    }
                persist()
        receipt["status"] = "passed" if len(completed) == len(plans) else "partial_failure"
    finally:
        if receipt["status"] == "running":
            receipt["status"] = "interrupted_or_failed"
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        persist()
    print(json.dumps({"status": receipt["status"], "receipt": str(OUTPUT / "receipt.json")}))
    raise SystemExit(0 if receipt["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
