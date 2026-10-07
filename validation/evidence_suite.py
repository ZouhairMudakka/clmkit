"""Sequential, bounded Codespaces orchestration for the registered CPU study."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validation.evidence_run import digest, require_cloud, write_json


def commands(phase: str, root: Path, data: Path) -> list[tuple[str, list[str], int]]:
    base = [sys.executable, "validation/evidence_run.py"]
    common = ["--data-root", str(data), "--protocol", str(root / "protocol.json"), "--results-root", str(root)]
    jobs = []
    if phase == "pilot":
        for method, limit in (("dense", 100), ("rerank", 30)):
            output = root / "pilot" / method
            jobs.append(
                (
                    f"pilot-{method}",
                    [
                        *base,
                        "run",
                        *common,
                        "--dataset",
                        "banking77",
                        "--method",
                        method,
                        "--query-limit",
                        str(limit),
                        "--output",
                        str(output),
                    ],
                    900,
                )
            )
        jobs.append(
            (
                "pilot-training",
                [
                    *base,
                    "train",
                    *common,
                    "--seed",
                    "42",
                    "--pilot-steps",
                    "10",
                    "--output",
                    str(root / "pilot" / "training"),
                ],
                900,
            )
        )
        return jobs
    if phase not in {"dev", "test"}:
        raise ValueError("Unknown suite phase")
    protocol = json.loads((root / "protocol.json").read_text())
    if phase == "dev":
        for seed in protocol["training"]["seeds"]:
            jobs.append(
                (
                    f"train-{seed}",
                    [*base, "train", *common, "--seed", str(seed), "--output", str(root / "training" / str(seed))],
                    3600,
                )
            )
    for dataset, methods in protocol["method_matrix"].items():
        for method in methods:
            seeds = protocol["training"]["seeds"] if method == "adapted_dense" else [None]
            for seed in seeds:
                name = f"adapted-{seed}" if seed is not None else method
                actual_method = "dense" if seed is not None else method
                argv = [
                    *base,
                    "run",
                    *common,
                    "--dataset",
                    dataset,
                    "--method",
                    actual_method,
                    "--split",
                    phase,
                    "--output",
                    str(root / phase / dataset / name),
                ]
                if seed is not None:
                    argv += ["--checkpoint", str(root / "training" / str(seed) / "final")]
                if phase == "test":
                    argv += ["--seal", str(root / "seal.json")]
                    if dataset == "clinc150":
                        argv += ["--threshold-file", str(root / "dev" / dataset / name / "threshold.json")]
                jobs.append((f"{phase}-{dataset}-{name}", argv, 2700))
    return jobs


def run_suite(phase: str, root: Path, data: Path, *, max_seconds: int, resume: bool) -> None:
    require_cloud()
    if max_seconds < 1:
        raise ValueError("Suite time limit must be positive")
    started = time.monotonic()
    jobs = commands(phase, root, data)
    status_path = root / f"suite-{phase}.json"
    status = {
        "phase": phase,
        "status": "running",
        "jobs": [],
        "time_budget_seconds": max_seconds,
        "protocol_sha256": digest(root / "protocol.json"),
    }
    write_json(status_path, status)
    for name, argv, limit in jobs:
        output = Path(argv[argv.index("--output") + 1])
        result_file = output / ("training-result.json" if argv[2] == "train" else "result.json")
        if resume and result_file.exists():
            result = json.loads(result_file.read_text())
            if result.get("status") != "passed" or result.get("protocol_sha256") != status["protocol_sha256"]:
                raise ValueError(f"Cannot resume mismatched result {result_file}")
            status["jobs"].append({"name": name, "status": "reused", "result": str(result_file)})
            write_json(status_path, status)
            continue
        remaining = max_seconds - (time.monotonic() - started)
        if remaining < 10:
            status["status"] = "time_budget_exhausted"
            write_json(status_path, status)
            raise TimeoutError("Suite budget exhausted; no further experiment started")
        log_path = root / "logs" / f"{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        item = {"name": name, "status": "running", "command": argv, "log": str(log_path)}
        status["jobs"].append(item)
        write_json(status_path, status)
        print(json.dumps({"starting": name, "log": str(log_path)}), flush=True)
        task_start = time.monotonic()
        try:
            with log_path.open("w", encoding="utf-8") as log:
                # argv is built above from fixed commands and Path arguments, without a shell.
                subprocess.run(  # noqa: S603
                    argv, stdout=log, stderr=subprocess.STDOUT, timeout=min(limit, remaining), check=True
                )
            if not result_file.exists():
                raise RuntimeError("Subprocess exited without its result artifact")
            result = json.loads(result_file.read_text())
            if result.get("status") != "passed":
                raise RuntimeError("Experiment did not report success")
            item.update(status="passed", result=str(result_file), elapsed_seconds=time.monotonic() - task_start)
        except Exception as error:
            item.update(
                status="failed", error=f"{type(error).__name__}: {error}", elapsed_seconds=time.monotonic() - task_start
            )
            status["status"] = "failed"
            write_json(status_path, status)
            raise
        write_json(status_path, status)
        print(json.dumps({"completed": name, "seconds": item["elapsed_seconds"]}), flush=True)
    status.update(status="passed", elapsed_seconds=time.monotonic() - started)
    write_json(status_path, status)
    print(json.dumps({"phase": phase, "status": "passed", "seconds": status["elapsed_seconds"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["pilot", "dev", "test"])
    parser.add_argument("--results-root", type=Path, default=Path("/workspaces/evidence-results"))
    parser.add_argument("--data-root", type=Path, default=Path("/workspaces/evidence-data"))
    parser.add_argument("--max-seconds", type=int, default=3600)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_suite(args.phase, args.results_root, args.data_root, max_seconds=args.max_seconds, resume=args.resume)


if __name__ == "__main__":
    main()
