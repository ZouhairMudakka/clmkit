"""Build compact, text-free evidence reports from the registered CPU study."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validation.evidence_metrics import paired_interval, ranking_metrics, rejection_metrics

MEANINGS = {
    "banking77": "Same-intent support matching proxy; not verified duplicate tickets or retrieved resolutions.",
    "clinc150": "Supported-intent routing and explicit out-of-scope rejection; not document answer absence.",
    "scifact": "Retrieval of judged scientific evidence; not scientific truth or medical decision accuracy.",
}
LIMITATIONS = [
    "Development findings are parameter-selection evidence; only the sealed full test matrix is held-out evidence.",
    "Public benchmark exposure during model pretraining is unknown.",
    "CPU Codespaces timing is observational and host-dependent; single sequential runs do not establish an SLA.",
    "Warm latency quantiles use a small query sample; batched throughput is a separate measurement.",
    "Reranker component durations are measured separately; their sum is not a measured end-to-end request.",
    "Bootstrap intervals are conditional on the registered trained seeds and supplied duplicate groups.",
    "Overlap filtering is a sensitivity analysis under the documented lexical grouping rule, not a replacement score.",
    "CLINC rejection thresholds apply only to their registered gallery/model and do not validate BANKING77 rejection.",
    "No results establish private-customer accuracy, business ROI, developer productivity or production readiness.",
]


def _read(path: Path) -> dict:
    def invalid(value):
        raise ValueError(f"Nonfinite JSON value in {path.name}: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid)


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _names(methods: list[str], seeds: list[int]):
    for method in methods:
        for name in [f"adapted-{seed}" for seed in seeds] if method == "adapted_dense" else [method]:
            yield name, "dense" if method == "adapted_dense" else method


def _prediction_rows(path: Path, result: dict) -> dict:
    if not path.is_file():
        raise ValueError("Result is missing its ranked prediction artifact")
    if result.get("predictions_sha256") != _digest(path) or result.get("predictions_bytes") != path.stat().st_size:
        raise ValueError("Ranked prediction artifact checksum/size differs from result")
    predictions = {}
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("query_id"), str):
                    raise ValueError("Malformed prediction row")
                qid = row["query_id"]
                hits = row.get("hits")
                if qid in predictions:
                    raise ValueError("Duplicate prediction query id")
                if not isinstance(hits, list) or any(not isinstance(hit, dict) for hit in hits):
                    raise ValueError("Malformed ranked hits")
                predictions[qid] = hits
    except (OSError, EOFError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Cannot decode ranked prediction artifact") from exc
    return predictions


def _overlap_ids(manifest: dict, phase: str) -> set[str]:
    excluded = set()
    for group in manifest.get("audit", {}).get("cross_split_groups", []):
        members = group["members"]
        if any(item["split"] == "train" for item in members):
            excluded.update(item["id"] for item in members if item["split"] == phase)
    return excluded


def _adaptation(runs: dict, queries: list[dict], seeds: list[int], excluded: set[str], target: float) -> dict | None:
    names = ["dense", *[f"adapted-{seed}" for seed in seeds]]
    if any(name not in runs for name in names):
        return None
    scores = [{qid: values["hit@1"] for qid, values in runs[name]["evaluation"]["per_query"].items()} for name in names]
    groups = {q["id"]: q.get("metadata", {}).get("group_id", q["id"]) for q in queries if q["id"] in scores[0]}

    def comparison(rows, grouping):
        interval = paired_interval(rows[0], rows[1:], grouping)
        mean_adapted = sum(sum(row.values()) / len(row) for row in rows[1:]) / len(seeds)
        delta = interval["mean_difference"]
        low = interval["ci95"][0]
        conclusion = (
            "no_gain"
            if delta <= 0
            else "inconclusive"
            if low <= 0
            else "improved_below_practical_target"
            if delta < target
            else "practical_improvement"
        )
        per_seed = [sum(row.values()) / len(row) for row in rows[1:]]
        return {
            **interval,
            "baseline_hit@1": sum(rows[0].values()) / len(rows[0]),
            "mean_adapted_hit@1": mean_adapted,
            "seeds": seeds,
            "seed_hit@1": per_seed,
            "seed_hit@1_range": [min(per_seed), max(per_seed)],
            "practical_target_absolute": target,
            "conclusion": conclusion,
        }

    result = {"official": comparison(scores, groups)}
    remaining = set(scores[0]) - excluded
    result["overlap_sensitivity"] = {
        "excluded_queries": len(set(scores[0]) & excluded),
        "remaining_queries": len(remaining),
        "policy": "Exclude only evaluation queries grouped with training examples; retain official comparison above.",
        "comparison": comparison(
            [{q: value for q, value in row.items() if q in remaining} for row in scores],
            {q: group for q, group in groups.items() if q in remaining},
        )
        if remaining
        else None,
    }
    return result


def build_report(results_root: Path, data_root: Path, output: Path, *, phase: str = "test") -> dict:
    if phase not in {"dev", "test"}:
        raise ValueError("phase must be dev or test")
    results_root, data_root = results_root.resolve(), data_root.resolve()
    protocol_path = results_root / "protocol.json"
    protocol = _read(protocol_path)
    protocol_hash = _digest(protocol_path)
    seeds = protocol["training"]["seeds"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Registered training seeds must be nonempty and distinct")
    inventory = {}

    def record(path, category):
        root = data_root if category == "data" else results_root
        key = f"{category}/{path.relative_to(root).as_posix()}"
        inventory[key] = {"sha256": _digest(path), "bytes": path.stat().st_size}

    record(protocol_path, "results")
    seal = None
    if phase == "test":
        seal_path = results_root / "seal.json"
        if not seal_path.exists():
            raise ValueError("Final test report requires the pre-test protocol seal")
        seal = _read(seal_path)
        if (
            seal.get("schema") != 2
            or seal.get("protocol_sha256") != protocol_hash
            or not seal.get("frozen_before_test")
        ):
            raise ValueError("Protocol does not match a frozen pre-test seal")
        record(seal_path, "results")
        expected_runs = sum(1 for methods in protocol["method_matrix"].values() for _ in _names(methods, seeds))
        for field, count in (
            ("development_results", expected_runs),
            ("development_predictions", expected_runs),
            ("training_reports", len(seeds)),
            ("threshold_files", len(protocol["method_matrix"].get("clinc150", []))),
        ):
            if len(seal.get(field, {})) != count:
                raise ValueError(f"Incomplete sealed {field}")
            for name, checksum in seal[field].items():
                path = Path(name).resolve()
                if not path.is_relative_to(results_root) or not path.is_file() or _digest(path) != checksum:
                    raise ValueError(f"Sealed {field} artifact changed or is missing")
                record(path, "results")
    missing, failures = [], []
    training = {}
    identities = {}
    for seed in seeds:
        path = results_root / "training" / str(seed) / "training-result.json"
        if not path.exists():
            missing.append(f"training/{seed}")
            continue
        report = _read(path)
        record(path, "results")
        if report.get("status") != "passed":
            failures.append(f"training/{seed}: {report.get('status', 'unknown')}")
            continue
        if (
            report.get("pilot")
            or report.get("protocol_sha256") != protocol_hash
            or report["config"]["seed"] != seed
            or report["result"]["global_step"] != protocol["training"]["max_steps"]
        ):
            raise ValueError("Training result is a pilot, incomplete or from another protocol")
        identity = report["checkpoint_identity"]
        if seal and (identity not in seal["checkpoint_identities"] or report["source_sha256"] != seal["source_sha256"]):
            raise ValueError("Training result differs from sealed checkpoint/source identity")
        identities[seed] = "sha256:" + identity
        training[str(seed)] = {
            "elapsed_seconds": report["elapsed_seconds"],
            "steps": report["result"]["global_step"],
            "training_examples": report["training_examples"],
            "checkpoint_identity": identity,
            "peak_rss_bytes": report["runtime"].get("peak_rss_bytes"),
        }
    datasets = {}
    banking_runs, banking_queries, banking_excluded = {}, [], set()
    for dataset, methods in protocol["method_matrix"].items():
        manifest_path = data_root / dataset / "manifest.json"
        if _digest(manifest_path) != protocol["dataset_manifests"][dataset]:
            raise ValueError(f"Dataset manifest changed: {dataset}")
        manifest = _read(manifest_path)
        record(manifest_path, "data")
        query_path = data_root / dataset / "queries" / f"{phase}.jsonl"
        if _digest(query_path) != manifest["files"][f"queries/{phase}.jsonl"]["sha256"]:
            raise ValueError("Query file differs from frozen manifest")
        record(query_path, "data")
        queries = [json.loads(line) for line in query_path.read_text(encoding="utf-8").splitlines()]
        expected_ids = manifest["split_ids"][phase]
        if sorted(q["id"] for q in queries) != sorted(expected_ids):
            raise ValueError("Query IDs differ from manifest")
        pinned = {}
        for name in ("corpus.jsonl", f"qrels/{phase}.json"):
            pinned_path = data_root / dataset / name
            if _digest(pinned_path) != manifest["files"][name]["sha256"]:
                raise ValueError("Corpus/qrels file differs from frozen manifest")
            record(pinned_path, "data")
            pinned[name] = pinned_path
        corpus = [json.loads(line) for line in pinned["corpus.jsonl"].read_text(encoding="utf-8").splitlines()]
        corpus_ids = {row["id"] for row in corpus}
        if len(corpus_ids) != len(corpus):
            raise ValueError("Duplicate corpus document IDs")
        qrels = _read(pinned[f"qrels/{phase}.json"])
        if set(qrels) != set(expected_ids):
            raise ValueError("Pinned qrels must cover every query explicitly")
        excluded = _overlap_ids(manifest, phase)
        summaries, accepted_runs = {}, {}
        for name, method in _names(methods, seeds):
            path = results_root / phase / dataset / name / "result.json"
            if not path.exists():
                missing.append(f"{phase}/{dataset}/{name}")
                continue
            result = _read(path)
            record(path, "results")
            if result.get("status") != "passed":
                failures.append(f"{phase}/{dataset}/{name}: {result.get('status', 'unknown')}")
                continue
            if (
                result.get("protocol_sha256") != protocol_hash
                or result.get("manifest_sha256") != _digest(manifest_path)
                or result.get("dataset") != dataset
                or result.get("split") != phase
                or result.get("method") != method
                or result.get("query_limit")
                or sorted(result["query_ids"]) != sorted(expected_ids)
                or result["evaluation"]["total_queries"] != len(expected_ids)
            ):
                raise ValueError(f"Result is incomplete or does not match registered comparison: {dataset}/{name}")
            model = protocol["models"]["minilm"]
            identity = (
                identities.get(int(name.split("-")[1]))
                if name.startswith("adapted-")
                else None
                if method == "bm25"
                else model["model"] + "@" + model["revision"]
            )
            if result.get("encoder_identity") != identity or (name.startswith("adapted-") and identity is None):
                raise ValueError("Result model/checkpoint identity differs from registered comparison")
            if seal and result.get("source_sha256") != seal["source_sha256"]:
                raise ValueError("Test executable source differs from seal")
            if dataset == "clinc150" and "rejection" not in result:
                raise ValueError("CLINC result is missing its separate rejection evaluation")
            predictions_path = path.parent / "predictions.jsonl.gz"
            predictions = _prediction_rows(predictions_path, result)
            verified_evaluation = ranking_metrics(queries, qrels, predictions, corpus_ids)
            if verified_evaluation != result["evaluation"]:
                raise ValueError("Stored evaluation differs from metrics recomputed from pinned qrels and predictions")
            if dataset == "clinc150":
                if seal:
                    threshold_path = results_root / "dev" / dataset / name / "threshold.json"
                    if str(threshold_path) not in seal["threshold_files"]:
                        raise ValueError("CLINC result has no sealed development threshold")
                    if result["rejection"]["threshold"] != _read(threshold_path)["selected"]["threshold"]:
                        raise ValueError("CLINC rejection threshold differs from sealed development threshold")
                verified_rejection = rejection_metrics(
                    queries, predictions, {row["id"]: row["label"] for row in corpus}, result["rejection"]["threshold"]
                )
                if verified_rejection != result["rejection"]:
                    raise ValueError("Stored rejection metrics differ from ranked predictions")
            record(predictions_path, "results")
            accepted_runs[name] = result
            evaluation = result["evaluation"]
            summaries[name] = {
                "metrics": evaluation["metrics"],
                "total_queries": evaluation["total_queries"],
                "rankable_queries": evaluation["rankable_queries"],
                "no_positive_queries": evaluation["no_positive_queries"],
                "macro_intent_hit@1": evaluation.get("macro_intent_hit@1"),
                "timing": result["timing"],
                "peak_rss_bytes": result["runtime"].get("peak_rss_bytes"),
                "rejection": result.get("rejection"),
                "candidate_coverage": result.get("candidate_coverage"),
            }
            retained = [row for qid, row in evaluation["per_query"].items() if qid not in excluded]
            summaries[name]["overlap_filtered_ranking"] = {
                "excluded_queries": len(set(result["query_ids"]) & excluded),
                "rankable_queries": len(retained),
                "metrics": {key: sum(row[key] for row in retained) / len(retained) for key in evaluation["metrics"]}
                if retained
                else {},
            }
            for filename in ("threshold.json",):
                artifact = path.parent / filename
                if artifact.exists():
                    record(artifact, "results")
        datasets[dataset] = {
            "meaning": MEANINGS[dataset],
            "rights": manifest["rights"],
            "counts": manifest["processed_counts"],
            "methods": summaries,
            "overlap_queries": len(excluded),
            "group_rule": manifest.get("audit", {}).get("rule"),
        }
        if dataset == "banking77":
            banking_runs, banking_queries, banking_excluded = accepted_runs, queries, excluded
    suite_history = []

    def collect_invocations(suite, suite_phase, *, historical=False):
        for previous in suite.get("previous_invocations", []):
            collect_invocations(previous, suite_phase, historical=True)
        status = suite.get("status", "unknown")
        suffix = " (previous invocation)" if historical else ""
        if status not in {"passed", "running"} or (historical and status == "running"):
            failures.append(f"suite-{suite_phase}: {status}{suffix}")
        jobs = []
        for job in suite.get("jobs", []):
            job_status = job.get("status", "unknown")
            if job_status not in {"passed", "reused", "running"} or (historical and job_status == "running"):
                failures.append(f"{job['name']}: {job_status}{suffix}")
            saved = {key: job[key] for key in ("name", "status", "elapsed_seconds") if key in job}
            if job.get("error"):
                saved["error_type"] = job["error"].split(":", 1)[0]
            if job.get("previous_attempt"):
                attempt = Path(job["previous_attempt"]).resolve()
                if not attempt.is_relative_to(results_root):
                    raise ValueError("Archived attempt path is outside results root")
                saved["previous_attempt"] = attempt.relative_to(results_root).as_posix()
                for filename in ("run.log", "output/result.json", "output/training-result.json"):
                    artifact = attempt / filename
                    if artifact.is_file():
                        record(artifact, "results")
            jobs.append(saved)
        suite_history.append(
            {
                "phase": suite_phase,
                "historical": historical,
                "status": status,
                "jobs": jobs,
                "elapsed_seconds": suite.get("elapsed_seconds"),
                "time_budget_seconds": suite.get("time_budget_seconds"),
            }
        )

    for suite_phase in ("pilot", "dev", "test"):
        if phase == "dev" and suite_phase == "test":
            continue
        path = results_root / f"suite-{suite_phase}.json"
        if path.exists():
            record(path, "results")
            collect_invocations(_read(path), suite_phase)
    if phase == "test" and missing:
        raise ValueError("Final test report requires complete registered results: " + ", ".join(missing))
    expected_count = sum(1 for methods in protocol["method_matrix"].values() for _ in _names(methods, seeds))
    actual_count = sum(len(dataset["methods"]) for dataset in datasets.values())
    if phase == "test" and (actual_count != expected_count or len(training) != len(seeds)):
        raise ValueError("Final test report requires successful completion of the registered matrix")
    adaptation = (
        _adaptation(banking_runs, banking_queries, seeds, banking_excluded, protocol["practical_adaptation_target"])
        if banking_runs
        else None
    )
    report = {
        "schema": 1,
        "phase": phase,
        "status": "complete" if phase == "test" else "partial_development",
        "held_out_evidence": phase == "test",
        "protocol_sha256": protocol_hash,
        "expected_evaluation_runs": expected_count,
        "completed_evaluation_runs": actual_count,
        "missing": missing,
        "failures": failures,
        "suite_invocations": suite_history,
        "datasets": datasets,
        "training": training,
        "banking77_adaptation": adaptation,
        "limitations": LIMITATIONS,
        "artifact_inventory": dict(sorted(inventory.items())),
        "artifact_count": len(inventory),
        "artifact_bytes": sum(item["bytes"] for item in inventory.values()),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "REPORT.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def render_markdown(report: dict) -> str:
    def number(value):
        return f"{value:.4f}" if isinstance(value, (int, float)) and math.isfinite(value) else "n/a"

    lines = [
        "# Compact public benchmark evidence",
        "",
        "Sealed held-out test results."
        if report["held_out_evidence"]
        else "**PARTIAL DEVELOPMENT REPORT — not held-out test evidence.**",
        "",
        f"Completed evaluation runs: {report['completed_evaluation_runs']}/{report['expected_evaluation_runs']}.",
        "",
    ]
    for name, dataset in report["datasets"].items():
        lines += [
            f"## {name}",
            "",
            dataset["meaning"],
            "",
            "| Method | Hit@1 | nDCG@10 | Recall@10 | Queries | Warm p95 (s) | Peak RSS (MiB) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for method, result in dataset["methods"].items():
            metrics, timing = result["metrics"], result["timing"]
            memory = result["peak_rss_bytes"] / 1024**2 if result["peak_rss_bytes"] is not None else None
            lines.append(
                f"| {method} | {number(metrics.get('hit@1'))} | {number(metrics.get('ndcg@10'))} | "
                f"{number(metrics.get('recall@10'))} | {result['total_queries']} | "
                f"{number(timing.get('warm_retrieval_p95_seconds'))} | {number(memory)} |"
            )
        lines += [
            "",
            f"Queries overlapping training under the fixed grouping rule: {dataset['overlap_queries']}. "
            "Official scores above retain them; separately labelled filtered scores are in report.json.",
            "",
        ]
        if name == "clinc150":
            for method, result in dataset["methods"].items():
                r = result["rejection"]
                lines.append(
                    f"- {method}: OOS false acceptance {r['out_of_scope_false_acceptance']:.2%}; "
                    f"in-scope coverage {r['in_scope_coverage']:.2%}; accepted in-scope accuracy "
                    f"{r['accepted_in_scope_accuracy']}. All OOS and in-scope queries remain in their denominators."
                )
            lines.append("")
    adaptation = report["banking77_adaptation"]
    if adaptation:
        a = adaptation["official"]
        lines += [
            "## Registered adaptation comparison",
            "",
            f"Mean over seeds {a['seeds']}: Hit@1 change {100 * a['mean_difference']:+.2f} percentage points; "
            f"paired grouped 95% interval [{100 * a['ci95'][0]:+.2f}, {100 * a['ci95'][1]:+.2f}] points.",
            f"Outcome: **{a['conclusion']}**. Practical target: {100 * a['practical_target_absolute']:.2f} points.",
            "",
        ]
    lines += [
        "## Resource costs and incomplete work",
        "",
        "Training, model loading, index rebuilding, retrieval and reranking costs "
        "are recorded separately in report.json. "
        "Warm p95 above measures retrieval only, including for reranked methods. Memory is process peak RSS.",
        "",
    ]
    for seed, training in report["training"].items():
        lines.append(f"- Training seed {seed}: {training['steps']} steps; {training['elapsed_seconds']:.1f} seconds.")
    lines += [
        f"- Missing: {', '.join(report['missing']) or 'none'}.",
        f"- Recorded failures/timeouts: {'; '.join(report['failures']) or 'none recorded'}.",
        "",
        "## Interpretation limits",
        "",
        *[f"- {item}" for item in report["limitations"]],
        "",
        f"Checksum inventory: {report['artifact_count']} artifacts, {report['artifact_bytes']} bytes. "
        "Only hashes/counts and aggregate results are exported; no query texts or model weights.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("/workspaces/evidence-results"))
    parser.add_argument("--data-root", type=Path, default=Path("/workspaces/evidence-data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["dev", "test"], default="test")
    args = parser.parse_args()
    report = build_report(args.results_root, args.data_root, args.output, phase=args.phase)
    print(json.dumps({"status": report["status"], "output": str(args.output), "artifacts": report["artifact_count"]}))


if __name__ == "__main__":
    main()
