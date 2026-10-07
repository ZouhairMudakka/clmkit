"""Report checks use tiny invented result artifacts, never downloaded benchmarks."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validation.evidence_metrics import ranking_metrics, rejection_metrics
from validation.evidence_report import _digest, build_report


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture
def study(tmp_path):
    root, data = tmp_path / "results", tmp_path / "data"
    datasets = ("banking77", "clinc150", "scifact")
    manifests = {}
    for dataset in datasets:
        files, split_ids = {}, {}
        corpus = [
            {"id": "a", "text": "PRIVATE DOCUMENT A", "label": "A"},
            {"id": "b", "text": "PRIVATE DOCUMENT B", "label": "B"},
        ]
        corpus_path = data / dataset / "corpus.jsonl"
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        corpus_path.write_text("\n".join(json.dumps(row) for row in corpus), encoding="utf-8")
        files["corpus.jsonl"] = {"sha256": _digest(corpus_path), "bytes": corpus_path.stat().st_size}
        for phase in ("dev", "test"):
            rows = [
                {
                    "id": f"{phase}-{i}",
                    "text": "PRIVATE FIXTURE TEXT DO NOT EXPORT",
                    "label": "A",
                    "in_scope": dataset != "clinc150" or i == 0,
                    "metadata": {"group_id": f"group-{i}"},
                }
                for i in range(2)
            ]
            path = data / dataset / "queries" / f"{phase}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            files[f"queries/{phase}.jsonl"] = {"sha256": _digest(path), "bytes": path.stat().st_size}
            split_ids[phase] = [row["id"] for row in rows]
            qrels = {row["id"]: {"a": 1} if row["in_scope"] else {} for row in rows}
            qrel_path = _write(data / dataset / "qrels" / f"{phase}.json", qrels)
            files[f"qrels/{phase}.json"] = {"sha256": _digest(qrel_path), "bytes": qrel_path.stat().st_size}
        manifest = {
            "files": files,
            "split_ids": split_ids,
            "processed_counts": {"corpus": 2},
            "rights": {"license": "fixture only"},
            "audit": {
                "rule": "fixture groups",
                "cross_split_groups": [
                    {"members": [{"split": "train", "id": "train-0"}, {"split": "test", "id": "test-0"}]}
                ],
            },
        }
        path = _write(data / dataset / "manifest.json", manifest)
        manifests[dataset] = _digest(path)
    protocol = {
        "training": {"seeds": [42, 1729, 2026], "max_steps": 2},
        "dataset_manifests": manifests,
        "models": {"minilm": {"model": "fixture", "revision": "fixed"}},
        "method_matrix": {"banking77": ["dense", "adapted_dense"], "clinc150": ["bm25"], "scifact": ["dense"]},
        "practical_adaptation_target": 0.02,
    }
    protocol_path = _write(root / "protocol.json", protocol)
    protocol_hash = _digest(protocol_path)
    for seed in protocol["training"]["seeds"]:
        _write(
            root / "training" / str(seed) / "training-result.json",
            {
                "status": "passed",
                "pilot": False,
                "protocol_sha256": protocol_hash,
                "config": {"seed": seed},
                "result": {"global_step": 2},
                "checkpoint_identity": str(seed),
                "source_sha256": "source",
                "elapsed_seconds": 3.0,
                "training_examples": 4,
                "runtime": {"peak_rss_bytes": 1024},
            },
        )
    development, development_predictions, thresholds = {}, {}, {}
    for phase in ("dev", "test"):
        for dataset in datasets:
            names = (
                ["dense", "adapted-42", "adapted-1729", "adapted-2026"]
                if dataset == "banking77"
                else (["bm25"] if dataset == "clinc150" else ["dense"])
            )
            for name in names:
                adapted = name.startswith("adapted-")
                queries = [
                    json.loads(line)
                    for line in (data / dataset / "queries" / f"{phase}.jsonl").read_text().splitlines()
                ]
                qrels = json.loads((data / dataset / "qrels" / f"{phase}.json").read_text())
                predictions = {
                    f"{phase}-{i}": [{"id": "a" if i == 0 and not adapted else "b", "score": 1.0}] for i in range(2)
                }
                if dataset == "clinc150":
                    predictions[f"{phase}-1"] = []
                evaluation = ranking_metrics(queries, qrels, predictions, {"a", "b"})
                result = {
                    "status": "passed",
                    "protocol_sha256": protocol_hash,
                    "manifest_sha256": manifests[dataset],
                    "dataset": dataset,
                    "split": phase,
                    "method": "dense" if adapted else name,
                    "query_limit": 0,
                    "query_ids": [f"{phase}-0", f"{phase}-1"],
                    "evaluation": evaluation,
                    "encoder_identity": "sha256:" + name.split("-")[1]
                    if adapted
                    else None
                    if name == "bm25"
                    else "fixture@fixed",
                    "source_sha256": "source",
                    "timing": {"warm_retrieval_p95_seconds": 0.01},
                    "runtime": {"peak_rss_bytes": 2 * 1024**2},
                }
                if dataset == "clinc150":
                    result["rejection"] = rejection_metrics(queries, predictions, {"a": "A", "b": "B"}, 0.5)
                path = _write(root / phase / dataset / name / "result.json", result)
                prediction_path = path.parent / "predictions.jsonl.gz"
                with gzip.open(prediction_path, "wt", encoding="utf-8") as handle:
                    for qid, hits in predictions.items():
                        handle.write(json.dumps({"query_id": qid, "hits": hits}) + "\n")
                result.update(
                    predictions_sha256=_digest(prediction_path), predictions_bytes=prediction_path.stat().st_size
                )
                _write(path, result)
                if phase == "dev":
                    development[str(path)] = _digest(path)
                    development_predictions[str(prediction_path)] = _digest(prediction_path)
                    if dataset == "clinc150":
                        threshold_path = _write(path.parent / "threshold.json", {"selected": result["rejection"]})
                        thresholds[str(threshold_path)] = _digest(threshold_path)
    _write(
        root / "seal.json",
        {
            "schema": 2,
            "protocol_sha256": protocol_hash,
            "frozen_before_test": True,
            "source_sha256": "source",
            "checkpoint_identities": ["42", "1729", "2026"],
            "development_results": development,
            "development_predictions": development_predictions,
            "training_reports": {str(path): _digest(path) for path in root.glob("training/*/training-result.json")},
            "threshold_files": thresholds,
        },
    )
    return root, data, tmp_path / "report"


def test_complete_report_keeps_official_no_gain_and_separate_overlap_sensitivity(study):
    root, data, output = study
    report = build_report(root, data, output)
    assert report["status"] == "complete" and report["held_out_evidence"]
    assert report["completed_evaluation_runs"] == report["expected_evaluation_runs"] == 6
    comparison = report["banking77_adaptation"]
    assert comparison["official"]["mean_difference"] == -0.5
    assert comparison["official"]["seed_hit@1"] == [0, 0, 0]
    assert comparison["official"]["conclusion"] == "no_gain"
    assert comparison["overlap_sensitivity"]["excluded_queries"] == 1
    assert comparison["overlap_sensitivity"]["comparison"]["mean_difference"] == 0
    banking = report["datasets"]["banking77"]["methods"]["dense"]
    assert banking["metrics"]["hit@1"] == 0.5
    assert banking["overlap_filtered_ranking"]["metrics"]["hit@1"] == 0
    assert report["datasets"]["clinc150"]["methods"]["bm25"]["rejection"]["in_scope_coverage"] == 1
    assert report["artifact_count"] == len(report["artifact_inventory"])
    assert report["artifact_bytes"] == sum(item["bytes"] for item in report["artifact_inventory"].values())
    public = (output / "report.json").read_text() + (output / "REPORT.md").read_text()
    assert "PRIVATE FIXTURE TEXT" not in public
    assert "no_gain" in public and "business ROI" in public and "not document answer absence" in public
    assert all("safetensors" not in path for path in report["artifact_inventory"])


def test_missing_test_result_cannot_be_published_as_complete(study):
    root, data, output = study
    (root / "test" / "banking77" / "adapted-1729" / "result.json").unlink()
    with pytest.raises(ValueError, match="complete registered results"):
        build_report(root, data, output)
    assert not output.exists()


def test_dev_report_is_always_partial_and_reports_failures_without_opening_test(study):
    root, data, output = study
    (root / "dev" / "banking77" / "adapted-1729" / "result.json").unlink()
    _write(root / "suite-dev.json", {"status": "failed", "jobs": [{"name": "dev-adapted-1729", "status": "timeout"}]})
    (root / "seal.json").write_text("invalid JSON", encoding="utf-8")
    (data / "banking77" / "queries" / "test.jsonl").write_text("invalid JSON", encoding="utf-8")
    report = build_report(root, data, output, phase="dev")
    assert report["status"] == "partial_development" and not report["held_out_evidence"]
    assert report["missing"] == ["dev/banking77/adapted-1729"]
    assert "dev-adapted-1729: timeout" in report["failures"]
    assert report["banking77_adaptation"] is None
    assert "PARTIAL DEVELOPMENT" in (output / "REPORT.md").read_text()


@pytest.mark.parametrize(
    "corruption", ["protocol", "manifest", "source", "model", "coverage", "sealed_dev", "predictions"]
)
def test_final_report_rejects_changed_provenance_and_incomplete_coverage(study, corruption):
    root, data, output = study
    if corruption == "protocol":
        path = root / "protocol.json"
    elif corruption == "manifest":
        path = data / "banking77" / "manifest.json"
    elif corruption == "sealed_dev":
        path = root / "dev" / "banking77" / "dense" / "result.json"
    else:
        path = root / "test" / "banking77" / "dense" / "result.json"
    result = json.loads(path.read_text())
    if corruption in {"protocol", "manifest", "sealed_dev"}:
        result["changed"] = True
    elif corruption == "source":
        result["source_sha256"] = "other"
    elif corruption == "model":
        result["encoder_identity"] = "other"
    elif corruption == "coverage":
        result["query_ids"] = ["test-0"]
    else:
        (path.parent / "predictions.jsonl.gz").unlink()
    _write(path, result)
    with pytest.raises(ValueError):
        build_report(root, data, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "corruption",
    [
        "bytes",
        "duplicate",
        "missing",
        "extra",
        "invalid_hit",
        "invalid_score",
        "duplicate_hit",
        "ranking",
        "metrics",
        "qrels",
        "rejection",
        "threshold",
    ],
)
def test_report_recomputes_metrics_from_checksum_bound_complete_predictions(study, corruption):
    root, data, output = study
    dataset = "clinc150" if corruption in {"rejection", "threshold"} else "banking77"
    name = "bm25" if dataset == "clinc150" else "dense"
    path = root / "test" / dataset / name / "result.json"
    result = json.loads(path.read_text())
    prediction_path = path.parent / "predictions.jsonl.gz"
    with gzip.open(prediction_path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if corruption == "duplicate":
        rows.append(dict(rows[0]))
    elif corruption == "missing":
        rows.pop()
    elif corruption == "extra":
        rows.append({"query_id": "unregistered", "hits": []})
    elif corruption == "invalid_hit":
        rows[0]["hits"][0]["id"] = "absent"
    elif corruption == "invalid_score":
        rows[0]["hits"][0]["score"] = float("nan")
    elif corruption == "duplicate_hit":
        rows[0]["hits"].append(dict(rows[0]["hits"][0]))
    elif corruption in {"ranking", "bytes"}:
        rows[0]["hits"][0]["id"] = "b"
    elif corruption == "metrics":
        result["evaluation"]["per_query"]["test-0"]["hit@1"] = 0
    elif corruption == "rejection":
        result["rejection"]["in_scope_coverage"] = 0
    elif corruption == "threshold":
        result["rejection"]["threshold"] = 0.6
    else:
        _write(data / dataset / "qrels" / "test.json", {"test-0": {}, "test-1": {}})
    with gzip.open(prediction_path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    if corruption != "bytes":
        result.update(predictions_sha256=_digest(prediction_path), predictions_bytes=prediction_path.stat().st_size)
    _write(path, result)
    with pytest.raises(ValueError):
        build_report(root, data, output)
    assert not output.exists()


def test_successful_resume_retains_previous_timeouts_costs_and_archive_paths(study):
    root, data, output = study
    attempt = root / "attempts" / "dev-dense-first"
    attempt.mkdir(parents=True)
    (attempt / "run.log").write_text("PRIVATE FAILED QUERY MUST NOT BE EXPORTED", encoding="utf-8")
    previous = {
        "status": "failed",
        "time_budget_seconds": 60,
        "jobs": [
            {
                "name": "dev-dense",
                "status": "failed",
                "elapsed_seconds": 60.1,
                "error": "TimeoutExpired: PRIVATE COMMAND TEXT",
            }
        ],
    }
    _write(
        root / "suite-dev.json",
        {
            "status": "passed",
            "elapsed_seconds": 5,
            "previous_invocations": [previous],
            "jobs": [{"name": "dev-dense", "status": "passed", "elapsed_seconds": 5, "previous_attempt": str(attempt)}],
        },
    )
    report = build_report(root, data, output)
    assert report["status"] == "complete"
    assert "dev-dense: failed (previous invocation)" in report["failures"]
    history = report["suite_invocations"]
    assert len(history) == 2 and history[0]["historical"] and not history[1]["historical"]
    assert history[0]["jobs"][0]["elapsed_seconds"] == 60.1
    assert history[0]["jobs"][0]["error_type"] == "TimeoutExpired"
    assert history[1]["jobs"][0]["previous_attempt"] == "attempts/dev-dense-first"
    assert "results/attempts/dev-dense-first/run.log" in report["artifact_inventory"]
    exported = (output / "report.json").read_text() + (output / "REPORT.md").read_text()
    assert "PRIVATE FAILED" not in exported and "PRIVATE COMMAND" not in exported
    assert "previous invocation" in (output / "REPORT.md").read_text()
