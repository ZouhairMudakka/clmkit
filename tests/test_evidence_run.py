"""Synthetic-only evidence runner/gate tests: no model loads or dataset downloads."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation import evidence_run as run
from validation.evidence_data import prepare_banking77, write_prepared


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "runtime", lambda: {"source_commit": "synthetic"})
    monkeypatch.setattr(run, "source_identity", lambda: "source-v1")
    args = SimpleNamespace(
        data_root=tmp_path / "data",
        results_root=tmp_path / "results",
        protocol=tmp_path / "protocol.json",
        output=tmp_path / "seal.json",
        dataset="banking77",
        split="test",
        method="dense",
        model="minilm",
        query_limit=0,
        seal=None,
        checkpoint=None,
        threshold_file=None,
    )
    for dataset in ("banking77", "clinc150", "scifact"):
        run.write_json(args.data_root / dataset / "manifest.json", {"split_ids": {"dev": ["q1", "q2"]}})
    protocol = run.lock_protocol(args.data_root, args.protocol)
    protocol_hash = run.digest(args.protocol)
    identities = {}
    for seed in protocol["training"]["seeds"]:
        folder = args.results_root / "training" / str(seed)
        (folder / "final").mkdir(parents=True)
        (folder / "final" / "model.safetensors").write_bytes(f"synthetic seed {seed}".encode())
        identities[seed] = run.checkpoint_identity(folder / "final")
        run.write_json(
            folder / "training-result.json",
            {
                "pilot": False,
                "status": "passed",
                "protocol_sha256": protocol_hash,
                "source_sha256": "source-v1",
                "config": {"seed": seed},
                "result": {"global_step": protocol["training"]["max_steps"]},
                "checkpoint_identity": identities[seed],
            },
        )
    base_identity = protocol["models"]["minilm"]["model"] + "@" + protocol["models"]["minilm"]["revision"]
    for dataset, methods in protocol["method_matrix"].items():
        for method in methods:
            names = [f"adapted-{s}" for s in identities] if method == "adapted_dense" else [method]
            for name in names:
                identity = (
                    "sha256:" + identities[int(name.split("-")[1])]
                    if method == "adapted_dense"
                    else None
                    if method == "bm25"
                    else base_identity
                )
                report = {
                    "status": "passed",
                    "dataset": dataset,
                    "split": "dev",
                    "method": "dense" if method == "adapted_dense" else method,
                    "model": None if method == "bm25" else "minilm",
                    "encoder_identity": identity,
                    "protocol_sha256": protocol_hash,
                    "source_sha256": "source-v1",
                    "query_limit": 0,
                    "manifest_sha256": protocol["dataset_manifests"][dataset],
                    "query_ids": ["q1", "q2"],
                    "evaluation": {"total_queries": 2},
                    "first_stage_predictions_sha256": "same-candidates",
                }
                folder = args.results_root / "dev" / dataset / name
                folder.mkdir(parents=True, exist_ok=True)
                prediction_path = folder / "predictions.jsonl.gz"
                with gzip.open(prediction_path, "wt", encoding="utf-8") as handle:
                    handle.write(json.dumps({"query_id": "q1", "hits": []}) + "\n")
                    handle.write(json.dumps({"query_id": "q2", "hits": []}) + "\n")
                report["predictions_sha256"] = run.digest(prediction_path)
                report["predictions_bytes"] = prediction_path.stat().st_size
                run.write_json(folder / "result.json", report)
                if dataset == "clinc150":
                    run.write_json(
                        folder / "threshold.json", {**report, "selection_split": "dev", "selected": {"threshold": 0.5}}
                    )
    return args, protocol


def test_full_matrix_seal_validates_then_allows_heldout_request(experiment):
    args, protocol = experiment
    seal = run.seal_protocol(args)
    assert len(seal["development_results"]) == 13
    args.seal = args.output
    run.validate_run_request(args, protocol, args.data_root / args.dataset / "manifest.json")


@pytest.mark.parametrize(
    "mutation", ["wrong_dataset", "missing_query", "wrong_model", "wrong_identity", "candidate_drift"]
)
def test_seal_rejects_mislabeled_incomplete_or_mismatched_matrix(experiment, mutation):
    args, _ = experiment
    method = "rerank" if mutation == "candidate_drift" else "dense"
    path = args.results_root / "dev" / "banking77" / method / "result.json"
    report = json.loads(path.read_text())
    field, value = {
        "wrong_dataset": ("dataset", "scifact"),
        "missing_query": ("query_ids", ["q1"]),
        "wrong_model": ("model", "qwen06"),
        "wrong_identity": ("encoder_identity", "unknown"),
        "candidate_drift": ("first_stage_predictions_sha256", "changed"),
    }[mutation]
    report[field] = value
    run.write_json(path, report)
    with pytest.raises(ValueError, match=r"matrix|candidate"):
        run.seal_protocol(args)
    assert not args.output.exists()


def test_seal_checks_actual_checkpoint_bytes_and_configuration(experiment):
    args, _ = experiment
    final = args.results_root / "training" / "42" / "final"
    first = run.checkpoint_identity(final)
    (final / "config.json").write_text('{"changed": true}', encoding="utf-8")
    assert run.checkpoint_identity(final) != first
    with pytest.raises(ValueError, match="checkpoint changed"):
        run.seal_protocol(args)


@pytest.mark.parametrize("mutation", ["no_seal", "subset", "source", "qwen", "threshold_missing", "threshold_wrong"])
def test_test_request_rejected_before_data_is_opened(experiment, monkeypatch, mutation):
    args, _ = experiment
    run.seal_protocol(args)
    args.seal = args.output
    args.output = args.results_root / "test-result"
    if mutation == "no_seal":
        args.seal = None
    elif mutation == "subset":
        args.query_limit = 1
    elif mutation == "source":
        monkeypatch.setattr(run, "source_identity", lambda: "source-v2")
    elif mutation == "qwen":
        args.model = "qwen06"
    elif mutation.startswith("threshold"):
        args.dataset = "clinc150"
        if mutation == "threshold_wrong":
            args.threshold_file = args.results_root / "dev" / "clinc150" / "bm25" / "threshold.json"

    def no_test_access(*args, **kwargs):
        pytest.fail("invalid request must be rejected before opening test data")

    monkeypatch.setattr(run, "load_prepared", no_test_access)
    with pytest.raises((PermissionError, ValueError)):
        run.run_evaluation(args)


@pytest.mark.parametrize("artifact", ["dev_result", "dev_predictions", "training_report", "checkpoint", "threshold"])
@pytest.mark.parametrize("remove", [False, True])
def test_test_gate_rechecks_all_sealed_artifacts_before_opening_queries(experiment, monkeypatch, artifact, remove):
    args, _ = experiment
    run.seal_protocol(args)
    args.seal = args.output
    args.output = args.results_root / "test-result"
    paths = {
        "dev_result": args.results_root / "dev" / "scifact" / "dense" / "result.json",
        "dev_predictions": args.results_root / "dev" / "scifact" / "dense" / "predictions.jsonl.gz",
        "training_report": args.results_root / "training" / "42" / "training-result.json",
        "checkpoint": args.results_root / "training" / "42" / "final" / "model.safetensors",
        "threshold": args.results_root / "dev" / "clinc150" / "bm25" / "threshold.json",
    }
    path = paths[artifact]
    if remove:
        path.unlink()
    else:
        path.write_bytes(b"changed synthetic artifact")

    def no_test_access(*args, **kwargs):
        pytest.fail("sealed artifact drift must be caught before opening test queries")

    monkeypatch.setattr(run, "load_prepared", no_test_access)
    with pytest.raises(ValueError):
        run.run_evaluation(args)


def test_seal_rejects_prediction_file_drift(experiment):
    args, _ = experiment
    path = args.results_root / "dev" / "banking77" / "dense" / "predictions.jsonl.gz"
    path.write_bytes(b"changed synthetic predictions")
    with pytest.raises(ValueError, match="predictions artifact"):
        run.seal_protocol(args)


def test_hybrid_receives_protocol_bm25_settings(experiment, monkeypatch):
    from clmkit import HashingEncoder, hybrid

    args, protocol = experiment
    protocol.update({"bm25_k1": 0.9, "bm25_b": 0.3})
    run.write_json(args.protocol, protocol)
    args.method, args.split, args.output = "hybrid", "dev", args.results_root / "hybrid-settings"
    monkeypatch.setattr(run, "load_model", lambda *args: (HashingEncoder(dim=16), "synthetic"))
    monkeypatch.setattr(
        run,
        "load_prepared",
        lambda *args, **kwargs: {
            "queries": [{"id": "q1", "text": "card"}],
            "corpus": [{"id": "d1", "text": "card support"}, {"id": "d2", "text": "moon stars"}],
            "qrels": {"q1": {"d1": 1}},
        },
    )
    actual = hybrid.HybridRetriever
    observed = {}

    def capture(encoder, **kwargs):
        observed.update(kwargs)
        return actual(encoder, **kwargs)

    monkeypatch.setattr(hybrid, "HybridRetriever", capture)
    assert run.run_evaluation(args)["status"] == "passed"
    assert observed["k1"] == 0.9
    assert observed["b"] == 0.3


def test_candidate_coverage_measures_actual_shortlist():
    queries = [{"id": "q1"}, {"id": "q2"}, {"id": "oos"}]
    qrels = {"q1": {"a": 1, "b": 1}, "q2": {"b": 1}, "oos": {}}
    predictions = {"q1": [{"id": "a"}, {"id": "b"}], "q2": [{"id": "c"}, {"id": "b"}], "oos": []}
    assert run.candidate_coverage(queries, qrels, predictions, 1) == {
        "depth": 1,
        "rankable_queries": 2,
        "hit_rate": 0.5,
        "mean_recall": 0.25,
    }


def test_bm25_subset_metrics_keep_full_gallery_and_exact_selected_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "runtime", lambda: {"source_commit": "synthetic"})
    monkeypatch.setattr(run, "source_identity", lambda: "source-v1")
    data_root = tmp_path / "data"
    for dataset in ("banking77", "clinc150", "scifact"):
        bundle = prepare_banking77(
            [(f"{label} request {i}", label) for label in ("card", "cash") for i in range(10)], []
        )
        write_prepared(bundle, data_root / dataset)
    protocol_path = tmp_path / "protocol.json"
    run.lock_protocol(data_root, protocol_path)
    args = SimpleNamespace(
        data_root=data_root,
        protocol=protocol_path,
        output=tmp_path / "result",
        dataset="banking77",
        split="dev",
        method="bm25",
        model="minilm",
        query_limit=2,
        seal=None,
        checkpoint=None,
        threshold_file=None,
    )
    report = run.run_evaluation(args)
    assert report["corpus_documents"] == 16
    assert report["evaluation"]["total_queries"] == 2
    assert report["evaluation"]["metrics"]["hit@1"] == 1.0
    with gzip.open(args.output / "predictions.jsonl.gz", "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    assert {r["query_id"] for r in rows} == set(report["query_ids"])
    assert report["predictions_sha256"] == run.digest(args.output / "predictions.jsonl.gz")
    assert report["predictions_bytes"] == (args.output / "predictions.jsonl.gz").stat().st_size
    assert report["timing"]["reranker_in_latency_quantiles"] is False
    with pytest.raises(FileExistsError):
        run.run_evaluation(args)


def test_selection_is_deterministic_stratified_and_negative_limits_rejected():
    queries = [{"id": str(i), "label": "a" if i < 5 else "b"} for i in range(10)]
    assert run.select_queries(queries, 3) == run.select_queries(list(reversed(queries)), 3)
    assert {q["label"] for q in run.select_queries(queries, 2)} == {"a", "b"}
    with pytest.raises(ValueError, match="nonnegative"):
        run.select_queries(queries, -1)


def test_reranking_preserves_fixed_candidates_and_reports_paired_component_latency(tmp_path, monkeypatch):
    from clmkit import HashingEncoder
    from clmkit import rerank as rerank_module

    monkeypatch.setattr(run, "runtime", lambda: {"source_commit": "synthetic"})
    monkeypatch.setattr(run, "source_identity", lambda: "source-v1")
    monkeypatch.setattr(run, "load_model", lambda *args: (HashingEncoder(dim=32), "synthetic-weights"))
    data_root = tmp_path / "data"
    for dataset in ("banking77", "clinc150", "scifact"):
        bundle = prepare_banking77(
            [(f"{label} request {i}", label) for label in ("card", "cash") for i in range(10)], []
        )
        write_prepared(bundle, data_root / dataset)
    protocol_path = tmp_path / "protocol.json"
    run.lock_protocol(data_root, protocol_path)
    args = SimpleNamespace(
        data_root=data_root,
        protocol=protocol_path,
        output=tmp_path / "dense",
        dataset="banking77",
        split="dev",
        method="dense",
        model="minilm",
        query_limit=0,
        seal=None,
        checkpoint=None,
        threshold_file=None,
    )
    baseline = run.run_evaluation(args)
    received = []

    class ReverseReranker:
        def rerank(self, query, hits, **kwargs):
            received.append([hit.id for hit in hits])
            return list(reversed(hits))

    monkeypatch.setattr(rerank_module, "load_reranker", lambda *args, **kwargs: ReverseReranker())
    args.method = "rerank"
    args.output = tmp_path / "rerank"
    report = run.run_evaluation(args)
    assert report["first_stage_predictions_sha256"] == baseline["first_stage_predictions_sha256"]
    assert report["first_stage_evaluation"] == baseline["evaluation"]
    assert len(received) == report["evaluation"]["total_queries"] == 4
    assert report["candidate_coverage"]["mean_recall"] == 1.0
    assert report["timing"]["combined_component_p50_seconds"] >= 0
    assert "measured separately" in report["timing"]["combined_latency_method"]
