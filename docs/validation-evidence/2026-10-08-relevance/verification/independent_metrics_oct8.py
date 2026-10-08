"""Independent, standard-library verification of the sealed held-out study.

Run only after the full held-out matrix has completed. This file imports no
clmkit/evidence code, NumPy, models, or networking libraries. It emits no dataset
text. JSON receipts contain counts, recomputed metrics, identities, and hashes.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_MATRIX = {
    "banking77": ["bm25", "dense", "hybrid", "rerank", "adapted_dense"],
    "clinc150": ["bm25", "dense"],
    "scifact": ["bm25", "dense", "hybrid", "rerank"],
}
METRICS = ("hit@1", "ndcg@10", "recall@10")
ABS_TOL = 1e-12
REL_TOL = 1e-10


class VerificationError(Exception):
    pass


def need(condition, code):
    if not condition:
        raise VerificationError(code)


def object_pairs(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, "duplicate_json_object_key")
        result[key] = value
    return result


def invalid_constant(_value):
    raise VerificationError("nonfinite_json_constant")


def decode(text):
    return json.loads(text, object_pairs_hook=object_pairs, parse_constant=invalid_constant)


def read(path):
    return decode(path.read_text(encoding="utf-8"))


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def same_number(actual, expected, code):
    need(number(actual) and number(expected), code + "_not_finite")
    need(math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL), code)


def identifier(value):
    return isinstance(value, str) and bool(value)


def under(path, root):
    resolved = Path(path).resolve()
    need(resolved.is_relative_to(root), "artifact_outside_expected_root")
    need(resolved.is_file(), "artifact_missing")
    return resolved


def artifact(path, expected_hash, inventory, label):
    actual_hash = digest(path)
    need(actual_hash == expected_hash, label + "_sha256_mismatch")
    inventory[label] = {"sha256": actual_hash, "bytes": path.stat().st_size}


def checkpoint_hash(path):
    need(path.is_dir(), "checkpoint_directory_missing")
    files = sorted(p for p in path.rglob("*") if p.is_file())
    need(any(p.suffix == ".safetensors" for p in files), "checkpoint_safe_weights_missing")
    need(all(p.resolve().is_relative_to(path.resolve()) for p in files), "checkpoint_external_link")
    listing = {str(p.relative_to(path)): digest(p) for p in files}
    return hashlib.sha256(json.dumps(listing, sort_keys=True).encode()).hexdigest()


def verify_source(repo, seal):
    files = sorted([*(repo / "src" / "clmkit").rglob("*.py"), *(repo / "validation").glob("evidence*.py")])
    need(bool(files) and (repo / "src" / "clmkit").is_dir(), "source_checkout_missing")
    listing = {p.relative_to(repo).as_posix(): digest(p) for p in files}
    actual = hashlib.sha256(json.dumps(listing, sort_keys=True).encode()).hexdigest()
    need(actual == seal["source_sha256"], "current_source_differs_from_seal")
    git = shutil.which("git")
    need(git is not None, "git_unavailable")
    commit = subprocess.check_output([git, "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    need(commit == seal["source_commit"], "checkout_commit_differs_from_seal")
    return {"sha256": actual, "commit": commit, "files": len(files)}


def load_dataset(root, name, protocol, inventory):
    folder = root / name
    manifest_path = folder / "manifest.json"
    artifact(manifest_path, protocol["dataset_manifests"][name], inventory, f"data/{name}/manifest.json")
    manifest = read(manifest_path)
    need(manifest["dataset"] == name, "manifest_dataset_mismatch")
    paths = {}
    for relative in ("corpus.jsonl", "queries/test.jsonl", "qrels/test.json"):
        path = under(folder / relative, root)
        spec = manifest["files"][relative]
        artifact(path, spec["sha256"], inventory, f"data/{name}/{relative}")
        need(path.stat().st_size == spec["bytes"], "prepared_file_size_mismatch")
        paths[relative] = path
    corpus = {}
    with paths["corpus.jsonl"].open(encoding="utf-8") as handle:
        for line in handle:
            row = decode(line)
            did = row["id"]
            need(identifier(did) and did not in corpus, "invalid_or_duplicate_corpus_id")
            corpus[did] = row.get("label")
    need(list(corpus) == manifest["corpus_ids"], "corpus_ids_differ_from_manifest")
    need(len(corpus) == manifest["processed_counts"]["corpus"], "corpus_count_mismatch")
    queries = {}
    with paths["queries/test.jsonl"].open(encoding="utf-8") as handle:
        for line in handle:
            row = decode(line)
            qid = row["id"]
            need(identifier(qid) and qid not in queries, "invalid_or_duplicate_query_id")
            queries[qid] = {key: row[key] for key in ("label", "in_scope") if key in row}
    need(list(queries) == manifest["split_ids"]["test"], "query_ids_differ_from_manifest")
    need(len(queries) == manifest["processed_counts"]["queries"]["test"], "query_count_mismatch")
    judgments = read(paths["qrels/test.json"])
    need(set(judgments) == set(queries), "qrels_query_coverage_mismatch")
    for relevant in judgments.values():
        need(isinstance(relevant, dict), "invalid_qrels_row")
        need(all(did in corpus and number(score) and score >= 0 for did, score in relevant.items()), "invalid_qrel")
    if name == "clinc150":
        need(all(identifier(label) for label in corpus.values()), "invalid_corpus_label")
        for qid, query in queries.items():
            need(isinstance(query.get("in_scope"), bool), "missing_explicit_scope_label")
            if query["in_scope"]:
                need(identifier(query.get("label")), "missing_in_scope_intent")
                need(any(score > 0 for score in judgments[qid].values()), "in_scope_query_missing_positive")
            else:
                need(not judgments[qid], "oos_query_has_relevance_judgments")
    return corpus, queries, judgments


def query_metrics(hits, judgments):
    # Binary hit and recall use every strictly positive judgment. nDCG follows
    # the declared linear-gain convention: grade / log2(one-based rank + 1).
    positive = {did for did, grade in judgments.items() if grade > 0}
    if not positive:
        return None
    top = [hit["id"] for hit in hits[:10]]
    discounts = [math.log2(position + 2) for position in range(10)]
    retrieved_gain = math.fsum(judgments.get(did, 0) / discounts[i] for i, did in enumerate(top))
    ideal_grades = sorted((grade for grade in judgments.values() if grade > 0), reverse=True)[:10]
    ideal_gain = math.fsum(grade / discounts[i] for i, grade in enumerate(ideal_grades))
    return {
        "hit@1": float(bool(hits) and hits[0]["id"] in positive),
        "ndcg@10": retrieved_gain / ideal_gain,
        "recall@10": len(set(top) & positive) / len(positive),
    }


def verify_run(folder, dataset, name, method, identity, corpus, queries, judgments, protocol, seal, inventory):
    result_path = folder / "result.json"
    result = read(result_path)
    expected = {
        "status": "passed", "dataset": dataset, "split": "test", "method": method,
        "model": None if method == "bm25" else "minilm", "encoder_identity": identity,
        "protocol_sha256": seal["protocol_sha256"], "source_sha256": seal["source_sha256"],
        "manifest_sha256": protocol["dataset_manifests"][dataset], "query_limit": 0,
        "corpus_documents": len(corpus),
    }
    need(all(result.get(key) == value for key, value in expected.items()), "result_provenance_mismatch")
    need(result["runtime"]["source_commit"] == seal["source_commit"], "result_commit_mismatch")
    need(len(result["query_ids"]) == len(queries) and set(result["query_ids"]) == set(queries), "result_query_ids_mismatch")
    inventory[f"test/{dataset}/{name}/result.json"] = {"sha256": digest(result_path), "bytes": result_path.stat().st_size}
    prediction_path = folder / "predictions.jsonl.gz"
    artifact(prediction_path, result["predictions_sha256"], inventory, f"test/{dataset}/{name}/predictions.jsonl.gz")
    need(prediction_path.stat().st_size == result["predictions_bytes"], "prediction_byte_count_mismatch")
    threshold = None
    if dataset == "clinc150":
        threshold_path = folder.parents[2] / "dev" / dataset / name / "threshold.json"
        need(seal["threshold_files"].get(str(threshold_path.resolve())) == digest(threshold_path), "threshold_not_sealed")
        calibration = read(threshold_path)
        fields = {key: expected[key] for key in ("dataset", "method", "model", "encoder_identity", "protocol_sha256", "manifest_sha256", "query_limit")}
        fields["selection_split"] = "dev"
        need(all(calibration.get(key) == value for key, value in fields.items()), "calibration_provenance_mismatch")
        threshold = calibration["selected"]["threshold"]
        need(number(threshold), "invalid_calibration_threshold")
    seen, per_query, differences = set(), {}, {metric: 0.0 for metric in METRICS}
    counts = {"in_scope": 0, "out_of_scope": 0, "accepted_in_scope": 0, "accepted_out_of_scope": 0, "correct_accepted_in_scope": 0}
    with gzip.open(prediction_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = decode(line)
            qid = row["query_id"]
            need(identifier(qid) and qid in queries and qid not in seen, "prediction_query_coverage_or_duplicate")
            seen.add(qid)
            hits = row["hits"]
            need(isinstance(hits, list) and len(hits) <= protocol["top_k"], "invalid_ranked_list_length")
            hit_ids = set()
            for hit in hits:
                need(isinstance(hit, dict) and hit.get("id") in corpus and number(hit.get("score")), "invalid_prediction_hit")
                need(hit["id"] not in hit_ids, "duplicate_ranked_document")
                hit_ids.add(hit["id"])
            computed = query_metrics(hits, judgments[qid])
            if computed is not None:
                per_query[qid] = computed
                stored = result["evaluation"]["per_query"].get(qid)
                need(isinstance(stored, dict), "rankable_query_missing_recorded_metrics")
                for metric, value in computed.items():
                    same_number(value, stored.get(metric), "per_query_metric_mismatch")
                    differences[metric] = max(differences[metric], abs(value - stored[metric]))
            if threshold is not None:
                query = queries[qid]
                accepted = bool(hits) and hits[0]["score"] >= threshold
                if query["in_scope"]:
                    counts["in_scope"] += 1
                    counts["accepted_in_scope"] += int(accepted)
                    counts["correct_accepted_in_scope"] += int(accepted and corpus[hits[0]["id"]] == query["label"])
                else:
                    counts["out_of_scope"] += 1
                    counts["accepted_out_of_scope"] += int(accepted)
    need(seen == set(queries), "missing_prediction_queries")
    evaluation = result["evaluation"]
    need(set(per_query) == set(evaluation["per_query"]), "recorded_per_query_denominator_mismatch")
    need(bool(per_query), "no_rankable_queries")
    need(evaluation["total_queries"] == len(queries), "total_query_denominator_mismatch")
    need(evaluation["rankable_queries"] == len(per_query), "rankable_denominator_mismatch")
    need(evaluation["no_positive_queries"] == len(queries) - len(per_query), "no_positive_denominator_mismatch")
    aggregate = {metric: math.fsum(row[metric] for row in per_query.values()) / len(per_query) for metric in METRICS}
    for metric, value in aggregate.items():
        same_number(value, evaluation["metrics"].get(metric), "aggregate_metric_mismatch")
    output = {
        "status": "passed", "dataset": dataset, "method_name": name, "encoder_identity": identity,
        "total_queries": len(queries), "rankable_queries": len(per_query),
        "no_positive_queries": len(queries) - len(per_query), "metrics": aggregate,
        "max_per_query_absolute_difference": differences,
        "aggregate_absolute_difference": {metric: abs(aggregate[metric] - evaluation["metrics"][metric]) for metric in METRICS},
    }
    if threshold is not None:
        inside, outside = counts["in_scope"], counts["out_of_scope"]
        ai, ao = counts["accepted_in_scope"], counts["accepted_out_of_scope"]
        need(inside > 0 and outside > 0, "missing_rejection_population")
        rejection = {
            "threshold": threshold, "in_scope_queries": inside, "out_of_scope_queries": outside,
            "accepted_in_scope": ai, "accepted_out_of_scope": ao,
            "out_of_scope_false_acceptance": ao / outside, "out_of_scope_rejection_recall": (outside - ao) / outside,
            "in_scope_coverage": ai / inside,
            "accepted_in_scope_accuracy": counts["correct_accepted_in_scope"] / ai if ai else None,
        }
        stored = result["rejection"]
        need(set(stored) == set(rejection), "rejection_schema_mismatch")
        for key, value in rejection.items():
            if value is None:
                need(stored[key] is None, "empty_accepted_population_mismatch")
            else:
                same_number(value, stored[key], "rejection_metric_mismatch")
        output["rejection"] = rejection
    return output


def verify(args, receipt):
    data_root, results_root, repo = args.data_root.resolve(), args.results_root.resolve(), args.repo_root.resolve()
    protocol_path, seal_path = results_root / "protocol.json", results_root / "seal.json"
    protocol, seal = read(protocol_path), read(seal_path)
    need(seal.get("schema") == 2 and seal.get("frozen_before_test") is True, "invalid_test_seal")
    need(protocol["method_matrix"] == EXPECTED_MATRIX, "unexpected_registered_matrix")
    need(protocol["training"]["seeds"] == [42, 1729, 2026], "unexpected_registered_seeds")
    need(digest(protocol_path) == seal["protocol_sha256"], "protocol_seal_mismatch")
    receipt.update(protocol_sha256=digest(protocol_path), seal_sha256=digest(seal_path), source=verify_source(repo, seal))
    inventory = receipt["artifact_inventory"]
    for field, count in (("development_results", 13), ("development_predictions", 13), ("training_reports", 3), ("threshold_files", 2)):
        need(len(seal[field]) == count, "sealed_artifact_count_mismatch")
        for raw_path, expected_hash in seal[field].items():
            path = under(raw_path, results_root)
            artifact(path, expected_hash, inventory, "sealed/" + path.relative_to(results_root).as_posix())
    need(len(seal["sealed_checkpoints"]) == 3, "sealed_checkpoint_count_mismatch")
    checkpoint_identities = {}
    for seed in protocol["training"]["seeds"]:
        path = results_root / "training" / str(seed) / "final"
        expected = seal["sealed_checkpoints"].get(str(path))
        need(expected is not None and checkpoint_hash(path) == expected, "sealed_checkpoint_changed")
        training = read(path.parent / "training-result.json")
        need(training["status"] == "passed" and not training["pilot"], "invalid_training_status")
        need(training["config"]["seed"] == seed and training["result"]["global_step"] == protocol["training"]["max_steps"], "training_seed_or_steps_mismatch")
        need(training["checkpoint_identity"] == expected and training["source_sha256"] == seal["source_sha256"], "training_identity_mismatch")
        checkpoint_identities[seed] = expected
    need(set(checkpoint_identities.values()) == set(seal["checkpoint_identities"]), "checkpoint_identity_set_mismatch")
    receipt["checkpoint_identities"] = {str(seed): value for seed, value in checkpoint_identities.items()}
    suite_path = results_root / "suite-test.json"
    suite = read(suite_path)
    need(suite["status"] == "passed" and len(suite["jobs"]) == 13, "heldout_suite_not_complete")
    need(all(job["status"] in {"passed", "reused"} for job in suite["jobs"]), "heldout_job_incomplete")
    need(suite["protocol_sha256"] == seal["protocol_sha256"] and suite["source_sha256"] == seal["source_sha256"], "heldout_suite_provenance_mismatch")
    receipt["suite_sha256"] = digest(suite_path)
    base = protocol["models"]["minilm"]["model"] + "@" + protocol["models"]["minilm"]["revision"]
    for dataset, methods in EXPECTED_MATRIX.items():
        corpus, queries, judgments = load_dataset(data_root, dataset, protocol, inventory)
        for method in methods:
            seeds = protocol["training"]["seeds"] if method == "adapted_dense" else [None]
            for seed in seeds:
                name = f"adapted-{seed}" if seed is not None else method
                receipt["current_comparison"] = dataset + "/" + name
                identity = "sha256:" + checkpoint_identities[seed] if seed is not None else None if method == "bm25" else base
                receipt["comparisons"].append(verify_run(
                    results_root / "test" / dataset / name, dataset, name,
                    "dense" if seed is not None else method, identity,
                    corpus, queries, judgments, protocol, seal, inventory,
                ))
    need(len(receipt["comparisons"]) == 13, "comparison_count_mismatch")
    receipt.pop("current_comparison", None)
    receipt["status"] = "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/workspaces/evidence-data"))
    parser.add_argument("--results-root", type=Path, default=Path("/workspaces/evidence-results"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an independent verification receipt")
    started = time.monotonic()
    receipt = {
        "schema": "independent-heldout-metrics-v1", "status": "failed",
        "verifier_sha256": digest(Path(__file__)), "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation": "Python standard library only; no production/evidence metric imports or model execution",
        "formula": "Hit@1 and Recall@10 use positive judgments; nDCG@10 uses linear grades and log2 discounts; ranking means exclude no-positive queries; rejection retains every scope-labeled query",
        "absolute_tolerance": ABS_TOL, "relative_tolerance": REL_TOL,
        "comparisons": [], "artifact_inventory": {},
        "limitations": [
            "Verifies recorded held-out rankings and identities, not model inference or annotation truth.",
            "Does not independently recompute bootstrap intervals, timing, or developer-productivity claims.",
            "Seal metadata is a local provenance contract, not an externally timestamped attestation.",
        ],
    }
    try:
        verify(args, receipt)
    except Exception as error:
        # Avoid emitting JSON source excerpts, dataset text, or arbitrary paths.
        receipt["error_type"] = type(error).__name__
        if isinstance(error, VerificationError):
            receipt["error_code"] = str(error)
    receipt["elapsed_seconds"] = time.monotonic() - started
    receipt["completed_comparisons"] = len(receipt["comparisons"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": receipt["status"], "completed_comparisons": receipt["completed_comparisons"], "output": str(args.output)}))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
