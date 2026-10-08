"""Prepared release-evidence packager: review first; do not run before final gates.

No networking, models, subprocesses, deletions, recursion over result directories,
or uploads. Only explicitly selected files enter the ZIP. Run in the authorized
Linux Codespace after final reporting and independent metrics verification:

  python package_relevance_oct8.py --independent-receipt \
    /workspaces/evidence-results/independent-metrics-oct8.json

The output directory must not exist. Partial output is deliberately preserved on
failure. A separate human/release task controls review and publication.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

RESULTS = Path("/workspaces/evidence-results")
DATA = Path("/workspaces/evidence-data")
OUTPUT = Path("/workspaces/evidence-release-oct8")
FROZEN_COMMIT = "e61cc76d6cf52f25422335d42eaf18dde44e3067"
EVIDENCE = Path("docs/validation-evidence/2026-10-08-relevance")
MATRIX = {
    "banking77": ["bm25", "dense", "hybrid", "rerank", "adapted-42", "adapted-1729", "adapted-2026"],
    "clinc150": ["bm25", "dense"],
    "scifact": ["bm25", "dense", "hybrid", "rerank"],
}
SCRIPTS = ["independent_metrics_oct8.py", "run_auxiliary_oct8.py"]
LOG = "attempts/dev-scifact-rerank-1362155307834f4083aa385caa19666d/run.log"
LOG_HASH = "fa60b399c9cc29d64842adb62ee3d28521f9b8e845e017e5f782ac8cd051a425"
MAX_FILE = 32 * 1024**2
MAX_TOTAL = 256 * 1024**2
MAX_INFLATED_PREDICTIONS = 128 * 1024**2
SECRET_KEYS = {"api_key", "authorization", "access_token", "password", "secret", "cookie", "cookies"}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def reject_constant(value):
    raise ValueError("Nonfinite JSON value")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def decode(data):
    return json.loads(data, parse_constant=reject_constant, object_pairs_hook=unique_object)


def checked_path(path, root):
    root = root.resolve(strict=True)
    path = Path(path)
    need(path.is_absolute(), "Input paths must be absolute")
    # Reject symlinks, including intermediate directories, before resolving.
    need(root in path.parents, "Input is outside its approved root")
    current = path
    while current != root:
        need(not current.is_symlink(), "Symlink inputs are forbidden")
        current = current.parent
    resolved = path.resolve(strict=True)
    need(resolved.is_relative_to(root) and resolved.is_file(), "Expected regular file within approved root")
    need(resolved.stat().st_size <= MAX_FILE, "Input exceeds the 32 MiB file budget")
    return resolved


def payload(path, root):
    return checked_path(path, root).read_bytes()


def read(path, root):
    return decode(payload(path, root))


def metadata(data):
    return {"sha256": sha(data), "bytes": len(data)}


def verify_entry(data, entry):
    need(
        isinstance(entry, dict) and metadata(data) == {k: entry.get(k) for k in ("sha256", "bytes")},
        "Artifact differs from its recorded SHA256/byte count",
    )


def privacy_check(value):
    """Reject credential fields and raw dataset text in structured reports.

    Environment *files* are never eligible. Existing reports contain deliberately
    limited thread/runtime metadata; they are not dumps of the process environment.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = key.casefold().replace("-", "_")
            need(normalized not in SECRET_KEYS, "Unexpected credential field in public report")
            need(
                normalized not in {"text", "query_text", "document_text", "raw_text"},
                "Unexpected raw text field in public report",
            )
            privacy_check(item)
    elif isinstance(value, list):
        for item in value:
            privacy_check(item)
    elif isinstance(value, str):
        need(
            not re.search(
                r"(?:Bearer\s+[A-Za-z0-9_.-]{12,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]+)", value
            ),
            "Possible credential value in public report",
        )


def inspect_predictions(path, manifest, phase, result):
    expected_queries = set(manifest["split_ids"][phase])
    corpus_ids = set(manifest["corpus_ids"])
    need(
        len(result["query_ids"]) == len(expected_queries) and set(result["query_ids"]) == expected_queries,
        "Result query coverage differs from the manifest",
    )
    seen, inflated = set(), 0
    with gzip.open(path, "rb") as handle:
        while True:
            line = handle.readline(1024 * 1024 + 1)
            if not line:
                break
            inflated += len(line)
            need(
                len(line) <= 1024 * 1024 and inflated <= MAX_INFLATED_PREDICTIONS,
                "Prediction expansion budget exceeded",
            )
            row = decode(line)
            need(
                set(row) == {"query_id", "hits"}
                and row["query_id"] in expected_queries
                and row["query_id"] not in seen,
                "Prediction rows must contain unique registered IDs and hits only",
            )
            seen.add(row["query_id"])
            need(isinstance(row["hits"], list) and len(row["hits"]) <= 100, "Unexpected candidate count")
            docs = set()
            for hit in row["hits"]:
                need(
                    set(hit) in ({"id", "score"}, {"id", "score", "stage"}), "Prediction hit contains non-public fields"
                )
                need(hit["id"] in corpus_ids and hit["id"] not in docs, "Invalid or duplicate predicted document")
                score = hit["score"]
                need(
                    isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score),
                    "Invalid prediction score",
                )
                need("stage" not in hit or hit["stage"] == "rerank", "Unexpected prediction stage")
                docs.add(hit["id"])
    need(seen == expected_queries, "Prediction coverage is incomplete")


def safe_progress_log(data):
    """Only the known 147-byte text-free JSON progress receipt may be published."""
    if len(data) != 147 or sha(data) != LOG_HASH:
        return False
    try:
        rows = [decode(line) for line in data.splitlines() if line.strip()]
        return bool(rows) and all(
            set(row) == {"stage", "dataset", "method", "queries_completed", "queries_total"}
            and row["stage"] == "retrieval"
            and row["dataset"] == "scifact"
            and row["method"] == "rerank"
            and type(row["queries_completed"]) is int
            and type(row["queries_total"]) is int
            and 0 <= row["queries_completed"] <= row["queries_total"] <= 162
            for row in rows
        )
    except (ValueError, TypeError, KeyError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("/workspaces/clmkit-evidence-aux"))
    parser.add_argument("--independent-receipt", type=Path, required=True)
    parser.add_argument("--dev-report", default="report-dev-oct8")
    parser.add_argument("--test-report", default="report-test-oct8")
    args = parser.parse_args()
    need(
        os.name == "posix" and os.environ.get("CODESPACES", "").lower() == "true", "Authorized Linux Codespace required"
    )
    need(not OUTPUT.exists() and not OUTPUT.is_symlink(), "Refusing existing output directory")
    for name, phase in ((args.dev_report, "dev"), (args.test_report, "test")):
        need(re.fullmatch(rf"report-{phase}(?:-[A-Za-z0-9_-]+)?", name), "Unexpected aggregate report directory name")
    repo = args.repo_root.resolve(strict=True)
    need(repo.is_relative_to(Path("/workspaces")), "Verification scripts must come from the authorized checkout")
    need(args.independent_receipt.name.endswith(".json"), "Independent receipt must be JSON")
    selected = {}
    omitted = {}

    def add(path, root, archive_name, expected=None):
        name = PurePosixPath(archive_name)
        need(not name.is_absolute() and ".." not in name.parts and "\\" not in archive_name, "Unsafe ZIP path")
        need(archive_name not in selected, "Duplicate archive member")
        data = payload(path, root)
        if expected is not None:
            verify_entry(data, expected)
        if path.suffix == ".json":
            privacy_check(decode(data))
        selected[archive_name] = {"path": path, "root": root, **metadata(data)}
        need(sum(item["bytes"] for item in selected.values()) <= MAX_TOTAL, "256 MiB package input budget exceeded")
        return data

    protocol_path, seal_path = RESULTS / "protocol.json", RESULTS / "seal.json"
    protocol = read(protocol_path, RESULTS)
    seal = read(seal_path, RESULTS)
    protocol_hash, seal_hash = sha(payload(protocol_path, RESULTS)), sha(payload(seal_path, RESULTS))
    need(
        seal.get("schema") == 2
        and seal.get("frozen_before_test") is True
        and seal.get("source_commit") == FROZEN_COMMIT
        and seal.get("protocol_sha256") == protocol_hash,
        "Protocol/source/test seal mismatch",
    )
    need(protocol["training"]["seeds"] == [42, 1729, 2026], "Unexpected registered training seeds")
    declared = {
        dataset: [
            name
            for method in methods
            for name in (
                [f"adapted-{seed}" for seed in protocol["training"]["seeds"]] if method == "adapted_dense" else [method]
            )
        ]
        for dataset, methods in protocol["method_matrix"].items()
    }
    need(declared == MATRIX, "Unexpected registered matrix")
    independent = read(args.independent_receipt, RESULTS)
    need(
        independent.get("schema") == "independent-heldout-metrics-v1"
        and independent.get("status") == "passed"
        and independent.get("completed_comparisons") == 13
        and len(independent.get("comparisons", [])) == 13
        and independent.get("protocol_sha256") == protocol_hash
        and independent.get("seal_sha256") == seal_hash
        and independent.get("source", {}).get("commit") == FROZEN_COMMIT
        and independent.get("source", {}).get("sha256") == seal["source_sha256"],
        "Passed matching independent held-out metric receipt required",
    )
    reports = {}
    for phase, folder in (("dev", args.dev_report), ("test", args.test_report)):
        report = read(RESULTS / folder / "report.json", RESULTS)
        need(
            report.get("phase") == phase
            and report.get("completed_evaluation_runs") == 13
            and report.get("expected_evaluation_runs") == 13
            and not report.get("missing")
            and report.get("protocol_sha256") == protocol_hash
            and report.get("held_out_evidence") is (phase == "test")
            and report.get("status") == ("complete" if phase == "test" else "partial_development"),
            "Complete matching development/test aggregate reports required",
        )
        reports[phase] = report
        for name in ("report.json", "REPORT.md"):
            content = add(RESULTS / folder / name, RESULTS, f"reports/{phase}/{name}")
            if name == "report.json":
                need(decode(content) == report, "Aggregate gate changed during packaging")

    def report_bound(path, phase):
        label = "results/" + path.relative_to(RESULTS).as_posix()
        entry = reports[phase]["artifact_inventory"].get(label)
        need(entry is not None, "Selected result artifact absent from aggregate report inventory")
        return entry

    for name in ("protocol.json", "seal.json"):
        content = add(RESULTS / name, RESULTS, name)
        need(
            sha(content) == (protocol_hash if name == "protocol.json" else seal_hash),
            "Gate file changed during packaging",
        )
    content = add(args.independent_receipt, RESULTS, "verification/independent-metrics-receipt.json")
    need(decode(content) == independent, "Independent receipt changed during packaging")
    for name in ("suite-pilot.json", "suite-dev.json", "suite-test.json"):
        path = RESULTS / name
        suite = read(path, RESULTS)
        need(
            suite.get("protocol_sha256") == protocol_hash and suite.get("status") == "passed",
            "Suite completion/protocol mismatch",
        )
        expected = report_bound(path, "test" if name == "suite-test.json" else "dev")
        add(path, RESULTS, name, expected)
    need(
        independent["suite_sha256"] == sha(payload(RESULTS / "suite-test.json", RESULTS)),
        "Independent suite hash changed",
    )

    manifests = {}
    for dataset in MATRIX:
        path = DATA / dataset / "manifest.json"
        manifest_data = payload(path, DATA)
        need(sha(manifest_data) == protocol["dataset_manifests"][dataset], "Dataset manifest differs from protocol")
        for phase in ("dev", "test"):
            verify_entry(manifest_data, reports[phase]["artifact_inventory"][f"data/{dataset}/manifest.json"])
        manifest = decode(manifest_data)
        manifests[dataset] = manifest
        add(path, DATA, f"data/{dataset}/manifest.json", metadata(manifest_data))
        # Explicit license allowlist: never copy raw train/test data or ZIP archives.
        notice = "LICENSE.md" if dataset == "scifact" else "LICENSE"
        need(notice in manifest["rights"]["notice_files"], "Expected original data notice is missing")
        add(
            DATA / dataset / "sources" / notice,
            DATA,
            f"data/{dataset}/sources/{notice}",
            {key: manifest["sources"][notice][key] for key in ("sha256", "bytes")},
        )

    sealed_seen = {
        field: set()
        for field in ("development_results", "development_predictions", "threshold_files", "training_reports")
    }

    def sealed(path, field, data):
        need(seal[field].get(str(path)) == sha(data), "Selected artifact differs from the seal")
        sealed_seen[field].add(str(path))

    for phase in ("dev", "test"):
        for dataset, methods in MATRIX.items():
            for method in methods:
                folder = RESULTS / phase / dataset / method
                result_path, predictions_path = folder / "result.json", folder / "predictions.jsonl.gz"
                result_data = add(
                    result_path, RESULTS, f"{phase}/{dataset}/{method}/result.json", report_bound(result_path, phase)
                )
                result = decode(result_data)
                need(
                    result.get("status") == "passed"
                    and result.get("dataset") == dataset
                    and result.get("split") == phase
                    and result.get("query_limit") == 0
                    and result.get("protocol_sha256") == protocol_hash
                    and result.get("source_sha256") == seal["source_sha256"]
                    and result.get("manifest_sha256") == protocol["dataset_manifests"][dataset],
                    "Result provenance mismatch",
                )
                predictions = add(
                    predictions_path,
                    RESULTS,
                    f"{phase}/{dataset}/{method}/predictions.jsonl.gz",
                    report_bound(predictions_path, phase),
                )
                verify_entry(
                    predictions, {"sha256": result["predictions_sha256"], "bytes": result["predictions_bytes"]}
                )
                inspect_predictions(predictions_path, manifests[dataset], phase, result)
                if phase == "dev":
                    sealed(result_path, "development_results", result_data)
                    sealed(predictions_path, "development_predictions", predictions)
                else:
                    for name, content in (("result.json", result_data), ("predictions.jsonl.gz", predictions)):
                        verify_entry(content, independent["artifact_inventory"][f"test/{dataset}/{method}/{name}"])
    for method in ("bm25", "dense"):
        path = RESULTS / "dev" / "clinc150" / method / "threshold.json"
        data = add(path, RESULTS, f"dev/clinc150/{method}/threshold.json", report_bound(path, "dev"))
        sealed(path, "threshold_files", data)
    for seed in (42, 1729, 2026):
        path = RESULTS / "training" / str(seed) / "training-result.json"
        data = add(path, RESULTS, f"training/{seed}/training-result.json", report_bound(path, "test"))
        verify_entry(data, report_bound(path, "dev"))
        sealed(path, "training_reports", data)
        training = decode(data)
        need(
            training.get("status") == "passed"
            and not training.get("pilot")
            and training["checkpoint_identity"] in seal["checkpoint_identities"]
            and training["checkpoint_identity"] == independent["checkpoint_identities"][str(seed)],
            "Training receipt mismatch",
        )
    for field, paths in sealed_seen.items():
        need(paths == set(seal[field]), "Seal contains missing or unexpected artifacts")

    aux_path = RESULTS / "aux-oct8" / "receipt.json"
    auxiliary = read(aux_path, RESULTS)
    need(
        auxiliary.get("status") in {"passed", "partial_failure", "interrupted_or_failed"},
        "Auxiliary receipt is not terminal",
    )
    need(
        auxiliary["provenance"]["protocol_sha256"] == protocol_hash
        and auxiliary["provenance"]["seal_sha256"] == seal_hash
        and auxiliary["provenance"]["frozen_core_commit"] == FROZEN_COMMIT,
        "Auxiliary provenance differs",
    )
    content = add(aux_path, RESULTS, "auxiliary/receipt.json")
    need(decode(content) == auxiliary, "Auxiliary receipt changed during packaging")
    for name in ("dx", "serving"):
        job = next((job for job in auxiliary["jobs"] if job["name"] == name), None)
        need(job is not None and "report" in job, "Both auxiliary reports must exist; failures must remain explicit")
        path = RESULTS / "aux-oct8" / f"{name}.json"
        need(Path(job["report"]["path"]) == path, "Unexpected auxiliary report path")
        add(path, RESULTS, f"auxiliary/{name}.json", {key: job["report"][key] for key in ("sha256", "bytes")})
    for name in SCRIPTS:
        data = add(repo / EVIDENCE / "verification" / name, repo, f"verification/{name}")
        expected = (
            independent["verifier_sha256"]
            if name.startswith("independent")
            else auxiliary["provenance"]["orchestrator_sha256"]
        )
        need(sha(data) == expected, "Packaged verification script differs from the executed script")
    add(repo / "LICENSE", repo, "LICENSE")

    log_path = RESULTS / LOG
    expected_log = reports["dev"]["artifact_inventory"].get("results/" + LOG)
    need(expected_log is not None, "Interrupted attempt must remain represented in the report inventory")
    if log_path.is_file():
        log_data = payload(log_path, RESULTS)
        verify_entry(log_data, expected_log)
        if safe_progress_log(log_data):
            add(log_path, RESULTS, LOG, expected_log)
        else:
            omitted[LOG] = {
                **expected_log,
                "reason": "Not approved by exact text-free 147-byte progress schema; hash only",
            }
    else:
        omitted[LOG] = {**expected_log, "reason": "File unavailable; retained inventory hash only"}

    readme = f"""# clmkit relevance evidence — 8 October 2026

Recorded study source: {FROZEN_COMMIT}
Protocol SHA256: {protocol_hash}
Seal SHA256: {seal_hash}

Includes all 13 development and 13 held-out ranked-ID/score runs, original
result JSON, CLINC development thresholds, training receipts (not weights),
protocol/seal/suite histories, aggregates, manifests/notices, auxiliary reports
and the verification scripts matching their execution receipts. Auxiliary status:
{auxiliary["status"]}. A packaged failure remains a failure, not a passing claim.

BANKING77 is same-intent matching, not resolution/duplicate truth. CLINC measures
supported-intent routing and OOS rejection, not authorization. SciFact measures
retrieval of judged scientific evidence, not scientific/medical correctness.
Data transformations include ID assignment, fixed train-derived/grouped splits,
intent-proxy judgments, and SciFact title/text concatenation; manifests disclose
counts, group limitations, source pins, hashes and attribution.

Dataset notices: BANKING77 CC BY 4.0 (PolyAI/Casanueva et al.); CLINC CC BY 3.0
(Larson et al.); SciFact claims/annotations CC BY 4.0 and abstracts ODC-By 1.0
(Wadden et al.; BEIR conversion by Thakur et al.). Preserve original notices.
The clmkit code license does not replace those data rights.
Verification code is covered by the included clmkit Apache-2.0 LICENSE.

Corpus/query text, raw downloaded datasets, model weights, environment/auth
files, index snapshots and private stdout/stderr logs are deliberately omitted.
Reproduce datasets using validation/evidence_data.py at the study commit and
the pinned protocol; fetch matching model revisions or reproduce registered
training. Aggregate inventories retain hashes for omitted inputs and failures.
The one interrupted progress log is included only after exact size/hash/schema
inspection establishes that it contains numerical progress and fixed labels;
otherwise its inventory hash remains in PACKAGE_INVENTORY.json.
Original report inventory names beginning results/ map to archive paths with
that prefix removed. Data manifests/notices retain data/; aggregate reports are
under reports/dev and reports/test. Original receipt paths remain provenance.

The alpha release source may later differ by its version declaration; this does
not change the study's source identity. Check the exact release diff separately;
do not relabel these runs as executed at a later commit. No production SLA,
business ROI, GPU/large-model capacity or human productivity claim is established.

PACKAGE_INVENTORY.json binds every payload member except itself. External
SHA256SUMS binds the ZIP and external inventory. The packaging manifest lists
the builder hash. Hashes establish artifact identity, not an external timestamp.
""".encode()
    inventory = {
        "schema": "clmkit-release-evidence-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "builder_sha256": sha(Path(__file__).read_bytes()),
        "study_commit": FROZEN_COMMIT,
        "protocol_sha256": protocol_hash,
        "seal_sha256": seal_hash,
        "independent_receipt_sha256": sha(payload(args.independent_receipt, RESULTS)),
        "files": {name: {key: item[key] for key in ("sha256", "bytes")} for name, item in sorted(selected.items())},
        "omitted_log_receipts": omitted,
        "excluded_categories": [
            "model weights",
            "corpus/query text",
            "raw datasets",
            "index snapshots",
            "private logs",
            "env/auth files",
        ],
    }
    inventory["files"]["README.md"] = metadata(readme)
    inventory_bytes = (json.dumps(inventory, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    total = sum(item["bytes"] for item in selected.values())
    need(shutil.disk_usage(OUTPUT.parent).free > 2 * total + 64 * 1024**2, "Insufficient packaging disk headroom")
    OUTPUT.mkdir(exist_ok=False)
    archive = OUTPUT / "clmkit-relevance-2026-10-08.zip"
    inventory_path = OUTPUT / "PACKAGE_INVENTORY.json"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for name, item in sorted(selected.items()):
            data = payload(item["path"], item["root"])
            verify_entry(data, {key: item[key] for key in ("sha256", "bytes")})
            bundle.writestr(name, data)
        bundle.writestr("README.md", readme)
        bundle.writestr("PACKAGE_INVENTORY.json", inventory_bytes)
    with inventory_path.open("xb") as handle:
        handle.write(inventory_bytes)
    with zipfile.ZipFile(archive) as bundle:
        need(set(bundle.namelist()) == set(inventory["files"]) | {"PACKAGE_INVENTORY.json"}, "Unexpected ZIP member")
        need(len(bundle.namelist()) == len(set(bundle.namelist())), "Duplicate ZIP members")
        for name, entry in inventory["files"].items():
            verify_entry(bundle.read(name), entry)
        need(bundle.read("PACKAGE_INVENTORY.json") == inventory_bytes, "Packaged inventory mismatch")
    checksum = hashlib.sha256()
    with archive.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    with (OUTPUT / "SHA256SUMS").open("x", encoding="ascii") as handle:
        handle.write(f"{checksum.hexdigest()}  {archive.name}\n{sha(inventory_bytes)}  {inventory_path.name}\n")
    print(
        json.dumps(
            {
                "status": "prepared_for_review_not_published",
                "zip": str(archive),
                "sha256": checksum.hexdigest(),
                "files": len(inventory["files"]),
                "bytes": archive.stat().st_size,
            }
        )
    )


if __name__ == "__main__":
    main()
