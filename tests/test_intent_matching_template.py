"""Tiny fixture checks for example mechanics, never semantic benchmark evidence."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from clmkit import HashingEncoder, Retriever

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation.evidence_data import prepare_banking77, prepare_clinc150, write_prepared

SCRIPT = Path(__file__).resolve().parents[1] / "templates" / "intent_matching" / "intent_matching.py"
spec = importlib.util.spec_from_file_location("intent_matching_example", SCRIPT)
assert spec is not None and spec.loader is not None
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


@pytest.fixture
def galleries(tmp_path):
    root = tmp_path / "data"
    banking = prepare_banking77(
        [(f"{label} request {i}", label) for label in ("card", "cash") for i in range(10)],
        [("held-out fixture must never be opened", "card")],
    )
    clinc = prepare_clinc150(
        {
            "train": [["card support", "card"], ["cash support", "cash"]],
            "val": [["card request", "card"]],
            "test": [["held-out example", "card"]],
            "oos_train": [["outside train", "oos"]],
            "oos_val": [["outside dev", "oos"]],
            "oos_test": [["outside test", "oos"]],
        }
    )
    for name, bundle in [("banking77", banking), ("clinc150", clinc)]:
        write_prepared(bundle, root / name)
        # Missing sealed test files must not prevent using a prepared gallery.
        (root / name / "queries" / "test.jsonl").unlink()
        (root / name / "qrels" / "test.json").unlink()
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "dataset_manifests": {
                    name: example.digest(root / name / "manifest.json") for name in ("banking77", "clinc150")
                }
            }
        ),
        encoding="utf-8",
    )
    return root, protocol


def test_full_gallery_build_strict_reload_and_match_without_opening_test(galleries, tmp_path):
    root, protocol = galleries
    data, contract = example.prepared_gallery(root, protocol, "banking77")
    encoder = HashingEncoder(dim=32)
    path = tmp_path / "index"
    built = example.build_index(path, encoder, "synthetic-v1", data["corpus"], contract)
    loaded = example.open_index(path, encoder, "synthetic-v1", data["corpus"], contract)
    result = example.match(loaded, "card request", "banking77")
    assert len(built) == result["gallery_documents"] == 16
    assert result["encoder_identity"] == "synthetic-v1"
    assert result["action_executed"] is False
    assert {hit["id"] for hit in result["matches"]} <= {row["id"] for row in data["corpus"]}
    assert all(hit["label"] in {"card", "cash"} for hit in result["matches"])
    with pytest.raises(FileExistsError, match="fresh"):
        example.build_index(path, encoder, "synthetic-v1", data["corpus"], contract)


@pytest.mark.parametrize("change", ["identity", "config", "gallery"])
def test_wrong_snapshot_contract_rejected_before_query_inference(galleries, tmp_path, monkeypatch, change):
    root, protocol = galleries
    data, contract = example.prepared_gallery(root, protocol, "banking77")
    encoder = HashingEncoder(dim=32)
    path = tmp_path / "index"
    example.build_index(path, encoder, "synthetic-v1", data["corpus"], contract)
    identity = "synthetic-v2" if change == "identity" else "synthetic-v1"
    if change == "config":
        encoder.document_template = "changed {text}"
    if change == "gallery":
        contract = {**contract, "corpus_sha256": "different"}

    def no_inference(*args, **kwargs):
        pytest.fail("mismatch must be rejected before query inference")

    monkeypatch.setattr(encoder, "encode", no_inference)
    with pytest.raises(ValueError):
        example.open_index(path, encoder, identity, data["corpus"], contract)


def test_gallery_checksum_and_full_count_are_required(galleries, tmp_path):
    root, protocol = galleries
    data, contract = example.prepared_gallery(root, protocol, "banking77")
    with pytest.raises(ValueError, match="partial"):
        example.build_index(tmp_path / "partial", HashingEncoder(dim=16), "synthetic", data["corpus"][:1], contract)
    with (root / "banking77" / "corpus.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        example.prepared_gallery(root, protocol, "banking77")


def threshold_files(tmp_path, contract, **changes):
    payload = {
        "dataset": "clinc150",
        "method": "dense",
        "model": "minilm",
        "encoder_identity": "synthetic-v1",
        "protocol_sha256": contract["protocol_sha256"],
        "manifest_sha256": contract["manifest_sha256"],
        "query_limit": 0,
        "selection_split": "dev",
        "selected": {"threshold": 0.5},
        **changes,
    }
    threshold = tmp_path / "threshold.json"
    threshold.write_text(json.dumps(payload), encoding="utf-8")
    seal = tmp_path / "seal.json"
    seal.write_text(
        json.dumps(
            {
                "schema": 2,
                "frozen_before_test": True,
                "protocol_sha256": contract["protocol_sha256"],
                "threshold_files": {str(threshold): example.digest(threshold)},
            }
        ),
        encoding="utf-8",
    )
    return threshold, seal


def test_clinc_uses_own_in_scope_gallery_and_sealed_threshold(galleries, tmp_path):
    root, protocol = galleries
    data, contract = example.prepared_gallery(root, protocol, "clinc150")
    assert len(data["corpus"]) == 2
    assert all(row["in_scope"] for row in data["corpus"])
    encoder = HashingEncoder(dim=32)
    retriever = example.build_index(tmp_path / "clinc", encoder, "synthetic-v1", data["corpus"], contract)
    path, seal = threshold_files(tmp_path, contract)
    value = example.calibrated_threshold(path, seal, contract, "synthetic-v1")
    assert value == 0.5
    assert example.match(retriever, "card support", "clinc150", threshold=value)["status"] == "accepted"
    rejected = example.match(retriever, "", "clinc150", threshold=value)
    assert rejected["status"] == "rejected"
    assert rejected["intent"] is None
    with pytest.raises(ValueError, match="requires"):
        example.match(retriever, "card", "clinc150")
    with pytest.raises(ValueError, match="BANKING77"):
        example.match(retriever, "card", "banking77", threshold=value)


@pytest.mark.parametrize(
    "changes",
    [
        {"dataset": "banking77"},
        {"encoder_identity": "other-model"},
        {"model": "qwen06"},
        {"query_limit": 100},
        {"manifest_sha256": "other-gallery"},
        {"method": "bm25"},
    ],
)
def test_calibration_rejects_wrong_dataset_model_gallery_or_subset(galleries, tmp_path, changes):
    root, protocol = galleries
    _, contract = example.prepared_gallery(root, protocol, "clinc150")
    path, seal = threshold_files(tmp_path, contract, **changes)
    with pytest.raises(ValueError, match="does not match"):
        example.calibrated_threshold(path, seal, contract, "synthetic-v1")


def test_threshold_tamper_is_rejected_and_cannot_transfer_to_banking(galleries, tmp_path):
    root, protocol = galleries
    _, contract = example.prepared_gallery(root, protocol, "clinc150")
    path, seal = threshold_files(tmp_path, contract)
    path.write_text(path.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen"):
        example.calibrated_threshold(path, seal, contract, "synthetic-v1")
    _, banking_contract = example.prepared_gallery(root, protocol, "banking77")
    with pytest.raises(ValueError, match="BANKING77"):
        example.calibrated_threshold(path, seal, banking_contract, "synthetic-v1")


def test_failed_staged_reload_preserves_previous_index(galleries, tmp_path, monkeypatch):
    root, protocol = galleries
    data, contract = example.prepared_gallery(root, protocol, "banking77")
    encoder = HashingEncoder(dim=16)
    old = tmp_path / "old"
    example.build_index(old, encoder, "synthetic-v1", data["corpus"], contract)
    saved = (old / "retriever.json").read_bytes()

    def fail_reload(*args, **kwargs):
        raise ValueError("injected reload failure")

    monkeypatch.setattr(Retriever, "load", fail_reload)
    target = tmp_path / "new"
    with pytest.raises(ValueError, match="reload failure"):
        example.build_index(target, encoder, "synthetic-v2", data["corpus"], contract)
    assert not target.exists()
    assert (old / "retriever.json").read_bytes() == saved
