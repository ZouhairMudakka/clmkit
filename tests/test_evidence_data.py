"""Synthetic fixtures only: never fetch datasets or model weights."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "validation" / "evidence_data.py"
_SPEC = importlib.util.spec_from_file_location("evidence_data", _PATH)
data = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(data)


def banking_fixture():
    return [(f"request {label} number {i}", label) for label in ("card", "cash") for i in range(10)]


def test_banking_split_is_deterministic_grouped_and_stratified():
    train = [*banking_fixture(), ("REQUEST CARD number 0!", "card")]
    first = data.prepare_banking77(train, [("request card number 0", "card")])
    second = data.prepare_banking77(train, [("request card number 0", "card")])
    assert first == second
    gallery = first["corpus"]
    dev = first["queries"]["dev"]
    assert {row["label"] for row in dev} == {"card", "cash"}
    assert not {data.fingerprint(r["text"]) for r in gallery} & {data.fingerprint(r["text"]) for r in dev}
    assert len(dev) in (4, 5)
    assert len(gallery) + len(dev) == len(train)
    assert first["manifest"]["audit"]["cross_split_groups"]
    assert first["manifest"]["processed_counts"]["queries"]["test"] == 1
    for query in dev:
        expected = {r["id"] for r in gallery if r["label"] == query["label"]}
        assert set(first["qrels"]["dev"][query["id"]]) == expected
    for query in first["queries"]["train"]:
        assert query["id"] not in first["qrels"]["train"][query["id"]]


def test_conflicting_duplicates_remain_together_and_are_disclosed():
    train = [*banking_fixture(), ("shared request", "card"), ("request shared!", "cash")]
    bundle = data.prepare_banking77(train, [])
    assert {r["label"] for r in bundle["corpus"] if "shared" in r["text"]} == {"cash", "card"}
    assert len(bundle["manifest"]["audit"]["conflicting_label_groups"]) == 1


def test_clinc_oos_never_indexed_or_assigned_positive_class():
    fixture = {split: [[f"{split} card", "card"], [f"{split} cash", "cash"]] for split in ("train", "val", "test")}
    fixture.update({f"oos_{split}": [[f"unsupported {split}", "oos"]] for split in ("train", "val", "test")})
    bundle = data.prepare_clinc150(fixture)
    assert len(bundle["corpus"]) == 2
    assert all(row["in_scope"] for row in bundle["corpus"])
    for split in ("train", "dev", "test"):
        assert len(bundle["queries"][split]) == 3
        assert len(bundle["qrels"][split]) == 3
        oos = next(q for q in bundle["queries"][split] if not q["in_scope"])
        assert bundle["qrels"][split][oos["id"]] == {}


def test_scifact_keeps_all_documents_and_multiple_judgments_groups_evidence():
    corpus = [{"_id": str(i), "title": f"Title {i}", "text": f"Body {i}"} for i in range(8)]
    queries = [{"_id": str(i), "text": f"claim number {i}"} for i in range(8)]
    train_qrels = {"0": {"0": 1, "1": 2}, "1": {"1": 1}, "2": {"2": 1}, "3": {"3": 1}, "4": {"4": 1}, "5": {"5": 1}}
    test_qrels = {"6": {"5": 1, "6": 1}}
    bundle = data.prepare_scifact(corpus, queries, train_qrels, test_qrels)
    assert len(bundle["corpus"]) == 8
    assert bundle["corpus"][0]["text"] == "Title 0 Body 0"
    assert bundle["qrels"]["test"] == test_qrels
    assert len(bundle["queries"]["dev"]) > 0
    membership = {r["id"]: split for split, records in bundle["queries"].items() for r in records}
    assert membership["0"] == membership["1"]
    assert bundle["qrels"][membership["0"]]["0"] == {"0": 1, "1": 2}
    train_docs = {doc for rels in bundle["qrels"]["train"].values() for doc in rels}
    dev_docs = {doc for rels in bundle["qrels"]["dev"].values() for doc in rels}
    assert not train_docs & dev_docs
    assert bundle["manifest"]["original_counts"]["queries_without_split_qrels"] == 1


def test_scifact_rejects_missing_positive_or_duplicate_ids():
    corpus = [{"_id": "d", "text": "document"}]
    queries = [{"_id": "q", "text": "query"}]
    with pytest.raises(ValueError, match="Missing corpus"):
        data.prepare_scifact(corpus, queries, {"q": {"missing": 1}}, {})
    with pytest.raises(ValueError, match="Duplicate SciFact query"):
        data.prepare_scifact(corpus, queries * 2, {"q": {"d": 1}}, {})
    with pytest.raises(ValueError, match="overlap"):
        data.prepare_scifact(corpus, queries, {"q": {"d": 1}}, {"q": {"d": 1}})


def test_roundtrip_checksums_sealed_test_and_immutable_files(tmp_path):
    bundle = data.prepare_banking77(banking_fixture(), [("new test", "card")])
    manifest = data.write_prepared(bundle, tmp_path)
    assert manifest == data.write_prepared(bundle, tmp_path)
    loaded = data.load_prepared(tmp_path)
    assert loaded["queries"] == bundle["queries"]["dev"]
    with pytest.raises(PermissionError, match="sealed"):
        data.load_prepared(tmp_path, "test")
    assert len(data.load_prepared(tmp_path, "test", allow_test=True)["queries"]) == 1
    (tmp_path / "corpus.jsonl").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        data.load_prepared(tmp_path)
    with pytest.raises(ValueError, match="frozen file"):
        data.write_prepared(bundle, tmp_path)


def test_fetch_guard_prevents_local_download_before_creating_files(tmp_path, monkeypatch):
    monkeypatch.delenv("CODESPACES", raising=False)
    with pytest.raises(RuntimeError, match="Codespace"):
        data.fetch_sources("banking77", tmp_path / "cache")
    assert not (tmp_path / "cache").exists()


def test_source_git_blob_and_md5_checksums_are_verified():
    payload = b"synthetic only\n"
    blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload, usedforsecurity=False).hexdigest()
    spec = {"git_blob_sha1": blob, "expected_bytes": len(payload)}
    result = data._verify_source(payload, spec)
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match="Git blob"):
        data._verify_source(b"synthetic ONLY\n", spec)
    with pytest.raises(ValueError, match="byte count"):
        data._verify_source(b"x", spec)
    with pytest.raises(ValueError, match="MD5"):
        data._verify_source(payload, {"md5": "0" * 32})


def test_qrels_parser_preserves_grades_and_rejects_duplicate_pairs():
    fixture = "query-id\tcorpus-id\tscore\nq\td1\t2\nq\td2\t1\n"
    assert data._read_qrels(fixture) == {"q": {"d1": 2, "d2": 1}}
    with pytest.raises(ValueError, match="Duplicate qrel"):
        data._read_qrels(fixture + "q\td1\t2\n")
