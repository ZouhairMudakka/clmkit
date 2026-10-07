"""Lifecycle compatibility regressions using only tiny deterministic encoders."""

from __future__ import annotations

import json

import pytest

from clmkit import HashingEncoder, Retriever
from clmkit.cli import _load_retriever, build_parser, main
from clmkit.retrieval import EncoderMismatchWarning


def test_changed_config_cannot_mix_vectors_or_relabel_snapshot(tmp_path, monkeypatch):
    encoder = HashingEncoder(dim=16)
    retriever = Retriever(encoder)
    retriever.add(["original document"], ids=["original"])
    snapshot = retriever.save(tmp_path / "original")
    meta_before = (snapshot / "retriever.json").read_bytes()
    encoder.document_template = "changed prefix {text}"

    def unexpected_encode(*args, **kwargs):
        pytest.fail("incompatible state must be rejected before inference")

    monkeypatch.setattr(encoder, "encode", unexpected_encode)
    with pytest.raises(ValueError, match="rebuild"):
        retriever.add(["new document"], ids=["new"])
    with pytest.raises(ValueError, match="rebuild"):
        retriever.save(snapshot)
    assert (snapshot / "retriever.json").read_bytes() == meta_before
    assert retriever.index.ids() == ["original"]


def test_changed_identity_requires_separate_rebuild(tmp_path):
    encoder = HashingEncoder(dim=16)
    original = Retriever(encoder, encoder_identity="weights-v1")
    original.add(["document"], ids=["doc"])
    original.save(tmp_path / "old")
    original.encoder_identity = "weights-v2"
    with pytest.raises(ValueError, match="rebuild"):
        original.validate_encoder()
    rebuilt = Retriever(encoder, encoder_identity="weights-v2")
    rebuilt.add(["document"], ids=["doc"])
    rebuilt.save(tmp_path / "new")
    loaded = Retriever.load(tmp_path / "new", encoder, encoder_identity="weights-v2", strict=True)
    assert loaded.search("document", k=1)[0].id == "doc"
    assert Retriever.read_meta(tmp_path / "old")["encoder_identity"] == "weights-v1"


@pytest.mark.parametrize("command", ["search", "serve", "mcp"])
def test_cli_loaders_reject_mismatch_before_encode_and_allow_explicit_query_override(tmp_path, monkeypatch, command):
    from clmkit import cli

    original = Retriever(HashingEncoder(dim=16))
    original.add(["document"], ids=["doc"])
    original.save(tmp_path, extra_meta={"encoder_spec": {"model": "hashing", "kwargs": {"dim": 16}}})
    changed = HashingEncoder(dim=16, document_template="changed {text}")
    calls = []
    actual_encode = changed.encode

    def counting_encode(*args, **kwargs):
        calls.append(1)
        return actual_encode(*args, **kwargs)

    monkeypatch.setattr(changed, "encode", counting_encode)
    monkeypatch.setattr(cli, "_build_encoder", lambda spec: changed)
    argv = [command, "--index", str(tmp_path)] + (["document"] if command == "search" else [])
    args = build_parser().parse_args(argv)
    with pytest.raises(ValueError, match="Rebuild"):
        _load_retriever(str(tmp_path), args)
    assert calls == []
    args = build_parser().parse_args([*argv, "--allow-encoder-mismatch"])
    with pytest.warns(EncoderMismatchWarning):
        loaded = _load_retriever(str(tmp_path), args)
    assert loaded.search("document", k=1)[0].id == "doc"
    assert calls == [1]
    with pytest.raises(ValueError, match="rebuild"):
        loaded.save(tmp_path)


def test_cli_identity_roundtrip_requires_caller_identity(tmp_path, capsys):
    source = tmp_path / "docs.txt"
    source.write_text("card cash\nmoon stars\n", encoding="utf-8")
    target = tmp_path / "index"
    assert (
        main(
            [
                "index",
                "--model",
                "hashing",
                "--model-arg",
                "dim=16",
                "--input",
                str(source),
                "--output",
                str(target),
                "--encoder-identity",
                "weights-sha256",
                "--instruction",
                "find topic",
            ]
        )
        == 0
    )
    meta = Retriever.read_meta(target)
    assert meta["encoder_identity"] == "weights-sha256"
    assert meta["query_instruction"] == "find topic"
    with pytest.raises(ValueError, match="identities"):
        main(["search", "--index", str(target), "card"])
    capsys.readouterr()
    assert main(["search", "--index", str(target), "card", "--encoder-identity", "weights-sha256", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["text"] == "card cash"


def test_queries_do_not_recompute_fingerprint(tmp_path, monkeypatch):
    encoder = HashingEncoder(dim=16)
    retriever = Retriever(encoder)
    retriever.add(["document"], ids=["doc"])
    retriever.save(tmp_path)

    def unexpected_hash():
        pytest.fail("queries should not hash encoder state")

    monkeypatch.setattr(encoder, "fingerprint", unexpected_hash)
    assert retriever.search("document", k=1)[0].id == "doc"


def test_legacy_shallow_load_remains_warned_and_can_be_resaved(tmp_path):
    encoder = HashingEncoder(dim=16)
    retriever = Retriever(encoder)
    retriever.add(["document"], ids=["doc"])
    retriever.save(tmp_path)
    meta = Retriever.read_meta(tmp_path)
    meta.pop("fingerprint_version")
    meta["encoder"] = f"{encoder.name}:{encoder.dim}"
    (tmp_path / "retriever.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.warns(EncoderMismatchWarning, match="legacy"):
        loaded = Retriever.load(tmp_path, encoder, strict=True)
    loaded.save(tmp_path / "upgraded")
    assert Retriever.read_meta(tmp_path / "upgraded")["encoder"].startswith("v2:")
