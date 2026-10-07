from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from clmkit import HashingEncoder, Retriever
from clmkit.cli import main


def test_cli_info(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert "hashing" in info["encoders"] and "torch" in info["backends"]


def test_cli_encode_index_search_eval_mine(tmp_path, capsys: pytest.CaptureFixture[str], toy_examples) -> None:  # type: ignore[no-untyped-def]
    from clmkit.data import save_examples

    # encode -> .npy and JSONL
    assert (
        main(
            [
                "encode",
                "--model",
                "hashing",
                "--model-arg",
                "dim=32",
                "hello",
                "world",
                "--output",
                str(tmp_path / "e.npy"),
            ]
        )
        == 0
    )
    assert np.load(tmp_path / "e.npy").shape == (2, 32)
    assert main(["encode", "--model", "hashing", "--model-arg", "dim=8", "--kind", "query", "hi"]) == 0
    assert len(json.loads(capsys.readouterr().out.splitlines()[0])["embedding"]) == 8

    # index a JSONL corpus with ids + metadata + chunking, then search without --model
    docs = tmp_path / "docs.jsonl"
    docs.write_text(
        "\n".join(
            json.dumps({"id": f"doc{i}", "text": p, "src": "toy"})
            for i, (_, p) in enumerate([(e.query, e.positive) for e in toy_examples])
        )
    )
    idx = tmp_path / "idx"
    assert (
        main(
            [
                "index",
                "--model",
                "hashing",
                "--model-arg",
                "dim=512",
                "--input",
                str(docs),
                "--output",
                str(idx),
                "--chunk-size",
                "500",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["search", "--index", str(idx), "what do cats eat", "-k", "2", "--json"]) == 0
    hits = json.loads(capsys.readouterr().out)
    assert hits[0]["id"] == "doc0#0" and hits[0]["metadata"]["src"] == "toy"
    assert hits[0]["metadata"]["parent_id"] == "doc0"
    assert main(["search", "--index", str(idx), "black hole gravity"]) == 0
    assert "doc5#0" in capsys.readouterr().out

    # eval + mine
    train = save_examples(tmp_path / "train.jsonl", toy_examples)
    assert (
        main(["eval", "--model", "hashing", "--data", str(train), "--ks", "1,5", "--output", str(tmp_path / "m.json")])
        == 0
    )
    assert json.loads((tmp_path / "m.json").read_text())["recall@5"] > 0.5
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(e.positive for e in toy_examples))
    assert (
        main(
            [
                "mine",
                "--model",
                "hashing",
                "--train",
                str(train),
                "--corpus",
                str(corpus),
                "--output",
                str(tmp_path / "mined.jsonl"),
                "-n",
                "2",
            ]
        )
        == 0
    )
    mined = [json.loads(line) for line in (tmp_path / "mined.jsonl").read_text().splitlines()]
    assert all(len(m["negatives"]) <= 2 and m["positive"] not in m["negatives"] for m in mined)


def test_cli_encode_from_stdin(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import io

    monkeypatch.setattr(sys, "stdin", io.StringIO("line one\n\nline two\n"))
    assert main(["encode", "--model", "hashing", "--model-arg", "dim=4", "--input", "-"]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 2


def test_cli_errors(tmp_path) -> None:
    with pytest.raises(SystemExit):
        main(["encode", "--model", "hashing", "--model-arg", "novalue", "x"])
    with pytest.raises(SystemExit):
        main(["serve"])
    with pytest.raises(SystemExit):
        main(["eval", "--model", "hashing"])
    enc = HashingEncoder(dim=16)
    r = Retriever(enc)
    r.add(["x"])
    r.save(tmp_path / "nospec")
    with pytest.raises(SystemExit, match="encoder spec"):
        main(["search", "--index", str(tmp_path / "nospec"), "x"])


@pytest.mark.parametrize("override", [None, "shared task"])
def test_cli_eval_preserves_instructions_and_batch_size(tmp_path, monkeypatch, override) -> None:
    from clmkit.data import ContrastiveExample, save_examples

    encoder = HashingEncoder(dim=16)
    calls = []
    encode = encoder.encode

    def spy(texts, **kwargs):
        calls.append(kwargs)
        return encode(texts, **kwargs)

    monkeypatch.setattr(encoder, "encode", spy)
    monkeypatch.setattr("clmkit.cli._build_encoder", lambda spec: encoder)
    data = save_examples(
        tmp_path / "eval.jsonl",
        [
            ContrastiveExample("refund", "refund policy", instruction="support task"),
            ContrastiveExample("password", "password reset", instruction="security task"),
        ],
    )
    args = ["eval", "--model", "hashing", "--data", str(data), "--batch-size", "1"]
    if override is not None:
        args.extend(["--instruction", override])
    assert main(args) == 0
    assert all(call["batch_size"] == 1 for call in calls)
    query_call = next(call for call in calls if call["kind"] == "query")
    assert query_call["instruction"] == (override or ["support task", "security task"])


def test_cli_mine_honors_batch_size(tmp_path, monkeypatch, toy_examples) -> None:
    from clmkit.data import save_examples

    encoder = HashingEncoder(dim=16)
    batches = []
    encode = encoder.encode

    def spy(texts, **kwargs):
        batches.append(kwargs["batch_size"])
        return encode(texts, **kwargs)

    monkeypatch.setattr(encoder, "encode", spy)
    monkeypatch.setattr("clmkit.cli._build_encoder", lambda spec: encoder)
    data = save_examples(tmp_path / "train.jsonl", toy_examples)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(e.positive for e in toy_examples))
    assert (
        main(
            [
                "mine",
                "--model",
                "hashing",
                "--train",
                str(data),
                "--corpus",
                str(corpus),
                "--output",
                str(tmp_path / "mined.jsonl"),
                "--batch-size",
                "1",
            ]
        )
        == 0
    )
    assert batches and all(batch == 1 for batch in batches)


def test_cli_rejects_model_args_without_model(capsys) -> None:
    with pytest.raises(SystemExit):
        main(["search", "--index", "unused", "query", "--model-arg", "device=cpu"])
    assert "--model-arg requires --model" in capsys.readouterr().err


@pytest.mark.parametrize(
    "option",
    [
        "api_key=literal-audit-secret",
        'headers={"Authorization":"Bearer literal-audit-secret"}',
        'headers={"X-Custom-Auth":"literal-audit-secret"}',
        'model_kwargs={"token":"literal-audit-secret"}',
        'tokenizer_kwargs={"token":"literal-audit-secret"}',
        'model_kwargs={"use_auth_token":"literal-audit-secret"}',
        "base_url=https://user:literal-audit-secret@example.invalid/v1",
        "base_url=https://example.invalid/v1?token=literal-audit-secret",
    ],
)
def test_cli_index_rejects_credentials_before_loading(tmp_path, monkeypatch, option) -> None:
    def unexpected_load(spec):
        pytest.fail("encoder loading must not run for a credential-bearing persisted spec")

    monkeypatch.setattr("clmkit.cli._build_encoder", unexpected_load)
    corpus = tmp_path / "docs.txt"
    corpus.write_text("refund policy")
    output = tmp_path / "index"
    with pytest.raises(SystemExit, match="api_key_env") as error:
        main(
            [
                "index",
                "--model",
                "openai:fixture",
                "--model-arg",
                option,
                "--input",
                str(corpus),
                "--output",
                str(output),
            ]
        )
    assert "literal-audit-secret" not in str(error.value)
    assert not output.exists()
    assert all(b"literal-audit-secret" not in path.read_bytes() for path in tmp_path.rglob("*") if path.is_file())


def test_cli_index_persists_environment_reference_without_value(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AUDIT_EMBEDDING_KEY", "literal-audit-secret")
    monkeypatch.setattr("clmkit.cli._build_encoder", lambda spec: HashingEncoder(dim=8))
    corpus = tmp_path / "docs.txt"
    corpus.write_text("refund policy")
    output = tmp_path / "index"
    assert (
        main(
            [
                "index",
                "--model",
                "openai:fixture",
                "--model-arg",
                "api_key_env=AUDIT_EMBEDDING_KEY",
                "--input",
                str(corpus),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    spec = Retriever.read_meta(output)["encoder_spec"]
    assert spec["kwargs"]["api_key_env"] == "AUDIT_EMBEDDING_KEY"
    assert all(b"literal-audit-secret" not in path.read_bytes() for path in output.rglob("*") if path.is_file())


@pytest.mark.parametrize("token", [True, False])
def test_cli_persisted_spec_allows_hf_auth_selection(token) -> None:
    from clmkit.cli import _validate_persisted_encoder_spec

    _validate_persisted_encoder_spec(
        {
            "model": "fixture",
            "kwargs": {
                "model_kwargs": {"token": token},
                "tokenizer_kwargs": {"token": token},
            },
        }
    )


def test_cli_serve_rejects_stored_credentials_before_loading(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    from clmkit.cli import cmd_serve

    def unexpected_load(*args):
        pytest.fail("encoder loading must not run before persisted-spec validation")

    monkeypatch.setattr("clmkit.utils.require", lambda module: SimpleNamespace())
    monkeypatch.setattr("clmkit.cli._load_retriever", unexpected_load)
    monkeypatch.setattr(
        Retriever,
        "read_meta",
        lambda path: {
            "encoder_spec": {"model": "openai:fixture", "kwargs": {"api_key": "literal-audit-secret"}},
        },
    )
    args = SimpleNamespace(index=str(tmp_path), model=None, allow_writes=True, save_on_exit=True)
    with pytest.raises(SystemExit, match="api_key_env") as error:
        cmd_serve(args)
    assert "literal-audit-secret" not in str(error.value)


def test_cli_read_only_search_keeps_legacy_auth_spec(tmp_path, monkeypatch, capsys) -> None:
    encoder = HashingEncoder(dim=8)
    retriever = Retriever(encoder)
    retriever.add(["refund policy"])
    retriever.save(tmp_path)
    spec = {"model": "openai:fixture", "kwargs": {"api_key": "literal-audit-secret"}}
    monkeypatch.setattr(Retriever, "read_meta", lambda path: {"encoder_spec": spec})
    seen = []

    def build(received):
        seen.append(received)
        return encoder

    monkeypatch.setattr("clmkit.cli._build_encoder", build)
    assert main(["search", "--index", str(tmp_path), "refund"]) == 0
    assert seen == [spec]
    assert "literal-audit-secret" not in capsys.readouterr().out
