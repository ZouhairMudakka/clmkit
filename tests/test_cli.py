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
