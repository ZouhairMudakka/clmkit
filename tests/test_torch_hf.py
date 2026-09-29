"""HFEncoder and HF rerankers against tiny, locally built Qwen3 models (no downloads)."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from clmkit import Retriever, load_encoder  # noqa: E402
from clmkit.encoders.hf import HFEncoder, resolve_torch_dtype  # noqa: E402
from clmkit.encoders.presets import CONFIG_FILENAME  # noqa: E402
from clmkit.rerank import LLMYesNoReranker, load_reranker  # noqa: E402

pytestmark = pytest.mark.hf

QWEN_LIKE = {
    "pooling": "last_token",
    "padding_side": "left",
    "ensure_eos": True,
    "eos_token": "<|endoftext|>",
    "query_template": "Instruct: {instruction}\nQuery:{text}",
    "default_instruction": "find the document",
}


def _encoder(path, **kw):  # type: ignore[no-untyped-def]
    return HFEncoder(str(path), device="cpu", **{**QWEN_LIKE, **kw})


def test_encode_shapes_and_padding_invariance(tiny_model_dir) -> None:  # type: ignore[no-untyped-def]
    enc = _encoder(tiny_model_dir)
    assert enc.native_dim == 32 and enc.dim == 32
    texts = ["cats eat fish", "the moon orbits the earth at a large distance in space", "tea"]
    batch = enc.encode(texts, batch_size=3)
    single = np.stack([enc.encode(t) for t in texts])
    assert batch.shape == (3, 32)
    # left padding + last-token pooling + masking => batching must not change embeddings
    np.testing.assert_allclose(batch, single, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(batch, axis=1), 1.0, rtol=1e-5)
    # right padding must give the same result too (pooling is padding-agnostic)
    right = _encoder(tiny_model_dir, padding_side="right").encode(texts)
    np.testing.assert_allclose(right, batch, atol=1e-5)


def test_eos_is_appended_once_and_is_the_pooled_token(tiny_model_dir) -> None:  # type: ignore[no-untyped-def]
    enc = _encoder(tiny_model_dir)
    eos = enc.tokenizer.convert_tokens_to_ids("<|endoftext|>")
    batch = enc.tokenize(["cats eat fish", "tea <|endoftext|>"])
    ids = batch["input_ids"]
    assert (ids[:, -1] == eos).all()  # left padded: last column is always the EOS
    assert int((ids[1] == eos).sum()) == 1  # not doubled when already present
    no_eos = _encoder(tiny_model_dir, ensure_eos=False).tokenize(["cats eat fish"])["input_ids"]
    assert int(no_eos[0, -1]) != eos
    with pytest.raises(ValueError, match="not in the vocabulary"):
        _encoder(tiny_model_dir, eos_token="<not-a-token>")


def test_query_instruction_changes_query_but_not_document(tiny_model_dir) -> None:  # type: ignore[no-untyped-def]
    enc = _encoder(tiny_model_dir)
    doc = enc.encode("what do cats eat", kind="document")
    q = enc.encode("what do cats eat", kind="query")
    q2 = enc.encode("what do cats eat", kind="query", instruction="retrieve relevant passages")
    assert not np.allclose(doc, q) and not np.allclose(q, q2)


def test_truncation_mrl_dtype_and_misc(tmp_path, tiny_model_dir) -> None:  # type: ignore[no-untyped-def]
    import shutil

    enc = _encoder(tiny_model_dir, max_length=6, output_dim=16)
    assert enc.encode("the " * 50).shape == (16,)
    assert enc.tokenize(["the " * 50])["input_ids"].shape[1] == 6
    ckpt = shutil.copytree(tiny_model_dir, tmp_path / "mrl")
    (ckpt / CONFIG_FILENAME).write_text(json.dumps({**QWEN_LIKE, "mrl_range": [16, 32]}))
    with pytest.warns(UserWarning, match="Matryoshka"):
        HFEncoder(str(ckpt), device="cpu", output_dim=8)
    assert resolve_torch_dtype("bf16", "cpu") == torch.bfloat16
    assert resolve_torch_dtype("auto", "cpu") == torch.float32
    with pytest.raises(ValueError, match="unsupported dtype"):
        resolve_torch_dtype("int4", "cpu")
    with pytest.raises(ValueError, match="model_name_or_path"):
        HFEncoder()
    for pooling in ("mean", "cls"):
        assert _encoder(tiny_model_dir, pooling=pooling).encode("cats").shape == (32,)


def test_forward_is_differentiable(tiny_model_dir) -> None:  # type: ignore[no-untyped-def]
    enc = _encoder(tiny_model_dir)
    out = enc.forward(["cats eat fish", "tea"], "query")
    assert out.requires_grad and out.shape == (2, 32)
    out.sum().backward()
    assert any(p.grad is not None for p in enc.model.parameters())


def test_save_and_reload_roundtrip(tmp_path, tiny_model_dir) -> None:  # type: ignore[no-untyped-def]
    enc = _encoder(tiny_model_dir, max_length=64)
    saved = enc.save_pretrained(tmp_path / "ckpt")
    cfg = json.loads((saved / CONFIG_FILENAME).read_text())
    assert cfg["pooling"] == "last_token" and cfg["eos_token"] == "<|endoftext|>" and cfg["max_length"] == 64
    reloaded = load_encoder(str(saved), device="cpu")  # settings come from clmkit_config.json
    assert reloaded.pooling == "last_token" and reloaded.padding_side == "left"
    np.testing.assert_allclose(
        reloaded.encode(["cats eat fish"], kind="query"), enc.encode(["cats eat fish"], kind="query"), atol=1e-6
    )


def test_hf_encoder_in_a_retriever(tiny_model_dir) -> None:  # type: ignore[no-untyped-def]
    r = Retriever(_encoder(tiny_model_dir))
    r.add(["cats eat fish", "the sun is hot", "python is a programming language"])
    hits = r.search("cats eat fish", k=3)
    assert len(hits) == 3 and hits[0].score >= hits[-1].score


def test_llm_yes_no_reranker(tiny_causal_lm_dir) -> None:  # type: ignore[no-untyped-def]
    rr = LLMYesNoReranker(str(tiny_causal_lm_dir), device="cpu", max_length=64)
    scores = rr.score("what do cats eat", ["cats eat fish", "the sun is hot", "tea"], batch_size=2)
    assert scores.shape == (3,) and ((scores > 0) & (scores < 1)).all()
    # batching must not change scores (left padding => last position is real)
    one_by_one = np.array([rr.score("what do cats eat", [d])[0] for d in ["cats eat fish", "the sun is hot", "tea"]])
    np.testing.assert_allclose(scores, one_by_one, atol=1e-5)
    assert isinstance(load_reranker(str(tiny_causal_lm_dir), yes_no=True, device="cpu"), LLMYesNoReranker)
    with pytest.raises(ValueError, match="too small"):
        LLMYesNoReranker(str(tiny_causal_lm_dir), device="cpu", max_length=4).score("q", ["d"])


def test_cross_encoder_reranker(tiny_hf_parts) -> None:  # type: ignore[no-untyped-def]
    from transformers import Qwen3ForSequenceClassification

    from clmkit.rerank import CrossEncoderReranker

    tiny_config, tiny_tokenizer = tiny_hf_parts

    torch.manual_seed(0)
    for num_labels in (1, 2):
        cfg = tiny_config()
        cfg.num_labels = num_labels
        model = Qwen3ForSequenceClassification(cfg)
        rr = CrossEncoderReranker(model=model, tokenizer=tiny_tokenizer(), device="cpu", max_length=32)
        scores = rr.score("what do cats eat", ["cats eat fish", "the sun is hot", "tea"], batch_size=2)
        assert scores.shape == (3,) and ((scores >= 0) & (scores <= 1)).all()
