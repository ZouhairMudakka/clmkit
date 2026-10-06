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


@pytest.mark.parametrize("output_dim", [None, 16])
def test_output_dim_roundtrip_and_overrides(tmp_path, tiny_model_dir, output_dim) -> None:  # type: ignore[no-untyped-def]
    enc = _encoder(tiny_model_dir, output_dim=output_dim)
    saved = enc.save_pretrained(tmp_path / "dimension")
    loaded = HFEncoder(str(saved), device="cpu")
    assert loaded.output_dim == output_dim
    assert loaded.encode("cats").shape == (output_dim or 32,)
    np.testing.assert_allclose(loaded.encode("cats"), enc.encode("cats"), atol=1e-6)
    assert HFEncoder(str(saved), device="cpu", output_dim=8).dim == 8
    assert HFEncoder(str(saved), device="cpu", output_dim=None).dim == 32
    # Old checkpoints omit output_dim and continue to reload at the native dimension.
    config = json.loads((saved / CONFIG_FILENAME).read_text())
    config.pop("output_dim")
    (saved / CONFIG_FILENAME).write_text(json.dumps(config))
    assert HFEncoder(str(saved), device="cpu").dim == 32


@pytest.mark.parametrize("merge", [False, True])
def test_truncated_adapter_roundtrip(tmp_path, tiny_model_dir, merge) -> None:  # type: ignore[no-untyped-def]
    peft = pytest.importorskip("peft")
    enc = _encoder(tiny_model_dir, output_dim=16)
    enc.model = peft.get_peft_model(enc.model, peft.LoraConfig(r=2, target_modules=["q_proj", "v_proj"]))
    enc.model.eval()
    expected = enc.encode("cats")
    saved = enc.save_pretrained(tmp_path / "adapter", merge_adapter=merge)
    loaded = HFEncoder(str(saved), device="cpu")
    assert loaded.dim == 16
    np.testing.assert_allclose(loaded.encode("cats"), expected, atol=1e-6)


@pytest.mark.parametrize("kind", ["encoder", "causal", "cross"])
def test_model_load_rejects_bin_without_deserialization(tmp_path, tiny_hf_parts, monkeypatch, kind) -> None:  # type: ignore[no-untyped-def]
    from transformers import Qwen3ForCausalLM, Qwen3ForSequenceClassification, Qwen3Model

    from clmkit.rerank import CrossEncoderReranker

    config, tokenizer = tiny_hf_parts
    classes = {"encoder": Qwen3Model, "causal": Qwen3ForCausalLM, "cross": Qwen3ForSequenceClassification}
    model = classes[kind](config())
    model.config.save_pretrained(tmp_path)
    tokenizer().save_pretrained(tmp_path)
    # Harmless bytes under the legacy filename: no executable pickle payload.
    (tmp_path / "pytorch_model.bin").write_bytes(b"not a checkpoint")

    def forbidden(*args, **kwargs):  # type: ignore[no-untyped-def]
        pytest.fail("legacy weights must never reach torch.load")

    monkeypatch.setattr(torch, "load", forbidden)
    loader = {"encoder": HFEncoder, "causal": LLMYesNoReranker, "cross": CrossEncoderReranker}[kind]
    with pytest.raises(OSError, match="safetensors"):
        loader(str(tmp_path), device="cpu")
    model.save_pretrained(tmp_path, safe_serialization=True)
    loaded = loader(str(tmp_path), device="cpu")
    assert loaded.model is not None


@pytest.mark.parametrize(
    "kwargs", [{"use_safetensors": False}, {"use_safetensors": None}, {"from_tf": True}, {"adapter_kwargs": {}}]
)
def test_model_kwargs_cannot_disable_safe_loading(tiny_model_dir, kwargs) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="safetensors"):
        _encoder(tiny_model_dir, model_kwargs=kwargs)


def test_adapter_load_rejects_bin(tmp_path, tiny_model_dir, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    peft = pytest.importorskip("peft")
    enc = _encoder(tiny_model_dir)
    enc.model = peft.get_peft_model(enc.model, peft.LoraConfig(r=2, target_modules=["q_proj"]))
    saved = enc.save_pretrained(tmp_path / "adapter")
    (saved / "adapter_model.safetensors").unlink()
    (saved / "adapter_model.bin").write_bytes(b"not a checkpoint")

    def forbidden(*args, **kwargs):  # type: ignore[no-untyped-def]
        pytest.fail("legacy adapter weights must never reach torch.load")

    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match=r"adapter_model\.safetensors"):
        HFEncoder(str(saved), device="cpu")
    with pytest.raises(ValueError, match=r"adapter_model\.safetensors"):
        _encoder(tiny_model_dir, adapter=str(saved))


def test_safe_hub_adapter_is_pinned_without_bin_fallback(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import huggingface_hub

    from clmkit.encoders.hf import _safe_adapter_path

    snapshot = tmp_path / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    calls = []

    def download(repo_id, filename, revision):  # type: ignore[no-untyped-def]
        calls.append((repo_id, filename, revision))
        return str(snapshot / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert _safe_adapter_path("test/adapter", "release") == str(snapshot)
    assert calls == [
        ("test/adapter", "adapter_model.safetensors", "release"),
        ("test/adapter", "adapter_config.json", "a" * 40),
    ]


def test_implicit_adapter_loading_is_rejected(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from clmkit.encoders.hf import _safe_model_kwargs

    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "adapter_model.bin").write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="implicit adapter"):
        _safe_model_kwargs(str(tmp_path), None)
    with pytest.raises(ValueError, match="implicit adapter"):
        _safe_model_kwargs(str(tmp_path), None, {"subfolder": "base"})


def test_safe_hub_model_pins_adapter_check_to_model_commit(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import transformers

    from clmkit.encoders.hf import _safe_model_kwargs

    commit = "b" * 40
    monkeypatch.setattr(transformers.utils.hub, "cached_file", lambda *a, **kw: f"/snapshots/{commit}/config.json")
    calls = []

    def find_adapter(path, **kwargs):  # type: ignore[no-untyped-def]
        calls.append((path, kwargs))

    monkeypatch.setattr(transformers.utils, "find_adapter_config_file", find_adapter)
    options = _safe_model_kwargs("test/model", "main")
    assert options["use_safetensors"] is True
    assert options["_commit_hash"] == commit
    assert options["adapter_kwargs"]["revision"] == commit
    assert calls == [("test/model", {"revision": commit})]


@pytest.mark.parametrize(
    "changed", [{"pooling": "mean"}, {"revision": "pinned"}, {"max_length": 12}, {"ensure_eos": False}]
)
def test_hf_fingerprint_tracks_inference_configuration(tiny_model_dir, changed) -> None:  # type: ignore[no-untyped-def]
    assert _encoder(tiny_model_dir).fingerprint() != _encoder(tiny_model_dir, **changed).fingerprint()


def test_hf_fingerprint_stable_after_tokenization(tiny_model_dir) -> None:  # type: ignore[no-untyped-def]
    enc = _encoder(tiny_model_dir)
    expected = enc.fingerprint()
    enc.encode(["cats", "cats eat fish"])
    assert enc.fingerprint() == expected
    enc.tokenizer.truncation_side = "left"
    assert enc.fingerprint() != expected


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
