from __future__ import annotations

import numpy as np
import pytest

from clmkit import Encoder, HashingEncoder, load_encoder
from clmkit.encoders.presets import CONFIG_FILENAME, DEFAULT_PRESET, ModelPreset, resolve_preset


def test_hashing_shapes_normalisation_and_determinism(hashing: HashingEncoder) -> None:
    one = hashing.encode("hello world")
    many = hashing.encode(["hello world", "another text"])
    assert one.shape == (256,) and many.shape == (2, 256)
    assert one.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(many, axis=1), 1.0, rtol=1e-5)
    np.testing.assert_array_equal(one, many[0])
    np.testing.assert_array_equal(one, HashingEncoder(dim=256).encode("hello world"))  # stable across instances


def test_hashing_semantics_lexical_overlap(hashing: HashingEncoder) -> None:
    q = hashing.encode("capital of france", kind="query")
    docs = hashing.encode(["paris is the capital of france", "bananas are yellow"])
    sims = docs @ q
    assert sims[0] > sims[1]
    # character n-grams give some robustness to typos
    assert float(hashing.encode("capitol of frnace") @ q) > 0.3


def test_empty_input_and_type_errors(hashing: HashingEncoder) -> None:
    assert hashing.encode([]).shape == (0, 256)
    with pytest.raises(TypeError, match="expects strings"):
        hashing.encode(["ok", 3])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="kind"):
        hashing.encode("x", kind="passage")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        HashingEncoder(dim=1)


def test_matryoshka_truncation(hashing: HashingEncoder) -> None:
    emb = hashing.encode(["a b c", "d e f"], dim=64)
    assert emb.shape == (2, 64)
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, rtol=1e-5)
    with pytest.raises(ValueError, match="exceeds"):
        hashing.encode("x", dim=1024)
    enc = HashingEncoder(dim=128, output_dim=32)
    assert enc.dim == 32 and enc.encode("x").shape == (32,)
    assert enc.fingerprint().startswith("v2:")


def test_unnormalised_output(hashing: HashingEncoder) -> None:
    raw = hashing.encode("the the the the", normalize=False)
    assert np.linalg.norm(raw) > 1.0


@pytest.mark.parametrize(
    "changed",
    [
        {"lowercase": False},
        {"word_ngrams": (1, 1)},
        {"char_ngrams": None},
        {"char_weight": 0.25},
        {"document_template": "passage: {text}"},
        {"query_template": "query: {text}"},
        {"default_instruction": "search"},
        {"output_dim": 16},
    ],
)
def test_fingerprint_tracks_hashing_configuration(changed) -> None:  # type: ignore[no-untyped-def]
    baseline = HashingEncoder(dim=32)
    assert baseline.fingerprint() == HashingEncoder(dim=32).fingerprint()
    assert baseline.fingerprint() != HashingEncoder(dim=32, **changed).fingerprint()


def test_prompt_templates_and_instructions() -> None:
    enc = HashingEncoder(
        dim=64, query_template="Instruct: {instruction}\nQuery:{text}", default_instruction="find docs"
    )
    assert enc.format_texts(["q"], "query") == ["Instruct: find docs\nQuery:q"]
    assert enc.format_texts(["q"], "query", "custom") == ["Instruct: custom\nQuery:q"]
    assert enc.format_texts(["q1", "q2"], "query", ["a", None]) == [
        "Instruct: a\nQuery:q1",
        "Instruct: find docs\nQuery:q2",
    ]
    assert enc.format_texts(["d"], "document") == ["d"]
    with pytest.raises(ValueError, match="match the number"):
        enc.format_texts(["a", "b"], "query", ["only one"])
    # a template requiring an instruction degrades gracefully when none exists
    bare = HashingEncoder(dim=64, query_template="Instruct: {instruction}\nQuery:{text}")
    assert bare.format_texts(["q"], "query") == ["q"]
    with pytest.raises(ValueError, match="text"):
        HashingEncoder(dim=64, query_template="no placeholder")


def test_similarity_helper(hashing: HashingEncoder) -> None:
    a = hashing.encode(["x y", "z"])
    sim = Encoder.similarity(a, a)
    np.testing.assert_allclose(np.diag(sim), 1.0, rtol=1e-5)


def test_custom_encoder_subclass_contract() -> None:
    class Broken(Encoder):
        name = "broken"

        @property
        def native_dim(self) -> int:
            return 4

        def _encode(self, texts, batch_size):  # type: ignore[no-untyped-def]
            return np.zeros((len(texts), 3))

    with pytest.raises(RuntimeError, match="expected"):
        Broken().encode(["x"])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, np.finfo(np.float64).max])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("dim", [None, 2])
def test_custom_encoder_rejects_nonfinite_float32_output(value, normalize, dim) -> None:  # type: ignore[no-untyped-def]
    class NonFinite(Encoder):
        @property
        def native_dim(self) -> int:
            return 4

        def _encode(self, texts, batch_size):  # type: ignore[no-untyped-def]
            # A bad trailing component must be rejected even when truncation
            # would remove it; float64 overflow must also fail at the boundary.
            return np.tile(np.array([1, 2, 3, value], dtype=np.float64), (len(texts), 1))

    with pytest.raises(RuntimeError, match="non-finite float32 embeddings"):
        NonFinite().encode(["x"], normalize=normalize, dim=dim)


def test_load_encoder_specs(hashing: HashingEncoder) -> None:
    assert load_encoder(hashing) is hashing
    assert isinstance(load_encoder("hashing", dim=64), HashingEncoder)
    enc = load_encoder({"type": "hashing", "dim": 32})
    assert enc.dim == 32
    with pytest.raises(TypeError):
        load_encoder("")
    oa = load_encoder("openai:text-embedding-3-small", base_url="http://localhost:9/v1", native_dim=8)
    assert oa.name == "openai:text-embedding-3-small"


def test_presets_resolution(tmp_path) -> None:
    q8 = resolve_preset("Qwen/Qwen3-Embedding-8B")
    assert q8.pooling == "last_token" and q8.padding_side == "left"
    assert q8.mrl_range == (32, 4096) and q8.eos_token == "<|endoftext|>" and q8.ensure_eos
    assert "{instruction}" in q8.query_template and q8.document_template == "{text}"
    assert resolve_preset("qwen/qwen3-embedding-0.6b").mrl_range == (32, 1024)
    assert resolve_preset("intfloat/e5-base-v2").document_template == "passage: {text}"
    assert resolve_preset("BAAI/bge-small-en-v1.5").pooling == "cls"
    assert resolve_preset("some/unknown-model") == DEFAULT_PRESET
    # saved checkpoint config wins
    saved = ModelPreset(pooling="cls", max_length=77, mrl_range=(8, 16))
    (tmp_path / CONFIG_FILENAME).write_text(__import__("json").dumps(saved.to_dict()))
    assert resolve_preset(str(tmp_path)) == saved
