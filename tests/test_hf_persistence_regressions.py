"""Checkpoint lifecycle tests using local tiny weights and mocked Hub revisions."""

import json

import numpy as np
import pytest

pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from clmkit.encoders.hf import HFEncoder  # noqa: E402

pytestmark = pytest.mark.hf


def _adapter(path):  # type: ignore[no-untyped-def]
    encoder = HFEncoder(str(path), device="cpu")
    encoder.model = peft.get_peft_model(encoder.model, peft.LoraConfig(r=2, target_modules=["q_proj"]))
    return encoder


def test_adapter_to_merged_save_rejects_before_mutating_model_or_checkpoint(tmp_path, tiny_model_dir):  # type: ignore[no-untyped-def]
    encoder = _adapter(tiny_model_dir)
    expected = encoder.encode("cats eat fish")
    path = encoder.save_pretrained(tmp_path / "adapter")
    original_model = encoder.model
    original_files = {file.name: file.read_bytes() for file in path.iterdir() if file.is_file()}
    with pytest.raises(ValueError, match="separate directories"):
        encoder.save_pretrained(path, merge_adapter=True)
    assert encoder.model is original_model
    assert original_files == {file.name: file.read_bytes() for file in path.iterdir() if file.is_file()}
    encoder.save_pretrained(path)  # Repeating the same format remains supported.
    np.testing.assert_allclose(HFEncoder(str(path), device="cpu").encode("cats eat fish"), expected, atol=1e-6)
    merged_path = encoder.save_pretrained(tmp_path / "merged", merge_adapter=True)
    encoder.save_pretrained(merged_path)
    np.testing.assert_allclose(HFEncoder(str(merged_path), device="cpu").encode("cats eat fish"), expected, atol=1e-6)


def test_full_to_adapter_save_rejects_before_overwriting_checkpoint(tmp_path, tiny_model_dir):  # type: ignore[no-untyped-def]
    path = HFEncoder(str(tiny_model_dir), device="cpu").save_pretrained(tmp_path / "full")
    original_files = {file.name: file.read_bytes() for file in path.iterdir() if file.is_file()}
    with pytest.raises(ValueError, match="separate directories"):
        _adapter(tiny_model_dir).save_pretrained(path)
    assert original_files == {file.name: file.read_bytes() for file in path.iterdir() if file.is_file()}


@pytest.mark.parametrize("override", [None, "explicit-base-revision"])
def test_adapter_roundtrip_preserves_resolved_base_revision(tmp_path, tiny_model_dir, monkeypatch, override):  # type: ignore[no-untyped-def]
    encoder = _adapter(tiny_model_dir)
    encoder.revision = "moving-base-tag"
    encoder.resolved_revision = "a" * 40
    encoder.adapter_revision = "unrelated-adapter-revision"
    path = encoder.save_pretrained(tmp_path / "adapter")
    config = json.loads((path / "adapter_config.json").read_text())
    assert config["revision"] == "a" * 40
    original = transformers.AutoModel.from_pretrained
    calls = []

    def load(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs.get("revision"))
        return original(*args, **kwargs)

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", load)
    loaded = HFEncoder(str(path), device="cpu", revision=override)
    assert calls == [override or "a" * 40]
    np.testing.assert_allclose(loaded.encode("cats"), encoder.encode("cats"), atol=1e-6)


def test_legacy_adapter_without_revision_still_loads(tmp_path, tiny_model_dir):  # type: ignore[no-untyped-def]
    encoder = _adapter(tiny_model_dir)
    path = encoder.save_pretrained(tmp_path / "legacy")
    config = json.loads((path / "adapter_config.json").read_text())
    config.pop("revision", None)
    (path / "adapter_config.json").write_text(json.dumps(config))
    np.testing.assert_allclose(HFEncoder(str(path), device="cpu").encode("cats"), encoder.encode("cats"), atol=1e-6)
