"""Encoders turn text into vectors. Heavy backends are imported lazily.

Use :func:`load_encoder` for config-driven construction::

    load_encoder("hashing")                                  # zero-dependency baseline
    load_encoder("Qwen/Qwen3-Embedding-8B", dtype="bfloat16")  # transformers backend
    load_encoder({"type": "openai", "model": "Qwen/Qwen3-Embedding-8B",
                  "base_url": "http://localhost:8000/v1"})    # remote vLLM/TEI server
"""

from __future__ import annotations

from typing import Any

from clmkit.encoders.base import Encoder
from clmkit.encoders.hashing import HashingEncoder
from clmkit.registry import ENCODERS

__all__ = ["Encoder", "HashingEncoder", "load_encoder"]


def load_encoder(spec: str | dict[str, Any] | Encoder, **kwargs: Any) -> Encoder:
    """Build an encoder from a spec.

    * an :class:`Encoder` instance is returned as-is;
    * a dict needs ``type`` (a registry name) plus constructor kwargs; for ``hf`` the
      model id goes in ``model`` / ``model_name_or_path``;
    * a string that names a registered encoder type (``"hashing"``) builds it with ``kwargs``;
    * ``"openai:<model>"`` builds the OpenAI-compatible client;
    * any other string is treated as a Hugging Face model id or local path.
    """
    if isinstance(spec, Encoder):
        return spec
    if isinstance(spec, dict):
        cfg = {**spec, **kwargs}
        kind = cfg.pop("type", "hf")
        if kind in ("hf", "transformers", "huggingface"):
            model = cfg.pop("model", None) or cfg.pop("model_name_or_path", None)
            return ENCODERS.build(kind, model, **cfg)
        return ENCODERS.build(kind, **cfg)
    if not isinstance(spec, str) or not spec:
        raise TypeError(f"encoder spec must be a non-empty str, dict or Encoder, got {spec!r}")
    if spec.startswith("openai:"):
        return ENCODERS.build("openai", spec.split(":", 1)[1], **kwargs)
    if spec in ENCODERS and spec not in ("hf", "transformers", "huggingface", "openai"):
        return ENCODERS.build(spec, **kwargs)
    return ENCODERS.build("hf", spec, **kwargs)
