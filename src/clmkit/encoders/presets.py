"""Known-good settings for popular contrastive embedding models.

Getting pooling / padding / prompt format wrong silently degrades retrieval
quality, so the defaults for well-known checkpoints live here. Anything not
matched falls back to mean pooling, which is a reasonable default for
BERT-style sentence encoders. Every field can be overridden explicitly.
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal

Pooling = Literal["last_token", "mean", "cls"]

QWEN3_QUERY_TEMPLATE = "Instruct: {instruction}\nQuery:{text}"
QWEN3_DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

#: File written next to fine-tuned checkpoints so they reload with the right settings.
CONFIG_FILENAME = "clmkit_config.json"


@dataclass(frozen=True)
class ModelPreset:
    pooling: Pooling = "mean"
    padding_side: Literal["left", "right"] = "right"
    query_template: str = "{text}"
    document_template: str = "{text}"
    default_instruction: str | None = None
    #: Make sure every sequence ends with ``eos_token`` (appending it only when the
    #: tokenizer did not). Required for last-token pooling on decoder-only embedders.
    ensure_eos: bool = False
    #: Token to pool from. ``None`` = the tokenizer's ``eos_token``. NOTE: for Qwen3 the
    #: tokenizer's eos_token is ``<|im_end|>`` but embeddings are pooled from ``<|endoftext|>``.
    eos_token: str | None = None
    max_length: int = 512
    #: Inclusive (min, max) Matryoshka dims the model was trained for, if any.
    mrl_range: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mrl_range"] = list(self.mrl_range) if self.mrl_range else None
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelPreset:
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        if kwargs.get("mrl_range") is not None:
            kwargs["mrl_range"] = tuple(kwargs["mrl_range"])
        return cls(**kwargs)


def _qwen3(max_dim: int) -> ModelPreset:
    return ModelPreset(
        pooling="last_token",
        padding_side="left",
        query_template=QWEN3_QUERY_TEMPLATE,
        default_instruction=QWEN3_DEFAULT_INSTRUCTION,
        ensure_eos=True,
        eos_token="<|endoftext|>",  # noqa: S106 - a vocabulary token, not a secret  # nosec B106
        max_length=8192,
        mrl_range=(32, max_dim),
    )


# Patterns are matched case-insensitively against the model id / path (fnmatch syntax).
PRESETS: dict[str, ModelPreset] = {
    "*qwen3-embedding-0.6b*": _qwen3(1024),
    "*qwen3-embedding-4b*": _qwen3(2560),
    "*qwen3-embedding-8b*": _qwen3(4096),
    "*e5-mistral-7b-instruct*": ModelPreset(
        pooling="last_token",
        padding_side="left",
        ensure_eos=True,
        max_length=4096,
        query_template="Instruct: {instruction}\nQuery: {text}",
        default_instruction=QWEN3_DEFAULT_INSTRUCTION,
    ),
    "*multilingual-e5-*instruct*": ModelPreset(
        pooling="mean",
        query_template="Instruct: {instruction}\nQuery: {text}",
        default_instruction=QWEN3_DEFAULT_INSTRUCTION,
    ),
    "*e5-*": ModelPreset(pooling="mean", query_template="query: {text}", document_template="passage: {text}"),
    "*bge-*-en-v1.5*": ModelPreset(
        pooling="cls",
        query_template="Represent this sentence for searching relevant passages: {text}",
    ),
    "*bge-m3*": ModelPreset(pooling="cls", max_length=8192),
    "*gte-*": ModelPreset(pooling="mean"),
    "*all-minilm-*": ModelPreset(pooling="mean", max_length=256),
    "*all-mpnet-*": ModelPreset(pooling="mean", max_length=384),
}

DEFAULT_PRESET = ModelPreset()


def resolve_preset(model_name_or_path: str) -> ModelPreset:
    """Find settings for a model: saved ``clmkit_config.json`` > known pattern > default."""
    local = Path(model_name_or_path) / CONFIG_FILENAME
    if local.is_file():
        return ModelPreset.from_dict(json.loads(local.read_text(encoding="utf-8")))
    name = model_name_or_path.replace("\\", "/").lower()
    for pattern, preset in PRESETS.items():
        if fnmatch.fnmatch(name, pattern):
            return preset
    return DEFAULT_PRESET
