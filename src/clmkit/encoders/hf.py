"""Hugging Face ``transformers`` encoder: Qwen3-Embedding, E5, BGE, GTE, MiniLM, ... or your fine-tune.

Requires ``pip install "clmkit[hf]"``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from clmkit.encoders.base import Encoder, InstructionArg
from clmkit.encoders.presets import CONFIG_FILENAME, DEFAULT_PRESET, ModelPreset, Pooling, resolve_preset
from clmkit.pooling import get_pooler
from clmkit.types import EncodeKind
from clmkit.utils import require, resolve_device, version_tuple

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class _Unset:
    """Distinguish an omitted output dimension from an explicit native-dimension override."""


_UNSET = _Unset()


def _safe_model_kwargs(path: str | None, revision: str | None, kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Enforce safetensors and reject implicit PEFT loading, which can fall back to pickle."""
    options = dict(kwargs or {})
    if "use_safetensors" in options and options["use_safetensors"] is not True:
        raise ValueError("clmkit requires use_safetensors=True")
    for key in ("from_tf", "from_flax", "adapter_kwargs", "_adapter_model_path", "_commit_hash"):
        if key in options:
            raise ValueError(f"model_kwargs[{key!r}] is unsupported by the safetensors-only loader")
    options["use_safetensors"] = True
    if path:
        transformers = require("transformers")
        hub_options = {k: options[k] for k in ("cache_dir", "token", "local_files_only", "subfolder") if k in options}
        # PEFT does not enforce a use_safetensors argument. Never let Transformers
        # transparently enter its adapter loader; HFEncoder handles adapters itself.
        local = Path(path)
        if local.is_dir():
            adapter_config = local / str(options.get("subfolder", "")) / "adapter_config.json"
            is_adapter = adapter_config.is_file() or (local / "adapter_config.json").is_file()
        else:
            # Pin the check and the eventual loader to one snapshot, so a moving
            # Hub branch cannot introduce an unchecked implicit adapter in between.
            config_path = transformers.utils.hub.cached_file(path, "config.json", revision=revision, **hub_options)
            commit = transformers.utils.hub.extract_commit_hash(config_path, None)
            if commit is None:
                raise ValueError("could not resolve an immutable model snapshot for safetensors loading")
            options["_commit_hash"] = commit
            revision = commit
            is_adapter = transformers.utils.find_adapter_config_file(path, revision=revision, **hub_options) is not None
        if is_adapter:
            raise ValueError(
                "implicit adapter loading is unsupported; use HFEncoder(base, adapter=...) with safetensors"
            )
        # Keep the dependency's own adapter lookup on the same revision and options.
        options["adapter_kwargs"] = {"revision": revision, **hub_options}
    return options


def _safe_adapter_path(adapter: str, revision: str | None = None) -> str:
    """Resolve only safe adapter weights; PEFT itself otherwise permits .bin fallback."""
    local = Path(adapter)
    if local.is_dir():
        if not (local / "adapter_model.safetensors").is_file():
            raise ValueError("adapter requires adapter_model.safetensors; legacy .bin weights are unsupported")
        return str(local)
    hub = require("huggingface_hub")
    weights = Path(hub.hf_hub_download(adapter, "adapter_model.safetensors", revision=revision))
    # The cache snapshot directory is an immutable commit, unlike a mutable branch.
    commit = weights.parent.name
    config = Path(hub.hf_hub_download(adapter, "adapter_config.json", revision=commit))
    if config.parent != weights.parent:
        raise ValueError("adapter config and safetensors must belong to the same snapshot")
    return str(weights.parent)


_DTYPES = {
    "float32": "float32",
    "fp32": "float32",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float16": "float16",
    "fp16": "float16",
}


def resolve_torch_dtype(dtype: str | None, device: str) -> torch.dtype:
    """``"auto"`` -> bf16 on CUDA with bf16 support, else fp32 (fp16/bf16 matmuls are slow on CPU)."""
    torch = require("torch")
    if dtype in (None, "auto"):
        if device.startswith("cuda") and torch.cuda.is_bf16_supported():  # pragma: no cover
            return torch.bfloat16
        return torch.float32
    try:
        return getattr(torch, _DTYPES[str(dtype).lower()])
    except KeyError:
        raise ValueError(f"unsupported dtype {dtype!r}; choose from {sorted(_DTYPES)} or 'auto'") from None


def _dtype_kwarg(torch_dtype: torch.dtype) -> dict[str, Any]:
    # transformers>=4.56 renamed `torch_dtype` to `dtype` (v5 removed the old name).
    transformers = require("transformers")
    key = "dtype" if version_tuple(transformers.__version__) >= (4, 56) else "torch_dtype"
    return {key: torch_dtype}


class HFEncoder(Encoder):
    """A contrastive text encoder backed by ``transformers.AutoModel``.

    Settings (pooling, padding side, prompt templates, EOS handling, max length)
    default to the model's preset (see :mod:`clmkit.encoders.presets`) or to the
    ``clmkit_config.json`` saved next to a fine-tuned checkpoint.

    Args:
        model_name_or_path: HF hub id or local directory.
        model / tokenizer: pass pre-built objects instead of loading (useful for tests).
        pooling: ``"last_token"`` | ``"mean"`` | ``"cls"``.
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, ``"cuda:1"``, ``"mps"``.
        dtype: ``"auto"``, ``"float32"``, ``"bfloat16"``, ``"float16"``.
        revision: pin a hub commit/tag (recommended for reproducibility & supply-chain safety).
        trust_remote_code: off by default; only enable for repositories you trust.
        adapter: path/id of a PEFT (LoRA) adapter to load on top of the base model.
    """

    def __init__(
        self,
        model_name_or_path: str | None = None,
        *,
        model: Any = None,
        tokenizer: Any = None,
        pooling: Pooling | None = None,
        padding_side: str | None = None,
        query_template: str | None = None,
        document_template: str | None = None,
        default_instruction: str | None = None,
        max_length: int | None = None,
        ensure_eos: bool | None = None,
        eos_token: str | None = None,
        output_dim: int | _Unset | None = _UNSET,
        device: str | None = "auto",
        dtype: str | None = "auto",
        revision: str | None = None,
        trust_remote_code: bool = False,
        adapter: str | None = None,
        adapter_revision: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if model_name_or_path is None and (model is None or tokenizer is None):
            raise ValueError("pass model_name_or_path, or both model= and tokenizer=")
        preset: ModelPreset = resolve_preset(model_name_or_path) if model_name_or_path else DEFAULT_PRESET
        resolved_output_dim = preset.output_dim if isinstance(output_dim, _Unset) else output_dim
        super().__init__(
            query_template=query_template or preset.query_template,
            document_template=document_template or preset.document_template,
            default_instruction=default_instruction if default_instruction is not None else preset.default_instruction,
            output_dim=resolved_output_dim,
        )
        torch = require("torch")
        transformers = require("transformers")

        self.pooling: Pooling = pooling or preset.pooling
        self._pool = get_pooler(self.pooling)
        self.padding_side = padding_side or preset.padding_side
        self.max_length = int(max_length or preset.max_length)
        self.ensure_eos = preset.ensure_eos if ensure_eos is None else ensure_eos
        self.mrl_range = preset.mrl_range
        self.device = resolve_device(device)

        if tokenizer is None:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name_or_path,
                revision=revision,
                trust_remote_code=trust_remote_code,
                **(tokenizer_kwargs or {}),
            )
        tokenizer.padding_side = self.padding_side
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        self.tokenizer = tokenizer
        self.eos_token = eos_token or preset.eos_token or tokenizer.eos_token
        self._eos_id = tokenizer.convert_tokens_to_ids(self.eos_token) if self.eos_token else None
        if self.ensure_eos and self._eos_id in (None, tokenizer.unk_token_id):
            raise ValueError(f"ensure_eos=True but token {self.eos_token!r} is not in the vocabulary")

        weights_path = model_name_or_path
        if model is None and model_name_or_path and adapter is None:
            # An adapter-only checkpoint (LoRA saved without merging): load base, then adapter.
            local = Path(model_name_or_path)
            if (local / "adapter_config.json").is_file() and not (local / "config.json").is_file():
                adapter_cfg = json.loads((local / "adapter_config.json").read_text(encoding="utf-8"))
                weights_path = adapter_cfg["base_model_name_or_path"]
                adapter = model_name_or_path
        if model is None:
            model = transformers.AutoModel.from_pretrained(
                weights_path,
                revision=revision,
                trust_remote_code=trust_remote_code,
                **_dtype_kwarg(resolve_torch_dtype(dtype, self.device)),
                **_safe_model_kwargs(weights_path, revision, model_kwargs),
            )
        if adapter:
            peft = require("peft")
            adapter_path = _safe_adapter_path(adapter, adapter_revision)
            model = peft.PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
        self.model = model.to(self.device)
        self.model.eval()
        self.name = model_name_or_path or type(model).__name__
        self.revision = revision
        self.resolved_revision = getattr(model.config, "_commit_hash", None)
        self.adapter = adapter
        self.adapter_revision = adapter_revision
        self.resolved_adapter_revision = Path(adapter_path).name if adapter and not Path(adapter).is_dir() else None
        self._torch = torch

        if self.output_dim and self.mrl_range and not (self.mrl_range[0] <= self.output_dim <= self.mrl_range[1]):
            warnings.warn(
                f"output_dim={self.output_dim} is outside the model's Matryoshka range {self.mrl_range}", stacklevel=2
            )

    # ------------------------------------------------------------ plumbing --
    @property
    def native_dim(self) -> int:
        return int(self.model.config.hidden_size)

    def tokenize(self, texts: Sequence[str]) -> dict[str, torch.Tensor]:
        """Tokenise formatted texts, guaranteeing the pooling EOS token when required."""
        tok = self.tokenizer
        if self.ensure_eos and self._eos_id is not None:
            enc = tok(list(texts), padding=False, truncation=True, max_length=self.max_length - 1)
            ids = [x if (x and x[-1] == self._eos_id) else [*x, self._eos_id] for x in enc["input_ids"]]
            batch = tok.pad({"input_ids": ids}, padding=True, return_tensors="pt")
        else:
            batch = tok(list(texts), padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        return {k: v.to(self.device) for k, v in batch.items() if k in ("input_ids", "attention_mask")}

    def embed_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward + pool (keeps the autograd graph; used by the trainer)."""
        out = self.model(**batch)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return self._pool(hidden, batch["attention_mask"])

    def forward(
        self, texts: Sequence[str], kind: EncodeKind = "document", instruction: InstructionArg = None
    ) -> torch.Tensor:
        """Differentiable encoding of raw texts -> ``(n, native_dim)`` un-normalised tensor."""
        return self.embed_batch(self.tokenize(self.format_texts(list(texts), kind, instruction)))

    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        torch = self._torch
        # Sort by length so each batch pads minimally; restore the order afterwards.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
        out = np.empty((len(texts), self.native_dim), dtype=np.float32)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                for start in range(0, len(order), batch_size):
                    idx = order[start : start + batch_size]
                    emb = self.embed_batch(self.tokenize([texts[i] for i in idx]))
                    out[idx] = emb.float().cpu().numpy()
        finally:
            self.model.train(was_training)
        return out

    # --------------------------------------------------------- persistence --
    def fingerprint_config(self) -> dict[str, Any]:
        tokenizer_config: dict[str, Any] = {
            "class": type(self.tokenizer).__name__,
            "padding_side": self.tokenizer.padding_side,
            "truncation_side": self.tokenizer.truncation_side,
            "special_tokens": self.tokenizer.special_tokens_map,
        }
        backend = getattr(self.tokenizer, "backend_tokenizer", None)
        if backend is not None:
            backend_config = json.loads(backend.to_str())
            # These fields are mutated on each tokenization call; the effective
            # settings are already represented by max_length and padding_side.
            backend_config.pop("padding", None)
            backend_config.pop("truncation", None)
            tokenizer_config["backend"] = backend_config
        else:
            tokenizer_config["vocab"] = self.tokenizer.get_vocab()
        tokenizer_digest = hashlib.sha256(
            json.dumps(tokenizer_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            **super().fingerprint_config(),
            **self.config_dict(),
            "revision": self.revision,
            "resolved_revision": self.resolved_revision,
            "adapter": self.adapter,
            "adapter_revision": self.adapter_revision,
            "resolved_adapter_revision": self.resolved_adapter_revision,
            "tokenizer": tokenizer_digest,
            "dtype": str(getattr(self.model, "dtype", None)),
            "attention_implementation": getattr(self.model.config, "_attn_implementation", None),
        }

    def config_dict(self) -> dict[str, Any]:
        return ModelPreset(
            pooling=self.pooling,
            padding_side=self.padding_side,  # type: ignore[arg-type]
            query_template=self.query_template,
            document_template=self.document_template,
            default_instruction=self.default_instruction,
            ensure_eos=self.ensure_eos,
            eos_token=self.eos_token,
            max_length=self.max_length,
            mrl_range=self.mrl_range,
            output_dim=self.output_dim,
        ).to_dict()

    def save_pretrained(self, path: str | Path, *, merge_adapter: bool = False) -> Path:
        """Save model (or LoRA adapter), tokenizer and ``clmkit_config.json``.

        With ``merge_adapter=True`` a PEFT model is merged into its base weights so the
        result loads as a plain checkpoint (no ``peft`` needed at inference time). The
        encoder then holds the merged model, so it stays usable for inference.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if merge_adapter and hasattr(self.model, "merge_and_unload"):
            self.model = self.model.merge_and_unload()
        self.model.save_pretrained(str(path), safe_serialization=True)
        self.tokenizer.save_pretrained(str(path))
        (path / CONFIG_FILENAME).write_text(json.dumps(self.config_dict(), indent=2), encoding="utf-8")
        logger.info("saved encoder to %s", path)
        return path
