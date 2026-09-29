"""Second-stage rerankers.

A contrastive bi-encoder retrieves candidates cheaply; a reranker then scores
each ``(query, document)`` pair jointly for higher precision.

* :class:`EncoderReranker` - rescore with another (usually larger) bi-encoder. No extra deps.
* :class:`LLMYesNoReranker` - decoder LLM judging relevance via P("yes"), e.g. **Qwen3-Reranker**.
* :class:`CrossEncoderReranker` - classic ``AutoModelForSequenceClassification`` cross-encoder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from clmkit.encoders.base import Encoder
from clmkit.types import SearchHit
from clmkit.utils import require, resolve_device


class Reranker(ABC):
    name: str = "reranker"

    @abstractmethod
    def score(
        self, query: str, documents: Sequence[str], *, instruction: str | None = None, batch_size: int = 8
    ) -> np.ndarray:
        """Relevance score per document (higher = more relevant)."""

    def rerank(
        self,
        query: str,
        hits: Sequence[SearchHit],
        k: int | None = None,
        *,
        instruction: str | None = None,
        batch_size: int = 8,
    ) -> list[SearchHit]:
        """Re-order hits by reranker score. Hits must carry ``text``.

        The first-stage score is preserved in ``metadata["retrieval_score"]``.
        """
        if not hits:
            return []
        missing = [h.id for h in hits if h.text is None]
        if missing:
            raise ValueError(f"cannot rerank hits without text: {missing[:5]}")
        scores = self.score(query, [h.text or "" for h in hits], instruction=instruction, batch_size=batch_size)
        rescored = [
            replace(h, score=float(s), metadata={**h.metadata, "retrieval_score": h.score})
            for h, s in zip(hits, scores, strict=True)
        ]
        rescored.sort(key=lambda h: h.score, reverse=True)
        return rescored[:k] if k else rescored


class EncoderReranker(Reranker):
    """Cosine similarity under a (typically stronger) second encoder."""

    def __init__(self, encoder: Encoder) -> None:
        self.encoder = encoder
        self.name = f"encoder:{encoder.name}"

    def score(
        self, query: str, documents: Sequence[str], *, instruction: str | None = None, batch_size: int = 8
    ) -> np.ndarray:
        q = self.encoder.encode(query, kind="query", instruction=instruction)
        d = self.encoder.encode(list(documents), kind="document", batch_size=batch_size)
        return d @ q


QWEN3_RERANK_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
QWEN3_RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
QWEN3_RERANK_TEMPLATE = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
DEFAULT_RERANK_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


class LLMYesNoReranker(Reranker):
    """Score = P("yes") from a causal LM asked whether the document answers the query.

    Defaults reproduce the Qwen3-Reranker (0.6B/4B/8B) reference implementation.
    """

    def __init__(
        self,
        model_name_or_path: str | None = None,
        *,
        model: Any = None,
        tokenizer: Any = None,
        prefix: str = QWEN3_RERANK_PREFIX,
        suffix: str = QWEN3_RERANK_SUFFIX,
        template: str = QWEN3_RERANK_TEMPLATE,
        default_instruction: str = DEFAULT_RERANK_INSTRUCTION,
        yes_token: str = "yes",  # noqa: S107 - vocabulary token
        no_token: str = "no",  # noqa: S107 - vocabulary token
        max_length: int = 8192,
        device: str | None = "auto",
        dtype: str | None = "auto",
        revision: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        from clmkit.encoders.hf import _dtype_kwarg, resolve_torch_dtype

        self._torch = require("torch")
        transformers = require("transformers")
        self.device = resolve_device(device)
        if tokenizer is None:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name_or_path, revision=revision, trust_remote_code=trust_remote_code
            )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        if model is None:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                revision=revision,
                trust_remote_code=trust_remote_code,
                **_dtype_kwarg(resolve_torch_dtype(dtype, self.device)),
            )
        self.tokenizer = tokenizer
        self.model = model.to(self.device).eval()
        self.prefix, self.suffix, self.template = prefix, suffix, template
        self.default_instruction = default_instruction
        self.max_length = max_length
        self.yes_id = tokenizer.convert_tokens_to_ids(yes_token)
        self.no_id = tokenizer.convert_tokens_to_ids(no_token)
        if tokenizer.unk_token_id is not None and tokenizer.unk_token_id in (self.yes_id, self.no_id):
            raise ValueError("yes/no tokens are not single tokens in this vocabulary")
        self._prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        self._suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        self.name = model_name_or_path or type(model).__name__

    def _build(self, pairs: list[str]) -> dict[str, Any]:
        body_max = self.max_length - len(self._prefix_ids) - len(self._suffix_ids)
        if body_max < 1:
            raise ValueError("max_length is too small for the reranker prompt")
        enc = self.tokenizer(pairs, padding=False, truncation=True, max_length=body_max, add_special_tokens=False)
        ids = [self._prefix_ids + x + self._suffix_ids for x in enc["input_ids"]]
        batch = self.tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")
        return {k: v.to(self.device) for k, v in batch.items() if k in ("input_ids", "attention_mask")}

    def score(
        self, query: str, documents: Sequence[str], *, instruction: str | None = None, batch_size: int = 8
    ) -> np.ndarray:
        torch = self._torch
        inst = instruction or self.default_instruction
        pairs = [self.template.format(instruction=inst, query=query, document=d) for d in documents]
        out: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(pairs), batch_size):
                batch = self._build(pairs[start : start + batch_size])
                logits = self.model(**batch).logits[:, -1, :]  # left padding => last position is real
                pair = torch.stack([logits[:, self.no_id], logits[:, self.yes_id]], dim=1).float()
                out.extend(torch.log_softmax(pair, dim=1)[:, 1].exp().cpu().tolist())
        return np.asarray(out, dtype=np.float32)


def load_reranker(spec: str | Reranker, **kwargs: Any) -> Reranker:
    """``"encoder:<encoder spec>"`` -> :class:`EncoderReranker`; ids containing
    ``qwen3-reranker`` -> :class:`LLMYesNoReranker`; anything else -> :class:`CrossEncoderReranker`."""
    if isinstance(spec, Reranker):
        return spec
    if spec.startswith("encoder:"):
        from clmkit.encoders import load_encoder

        return EncoderReranker(load_encoder(spec.split(":", 1)[1], **kwargs))
    if "qwen3-reranker" in spec.lower() or kwargs.pop("yes_no", False):
        return LLMYesNoReranker(spec, **kwargs)
    return CrossEncoderReranker(spec, **kwargs)


class CrossEncoderReranker(Reranker):
    """``AutoModelForSequenceClassification`` cross-encoder (e.g. ``BAAI/bge-reranker-v2-m3``)."""

    def __init__(
        self,
        model_name_or_path: str | None = None,
        *,
        model: Any = None,
        tokenizer: Any = None,
        max_length: int = 512,
        device: str | None = "auto",
        revision: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        self._torch = require("torch")
        transformers = require("transformers")
        self.device = resolve_device(device)
        self.tokenizer = tokenizer or transformers.AutoTokenizer.from_pretrained(
            model_name_or_path, revision=revision, trust_remote_code=trust_remote_code
        )
        model = model or transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, revision=revision, trust_remote_code=trust_remote_code
        )
        self.model = model.to(self.device).eval()
        self.max_length = max_length
        self.name = model_name_or_path or type(model).__name__

    def score(
        self, query: str, documents: Sequence[str], *, instruction: str | None = None, batch_size: int = 8
    ) -> np.ndarray:
        torch = self._torch
        out: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(documents), batch_size):
                docs = list(documents[start : start + batch_size])
                batch = self.tokenizer(
                    [query] * len(docs),
                    docs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**batch).logits.float()
                probs = torch.sigmoid(logits[:, 0]) if logits.shape[1] == 1 else logits.softmax(-1)[:, -1]
                out.extend(probs.cpu().tolist())
        return np.asarray(out, dtype=np.float32)
