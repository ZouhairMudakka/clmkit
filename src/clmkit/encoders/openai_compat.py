"""Client for any OpenAI-compatible ``/v1/embeddings`` endpoint.

Works with OpenAI, vLLM (``vllm serve Qwen/Qwen3-Embedding-8B --task embed``),
Hugging Face TEI, Infinity, Ollama, LiteLLM, and clmkit's own ``clmkit serve``.
Uses only the standard library, so it adds no dependencies.

Note: servers embed exactly the text they receive. For instruction-tuned models
such as Qwen3-Embedding, the query prompt is applied **client-side** by the
template settings below (auto-selected from the model name).
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse

import numpy as np

from clmkit.encoders.base import Encoder
from clmkit.encoders.presets import resolve_preset

logger = logging.getLogger(__name__)

_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class RemoteEncoderError(RuntimeError):
    """The embedding endpoint returned an error or malformed payload."""


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """urllib forwards the Authorization header on redirects, even to other hosts, which
    would leak the API key. Embedding APIs never need redirects, so we refuse them."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, f"refusing redirect to {newurl}", headers, fp)


_OPENER = urllib.request.build_opener(_RefuseRedirects)


class OpenAICompatibleEncoder(Encoder):
    """Embed through an HTTP embeddings API.

    Args:
        model: model name sent in the request (e.g. ``"Qwen/Qwen3-Embedding-8B"``).
        base_url: API root, e.g. ``"http://localhost:8000/v1"``.
        api_key: explicit key. Prefer ``api_key_env`` so keys never live in configs.
        api_key_env: environment variable holding the key (default ``OPENAI_API_KEY``).
        dimensions: request server-side Matryoshka truncation (if the server supports it).
        native_dim: known output size; otherwise discovered on the first request.
        max_batch: texts per HTTP request.
        max_retries: retries on 408/409/429/5xx and network errors, with exponential backoff.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        dimensions: int | None = None,
        native_dim: int | None = None,
        timeout: float = 60.0,
        max_batch: int = 64,
        max_retries: int = 3,
        backoff: float = 1.0,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        preset = resolve_preset(model)
        kwargs.setdefault("query_template", preset.query_template)
        kwargs.setdefault("document_template", preset.document_template)
        kwargs.setdefault("default_instruction", preset.default_instruction)
        super().__init__(**kwargs)
        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"base_url must be an http(s) URL, got {base_url!r}")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key if api_key is not None else os.environ.get(api_key_env)
        if self._api_key and parsed.scheme == "http" and parsed.hostname not in _LOCAL_HOSTS:
            logger.warning("sending an API key over plain HTTP to %s", parsed.hostname)
        self.dimensions = dimensions
        self._native_dim = dimensions or native_dim
        self.timeout = timeout
        self.max_batch = max_batch
        self.max_retries = max_retries
        self.backoff = backoff
        self.extra_headers = dict(headers or {})
        self.name = f"openai:{model}"

    def __repr__(self) -> str:  # never leak the key
        return f"OpenAICompatibleEncoder(model={self.model!r}, base_url={self.base_url!r})"

    def fingerprint_config(self) -> dict[str, Any]:
        # Credentials/headers are deliberately excluded. A mutable deployment
        # still needs a caller-provided encoder_identity when saving a retriever.
        return {**super().fingerprint_config(), "base_url": self.base_url, "dimensions": self.dimensions}

    @property
    def native_dim(self) -> int:
        if self._native_dim is None:
            self._native_dim = int(self._request(["dimension probe"]).shape[1])
        return self._native_dim

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        data = json.dumps(payload).encode("utf-8")
        attempt = 0
        while True:
            req = urllib.request.Request(  # noqa: S310 - scheme validated in __init__
                f"{self.base_url}/embeddings", data=data, headers=headers, method="POST"
            )
            try:
                # B310: base_url scheme is restricted to http(s) in __init__, and redirects are refused.
                with _OPENER.open(req, timeout=self.timeout) as resp:  # nosec B310
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if 300 <= exc.code < 400:
                    raise RemoteEncoderError(
                        f"embeddings endpoint answered with a redirect ({exc.code}); use the final URL as base_url"
                    ) from exc
                body = exc.read().decode("utf-8", "replace")[:500] if exc.fp else ""
                if exc.code in _RETRY_STATUS and attempt < self.max_retries:
                    attempt += 1
                    time.sleep(self.backoff * 2 ** (attempt - 1))
                    continue
                raise RemoteEncoderError(f"embeddings request failed ({exc.code}): {body}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt < self.max_retries:
                    attempt += 1
                    time.sleep(self.backoff * 2 ** (attempt - 1))
                    continue
                raise RemoteEncoderError(f"embeddings endpoint unreachable: {exc}") from exc
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RemoteEncoderError("embeddings endpoint returned invalid JSON") from exc

    def _request(self, texts: list[str]) -> np.ndarray:
        payload: dict[str, Any] = {"model": self.model, "input": texts, "encoding_format": "float"}
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        body = self._post(payload)
        try:
            items = body["data"]
            if not isinstance(items, list) or len(items) != len(texts):
                raise ValueError("data must contain one embedding per input")
            indices = [item["index"] for item in items]
            if any(type(index) is not int for index in indices) or set(indices) != set(range(len(texts))):
                raise ValueError("indices must be a permutation of the input positions")
            items = sorted(items, key=lambda d: d["index"])
            emb = np.asarray([d["embedding"] for d in items], dtype=np.float32)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RemoteEncoderError("malformed embeddings response") from exc
        if emb.ndim != 2 or emb.shape[0] != len(texts) or emb.shape[1] == 0:
            raise RemoteEncoderError(f"expected {len(texts)} embeddings, got shape {emb.shape}")
        if self._native_dim is not None and emb.shape[1] != self._native_dim:
            raise RemoteEncoderError(f"expected embedding dimension {self._native_dim}, got {emb.shape[1]}")
        if not np.isfinite(emb).all():
            raise RemoteEncoderError("embeddings response contains non-finite values")
        # Discover the dimension on the first batch, so later batches are checked
        # before concatenation (including a server that changes dimensions).
        if self._native_dim is None:
            self._native_dim = int(emb.shape[1])
        return emb

    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        size = max(1, min(batch_size, self.max_batch))
        parts = [self._request(texts[i : i + size]) for i in range(0, len(texts), size)]
        out = np.concatenate(parts, axis=0)
        if self._native_dim is None:
            self._native_dim = out.shape[1]
        return out
