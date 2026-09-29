"""Plugin registries.

Every pluggable component family (encoders, indexes, rerankers, losses) has a
:class:`Registry`. Components can be registered three ways:

1. Decorator, for classes defined in an already-imported module::

       @ENCODERS.register("my-encoder")
       class MyEncoder(Encoder): ...

2. Lazily, by dotted path, so heavy backends (torch/transformers) are only
   imported when actually built::

       ENCODERS.register_lazy("hf", "clmkit.encoders.hf:HFEncoder")

3. From third-party packages, via Python entry points in the group
   ``clmkit.<kind>`` (e.g. ``clmkit.encoders``). Nothing to import manually::

       # pyproject.toml of your plugin
       [project.entry-points."clmkit.encoders"]
       my-encoder = "my_pkg.encoders:MyEncoder"
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from importlib.metadata import entry_points
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class RegistryError(KeyError):
    """Unknown or conflicting registry entry."""


class Registry(Generic[T]):
    """A name -> factory mapping with lazy imports and entry-point discovery."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T | str] = {}
        self._aliases: dict[str, str] = {}
        self._entry_points_loaded = False

    # ------------------------------------------------------------ register --
    def register(self, name: str, *, aliases: Iterable[str] = (), override: bool = False) -> Callable[[T], T]:
        """Class/function decorator registering ``obj`` under ``name`` (+ aliases)."""

        def decorator(obj: T) -> T:
            self._add(name, obj, aliases, override)
            return obj

        return decorator

    def register_lazy(self, name: str, target: str, *, aliases: Iterable[str] = (), override: bool = False) -> None:
        """Register a ``"package.module:Attribute"`` path that is imported on first use."""
        if ":" not in target:
            raise ValueError(f"lazy target must look like 'module:attr', got {target!r}")
        self._add(name, target, aliases, override)

    def _add(self, name: str, obj: T | str, aliases: Iterable[str], override: bool) -> None:
        key = self._norm(name)
        if key in self._items and not override:
            raise RegistryError(f"{self.kind} {name!r} is already registered (pass override=True)")
        self._items[key] = obj
        for alias in aliases:
            self._aliases[self._norm(alias)] = key

    # --------------------------------------------------------------- query --
    @staticmethod
    def _norm(name: str) -> str:
        return name.strip().lower().replace("_", "-")

    def _resolve_key(self, name: str) -> str | None:
        key = self._norm(name)
        key = self._aliases.get(key, key)
        if key in self._items:
            return key
        if not self._entry_points_loaded:
            self._load_entry_points()
            key = self._aliases.get(key, key)
            if key in self._items:
                return key
        return None

    def _load_entry_points(self) -> None:
        self._entry_points_loaded = True
        for ep in entry_points(group=f"clmkit.{self.kind}"):
            key = self._norm(ep.name)
            if key not in self._items:  # built-ins win over plugins
                self._items[key] = ep.value

    def get(self, name: str) -> T:
        """Return the registered object, importing it if it was registered lazily."""
        key = self._resolve_key(name)
        if key is None:
            raise RegistryError(f"unknown {self.kind} {name!r}; available: {', '.join(self.names()) or '(none)'}")
        obj = self._items[key]
        if isinstance(obj, str):
            module_name, _, attr = obj.partition(":")
            obj = getattr(importlib.import_module(module_name), attr)
            self._items[key] = obj
        return obj

    def build(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Instantiate/call the registered factory."""
        factory = self.get(name)
        return factory(*args, **kwargs)  # type: ignore[operator]

    def names(self) -> list[str]:
        if not self._entry_points_loaded:
            self._load_entry_points()
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self._resolve_key(name) is not None

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, {self.names()})"


ENCODERS: Registry[Any] = Registry("encoders")
INDEXES: Registry[Any] = Registry("indexes")
RERANKERS: Registry[Any] = Registry("rerankers")
LOSSES: Registry[Any] = Registry("losses")

# Built-ins that need optional dependencies are registered lazily so that
# `import clmkit` never imports torch.
ENCODERS.register_lazy("hashing", "clmkit.encoders.hashing:HashingEncoder")
ENCODERS.register_lazy("hf", "clmkit.encoders.hf:HFEncoder", aliases=("transformers", "huggingface"))
ENCODERS.register_lazy(
    "openai", "clmkit.encoders.openai_compat:OpenAICompatibleEncoder", aliases=("openai-compatible", "vllm", "tei")
)
INDEXES.register_lazy("numpy", "clmkit.index.numpy_index:NumpyIndex", aliases=("exact", "flat"))
INDEXES.register_lazy("faiss", "clmkit.index.faiss_index:FaissIndex")
RERANKERS.register_lazy("encoder", "clmkit.rerank:EncoderReranker", aliases=("bi-encoder",))
RERANKERS.register_lazy("llm-yes-no", "clmkit.rerank:LLMYesNoReranker", aliases=("qwen3-reranker",))
RERANKERS.register_lazy("cross-encoder", "clmkit.rerank:CrossEncoderReranker")
LOSSES.register_lazy("infonce", "clmkit.losses:InfoNCELoss", aliases=("mnrl", "contrastive"))
LOSSES.register_lazy("cosent", "clmkit.losses:CoSENTLoss")
LOSSES.register_lazy("triplet", "clmkit.losses:TripletLoss")
