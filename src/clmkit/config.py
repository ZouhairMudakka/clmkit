"""Config files (JSON or YAML) and strict dataclass construction.

Strictness matters for training configs: a typo like ``learnig_rate`` must be an
error, not a silently ignored key that wastes a GPU-day.
"""

from __future__ import annotations

import dataclasses
import difflib
import json
import types
import typing
from pathlib import Path
from typing import Any, TypeVar

from clmkit.utils import require

T = TypeVar("T")


class ConfigError(ValueError):
    """Invalid configuration."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a ``.json``, ``.yaml`` or ``.yml`` file into a dict (YAML via ``safe_load`` only)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        yaml = require("yaml")
        data = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        raise ConfigError(f"unsupported config format {path.suffix!r} (use .json, .yaml or .yml)")
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def from_dict(cls: type[T], data: dict[str, Any] | None, *, where: str = "") -> T:
    """Build dataclass ``cls`` from ``data``, rejecting unknown keys (with suggestions)
    and recursing into nested dataclass-typed fields."""
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    data = dict(data or {})
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        tips = []
        for key in unknown:
            close = difflib.get_close_matches(key, names, n=1)
            tips.append(f"{key!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
        prefix = f"{where}: " if where else ""
        raise ConfigError(f"{prefix}unknown {cls.__name__} option(s): {', '.join(tips)}")
    for key, value in list(data.items()):
        nested = _dataclass_in(hints.get(key))
        if nested is not None and isinstance(value, dict):
            data[key] = from_dict(nested, value, where=f"{where}.{key}" if where else key)
    return cls(**data)


def _dataclass_in(tp: Any) -> type | None:
    """Return the dataclass inside ``X`` / ``X | None`` / ``Optional[X]``, else None."""
    if tp is None:
        return None
    if dataclasses.is_dataclass(tp) and isinstance(tp, type):
        return tp
    if typing.get_origin(tp) in (typing.Union, types.UnionType):
        for arg in typing.get_args(tp):
            if dataclasses.is_dataclass(arg) and isinstance(arg, type):
                return arg
    return None


def to_dict(obj: Any) -> dict[str, Any]:
    return dataclasses.asdict(obj)
