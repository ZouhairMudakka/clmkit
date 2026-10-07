from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from clmkit.config import ConfigError, from_dict, load_config
from clmkit.registry import ENCODERS, INDEXES, LOSSES, RERANKERS, Registry, RegistryError


def test_register_build_and_aliases() -> None:
    reg: Registry[type] = Registry("widgets")

    @reg.register("my_widget", aliases=("mw",))
    class Widget:
        def __init__(self, size: int = 1) -> None:
            self.size = size

    assert "my-widget" in reg and "MW" in reg and "my_widget" in reg
    assert reg.build("mw", size=3).size == 3
    assert reg.names() == ["my-widget"]
    with pytest.raises(RegistryError, match="already registered"):
        reg.register("my-widget")(Widget)
    with pytest.raises(RegistryError, match="unknown widgets"):
        reg.get("nope")


def test_lazy_registration_imports_on_demand() -> None:
    reg: Registry[object] = Registry("things")
    reg.register_lazy("ordered", "collections:OrderedDict")
    assert reg.get("ordered").__name__ == "OrderedDict"
    with pytest.raises(ValueError, match="module:attr"):
        reg.register_lazy("bad", "collections.OrderedDict")


def test_builtin_registries_are_populated() -> None:
    assert {"hashing", "hf", "openai"} <= set(ENCODERS.names())
    assert {"numpy", "faiss"} <= set(INDEXES.names())
    assert {"encoder", "llm-yes-no", "cross-encoder"} <= set(RERANKERS.names())
    assert {"infonce", "cosent", "triplet"} <= set(LOSSES.names())
    assert "Registry('encoders'" in repr(ENCODERS)


@dataclass
class Inner:
    r: int = 8


@dataclass
class Outer:
    lr: float = 1e-3
    inner: Inner | None = None
    tags: list[str] = field(default_factory=list)


def test_from_dict_strict_nested_and_suggestions() -> None:
    cfg = from_dict(Outer, {"lr": 0.1, "inner": {"r": 4}})
    assert cfg.lr == 0.1 and isinstance(cfg.inner, Inner) and cfg.inner.r == 4
    with pytest.raises(ConfigError, match="did you mean 'lr'"):
        from_dict(Outer, {"lrr": 1})
    with pytest.raises(ConfigError, match=r"inner: unknown Inner option"):
        from_dict(Outer, {"inner": {"rank": 1}})


def test_load_config_json_and_yaml(tmp_path) -> None:
    (tmp_path / "a.json").write_text(json.dumps({"x": 1}))
    assert load_config(tmp_path / "a.json") == {"x": 1}
    pytest.importorskip("yaml")
    (tmp_path / "b.yaml").write_text("x: 2\nnested: {y: [1, 2]}\n")
    assert load_config(tmp_path / "b.yaml") == {"x": 2, "nested": {"y": [1, 2]}}
    (tmp_path / "c.yaml").write_text("- 1\n- 2\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(tmp_path / "c.yaml")
    (tmp_path / "d.toml").write_text("x=1")
    with pytest.raises(ConfigError, match="unsupported"):
        load_config(tmp_path / "d.toml")


def test_yaml_loader_is_safe(tmp_path) -> None:
    pytest.importorskip("yaml")
    (tmp_path / "evil.yaml").write_text("x: !!python/object/apply:os.system ['echo pwned']\n")
    with pytest.raises(Exception, match="python/object"):
        load_config(tmp_path / "evil.yaml")


def test_entry_point_plugins_are_discovered(monkeypatch: pytest.MonkeyPatch) -> None:
    import clmkit.registry as registry_mod

    class FakeEP:
        name = "my_plugin"
        value = "collections:Counter"

    seen_groups = []

    def fake_entry_points(group: str):  # type: ignore[no-untyped-def]
        seen_groups.append(group)
        return [FakeEP()]

    monkeypatch.setattr(registry_mod, "entry_points", fake_entry_points)
    reg: Registry[object] = Registry("gadgets")
    reg.register("builtin")(dict)
    assert "my-plugin" in reg
    assert reg.get("my_plugin").__name__ == "Counter"
    assert seen_groups == ["clmkit.gadgets"]
    assert reg.names() == ["builtin", "my-plugin"]


def test_missing_dependency_hint_works_for_unpublished_package(monkeypatch: pytest.MonkeyPatch) -> None:
    from clmkit import utils

    def unavailable(name):
        raise ImportError(name)

    monkeypatch.setattr(utils.importlib, "import_module", unavailable)
    with pytest.raises(utils.MissingDependencyError) as error:
        utils.require("torch")
    assert 'python -m pip install ".[hf]"' in str(error.value)
    assert "https://github.com/ZouhairMudakka/clmkit#install" in str(error.value)
