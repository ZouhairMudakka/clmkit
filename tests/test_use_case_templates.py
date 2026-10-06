"""Exercise copyable templates against the active installed clmkit package."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from clmkit import Retriever, load_encoder

TEMPLATES = Path(__file__).resolve().parents[1] / "templates"


def load_template(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, TEMPLATES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_support_search_ranking_citations_and_persistence(tmp_path: Path) -> None:
    template = load_template("support_search")
    encoder = load_encoder("hashing")
    retriever = template.build_retriever(encoder)
    query = "refund policy receipt within 30 days"
    before = template.search_support(retriever, query, trusted_tenant_id="north")
    assert before["status"] == "matched"
    top = before["matches"][0]
    assert top["id"] == "north:refunds:v1"
    assert "30 days" in top["text"]
    assert top["citation"] == {
        "document_id": "north:refunds:v1",
        "title": "Refund policy",
        "url": "https://example.invalid/north/refunds",
        "revision": "v1",
    }
    retriever.save(tmp_path / "faq")
    loaded = Retriever.load(tmp_path / "faq", encoder, strict=True)
    assert template.search_support(loaded, query, trusted_tenant_id="north") == before
    with pytest.raises(ValueError, match="built with encoder"):
        Retriever.load(tmp_path / "faq", load_encoder("hashing", dim=1024), strict=True)


def test_support_search_excludes_more_similar_other_tenant() -> None:
    template = load_template("support_search")
    retriever = template.build_retriever(load_encoder("hashing"))
    other_tenant = next(row for row in template.FAQS if row["metadata"]["tenant_id"] == "south")
    query = other_tenant["text"]
    assert retriever.search(query, k=1)[0].id == other_tenant["id"]
    result = template.search_support(retriever, query, trusted_tenant_id="north", min_score=-1)
    assert result["matches"]
    assert all(hit["id"].startswith("north:") for hit in result["matches"])


@pytest.mark.parametrize(
    ("query", "tenant"),
    [("volcanic basalt tectonics", "north"), ("  ", "north"), ("refund receipt", "missing")],
)
def test_support_search_no_matches(query: str, tenant: str) -> None:
    template = load_template("support_search")
    result = template.search_support(template.build_retriever(load_encoder("hashing")), query, trusted_tenant_id=tenant)
    assert result == {"query": query, "status": "no_match", "matches": []}


def test_support_search_empty_store_and_missing_scope() -> None:
    template = load_template("support_search")
    empty = Retriever(load_encoder("hashing"))
    assert template.search_support(empty, "refund", trusted_tenant_id="north")["matches"] == []
    with pytest.raises(ValueError, match="trusted_tenant_id"):
        template.search_support(empty, "refund", trusted_tenant_id=" ")


def test_memory_scope_write_override_and_reload(tmp_path: Path) -> None:
    template = load_template("assistant_memory")
    encoder = load_encoder("hashing")
    memory = template.ScopedMemory(encoder)
    alice_id = memory.remember(
        "I prefer concise project updates by email.",
        trusted_user_id="alice",
        memory_id="updates",
        metadata={"user_id": "bob", "kind": "preference"},
    )
    bob_text = "I prefer detailed project updates by phone."
    bob_id = memory.remember(bob_text, trusted_user_id="bob", memory_id="updates")
    assert alice_id != bob_id
    # Even Bob's exact text must never reveal Bob's memory to Alice.
    before = memory.recall(bob_text, trusted_user_id="alice")
    assert [hit["id"] for hit in before] == [alice_id]
    assert before[0]["metadata"]["user_id"] == "alice"
    assert before[0]["metadata"]["kind"] == "preference"
    assert "email" in before[0]["text"]
    assert [hit["id"] for hit in memory.recall(bob_text, trusted_user_id="bob")] == [bob_id]
    assert memory.recall(bob_text, trusted_user_id="unknown") == []
    assert memory.recall("  ", trusted_user_id="alice") == []
    memory.save(tmp_path / "memory")
    loaded = template.ScopedMemory.load(tmp_path / "memory", encoder)
    assert loaded.recall(bob_text, trusted_user_id="alice") == before
    assert [hit["id"] for hit in loaded.recall(bob_text, trusted_user_id="bob")] == [bob_id]
    assert loaded.recall(bob_text, trusted_user_id="unknown") == []
    with pytest.raises(ValueError, match="built with encoder"):
        template.ScopedMemory.load(tmp_path / "memory", load_encoder("hashing", dim=1024))


def test_memory_rejects_missing_scope_and_handles_empty_store() -> None:
    template = load_template("assistant_memory")
    memory = template.ScopedMemory(load_encoder("hashing"))
    assert memory.recall("project updates", trusted_user_id="alice") == []
    with pytest.raises(ValueError, match="trusted_user_id"):
        memory.remember("a preference", trusted_user_id="", memory_id="pref")
    with pytest.raises(ValueError, match="trusted_user_id"):
        memory.recall("a preference", trusted_user_id=" ")


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("please refund the duplicate invoice charge", "billing"),
        ("help reset my account password", "account"),
        ("track my parcel delivery", "delivery"),
        ("volcanic basalt tectonics", None),
        ("", None),
    ],
)
def test_routing_held_out_requests_and_no_mutation(query: str, expected: str | None) -> None:
    template = load_template("support_routing")
    router = template.build_router(load_encoder("hashing"))
    calibration = template.calibration_report(router)
    assert calibration["unknown_max_score"] < router.threshold < calibration["known_min_score"]
    # Selection must not mutate the queue definitions or dispatch an action.
    before = {name: (list(route.utterances), dict(route.metadata)) for name, route in router.routes.items()}
    result = template.select_route(router, query)
    assert result["route"] == expected
    assert result["status"] == ("abstained" if expected is None else "selected")
    assert result["action_executed"] is False
    assert before == {name: (list(route.utterances), dict(route.metadata)) for name, route in router.routes.items()}
    if expected is not None:
        assert result["score"] >= router.threshold
    else:
        assert result["score"] is None


def test_routing_accepts_application_threshold_and_encoder() -> None:
    template = load_template("support_routing")
    encoder = load_encoder("hashing", dim=1024)
    router = template.build_router(encoder, threshold=1.1)
    assert router.encoder is encoder
    assert template.select_route(router, "account password reset login")["status"] == "abstained"


@pytest.mark.parametrize("name", ["support_search", "assistant_memory", "support_routing"])
def test_template_cli_isolated_json_and_no_leftovers(name: str, tmp_path: Path) -> None:
    # -I ignores PYTHONPATH and excludes the script directory from sys.path.
    # The installed-wheel harness copies templates/ beside its copied tests/.
    env = {**os.environ, "TMP": str(tmp_path), "TEMP": str(tmp_path), "TMPDIR": str(tmp_path)}
    completed = subprocess.run(
        [sys.executable, "-I", str(TEMPLATES / f"{name}.py")],
        cwd=tmp_path,
        env=env,
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    result = json.loads(completed.stdout)
    assert completed.stderr == ""
    assert result["use_case"] == name
    assert result["synthetic_data"] is True
    assert result["encoder"] == "hashing-512"
    assert list(tmp_path.iterdir()) == []
    if name == "support_search":
        assert result["reload_equal"] is True
        assert result["known"]["matches"][0]["id"] == "north:refunds:v1"
        assert result["unknown"]["status"] == "no_match"
    elif name == "assistant_memory":
        assert result["reload_equal"] is True
        assert result["unknown_user"] == []
        assert [hit["metadata"]["user_id"] for hit in result["alice"]] == ["alice"]
        assert [hit["metadata"]["user_id"] for hit in result["bob"]] == ["bob"]
    else:
        assert [row["route"] for row in result["requests"]] == ["billing", "account", "delivery", None, None]
        assert all(row["action_executed"] is False for row in result["requests"])
