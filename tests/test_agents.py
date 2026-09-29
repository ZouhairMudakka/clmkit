from __future__ import annotations

import json

import pytest

from clmkit import HashingEncoder, Retriever
from clmkit.agents import (
    Route,
    SemanticMemory,
    SemanticRouter,
    ToolError,
    ToolKit,
    memory_tools,
    retriever_tools,
    router_tools,
)


class FakeClock:
    def __init__(self) -> None:
        self.t = 1_000.0

    def __call__(self) -> float:
        return self.t


# ------------------------------------------------------------------ memory --
def test_memory_remember_recall_forget(hashing: HashingEncoder) -> None:
    mem = SemanticMemory(hashing)
    mid = mem.remember("The user's favourite language is Python", {"kind": "preference"})
    mem.remember_many(
        ["The user lives in Beirut", "Meeting with Sara on Friday"], [{"kind": "fact"}, {"kind": "event"}]
    )
    assert len(mem) == 3
    top = mem.recall("which programming language does the user like?", k=1)[0]
    assert top.id == mid and top.metadata["kind"] == "preference"
    assert [h.metadata["kind"] for h in mem.recall("user", k=3, filter={"kind": "event"})] == ["event"]
    assert mem.recall("python language", k=3, min_score=0.99) == []
    assert mem.forget([mid]) == 1 and len(mem) == 2


def test_memory_recency_decay_and_expiry(hashing: HashingEncoder) -> None:
    clock = FakeClock()
    mem = SemanticMemory(hashing, recency_half_life=100.0, clock=clock)
    old = mem.remember("the deploy password rotation happens monthly")
    clock.t += 1_000  # ten half-lives later
    new = mem.remember("the deploy happens on monday")
    hits = mem.recall("when does the deploy happen", k=2)
    assert hits[0].id == new
    old_hit = next(h for h in hits if h.id == old)
    assert old_hit.score < 0.01
    assert mem.forget_older_than(500) == 1 and len(mem) == 1
    with pytest.raises(ValueError):
        SemanticMemory(hashing, recency_half_life=0)


def test_memory_persistence(tmp_path, hashing: HashingEncoder) -> None:
    mem = SemanticMemory(hashing)
    mem.remember("remember the milk", {"kind": "todo"})
    mem.save(tmp_path / "mem")
    loaded = SemanticMemory.load(tmp_path / "mem", HashingEncoder(dim=256))
    assert loaded.recall("milk", k=1)[0].metadata["kind"] == "todo"


# ------------------------------------------------------------------ router --
ROUTES = [
    Route("weather", ["what's the weather like", "will it rain tomorrow", "temperature forecast"]),
    Route("calendar", ["schedule a meeting", "what's on my calendar", "book an appointment"]),
    Route("code", ["fix this python bug", "write a function", "refactor my code"], threshold=0.2),
]


@pytest.mark.parametrize("aggregation", ["max", "mean", "centroid"])
def test_router_picks_the_right_route(hashing: HashingEncoder, aggregation: str) -> None:
    router = SemanticRouter(hashing, ROUTES, threshold=0.1, aggregation=aggregation)  # type: ignore[arg-type]
    assert router.route("is it going to rain tomorrow?").name == "weather"  # type: ignore[union-attr]
    assert router.route("please schedule a meeting with Bob").name == "calendar"  # type: ignore[union-attr]
    assert router.top_k("write a python function", k=2)[0][0] == "code"


def test_router_threshold_and_management(hashing: HashingEncoder) -> None:
    router = SemanticRouter(hashing, ROUTES, threshold=0.95)
    assert router.route("completely unrelated gibberish zzzz") is None
    assert SemanticRouter(hashing).route("x") is None
    with pytest.raises(ValueError, match="duplicate"):
        router.add_route(Route("weather", ["x"]))
    router.remove_route("weather")
    assert "weather" not in router.scores("rain")
    with pytest.raises(ValueError):
        Route("empty", [])
    with pytest.raises(ValueError):
        SemanticRouter(hashing, aggregation="median")  # type: ignore[arg-type]
    described = SemanticRouter(hashing, [Route("maps", [], description="directions and navigation")], threshold=0.1)
    assert described.route("navigation directions to the airport").name == "maps"  # type: ignore[union-attr]


# ------------------------------------------------------------------- tools --
@pytest.fixture
def kit(hashing: HashingEncoder) -> ToolKit:
    r = Retriever(hashing)
    r.add(["Qwen3-Embedding-8B produces 4096-dim vectors", "InfoNCE is a contrastive loss"], ids=["a", "b"])
    return (
        retriever_tools(r, allow_write=True)
        + memory_tools(SemanticMemory(hashing))
        + router_tools(SemanticRouter(hashing, ROUTES, threshold=0.1))
    )


def test_tool_specs_formats(kit: ToolKit) -> None:
    assert kit.names() == ["knowledge_search", "knowledge_add", "memory_remember", "memory_recall", "route_request"]
    anth = kit.to_anthropic()
    assert anth[0]["input_schema"]["required"] == ["query"]
    oai = kit.to_openai()
    assert oai[0]["type"] == "function" and oai[0]["function"]["parameters"]["type"] == "object"
    json.dumps(anth)
    json.dumps(oai)
    assert len(kit) == 5


def test_tool_calls_and_validation(kit: ToolKit) -> None:
    hits = kit.call("knowledge_search", {"query": "contrastive loss", "k": 1})
    assert hits[0]["id"] == "b" and set(hits[0]) == {"id", "score", "text", "metadata"}
    new = kit.call("knowledge_add", '{"text": "LoRA adapts large models cheaply", "metadata": {"src": "t"}}')
    assert kit.call("knowledge_search", {"query": "LoRA adapts models", "k": 1})[0]["id"] == new["id"]
    kit.call("memory_remember", {"text": "user prefers short answers", "tags": {"k": "pref"}})
    recalled = kit.call("memory_recall", {"query": "short answers"})
    assert recalled[0]["metadata"] == {"k": "pref"}  # internal "_created_at" is hidden
    assert kit.call("route_request", {"request": "will it rain"})["match"] == "weather"
    for bad_args, msg in [
        ({"k": 2}, "missing required"),
        ({"query": "x", "k": 0}, ">= 1"),
        ({"query": "x", "k": 999}, "<= 20"),
        ({"query": "x", "k": True}, "integer"),
        ({"query": 5}, "string"),
        ({"query": "x", "extra": 1}, "unexpected"),
        ({"query": "x" * 30_000}, "longer than"),
        ("[1, 2]", "must be an object"),
        ("{not json", "not valid JSON"),
    ]:
        with pytest.raises(ToolError, match=msg):
            kit.call("knowledge_search", bad_args)
    with pytest.raises(ToolError, match="unknown tool"):
        kit.call("rm_rf", {})
    with pytest.raises(ToolError, match="missing required"):  # empty string -> {} -> no query
        kit.call("memory_recall", "")
    with pytest.raises(ValueError, match="duplicate tool"):
        kit + kit
