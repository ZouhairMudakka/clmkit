"""Expose retrieval and memory as LLM tools, independent of any agent framework.

A :class:`ToolKit` holds JSON-Schema tool definitions plus the Python handlers.
Export them in the format your stack expects and dispatch calls back::

    kit = retriever_tools(retriever) + memory_tools(memory)
    tools = kit.to_anthropic()     # Claude Messages API / Agent SDK
    tools = kit.to_openai()        # OpenAI-style function calling (also vLLM, Ollama, LiteLLM)
    result = kit.call(name, arguments)   # arguments: dict or JSON string

The same toolkit backs the MCP server (``clmkit.serve.mcp_server``).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from clmkit.agents.memory import SemanticMemory
    from clmkit.agents.router import SemanticRouter
    from clmkit.retrieval import Retriever


class ToolError(ValueError):
    """Invalid tool call (unknown tool or bad arguments). Safe to show to the model."""


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]


class ToolKit:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self.tools: dict[str, Tool] = {}
        for t in tools or []:
            self.add(t)

    def add(self, tool: Tool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"duplicate tool {tool.name!r}")
        self.tools[tool.name] = tool

    def __add__(self, other: ToolKit) -> ToolKit:
        return ToolKit([*self.tools.values(), *other.tools.values()])

    def __len__(self) -> int:
        return len(self.tools)

    def names(self) -> list[str]:
        return list(self.tools)

    def to_anthropic(self) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in self.tools.values()
        ]

    def to_openai(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": t.input_schema},
            }
            for t in self.tools.values()
        ]

    def call(self, name: str, arguments: dict[str, Any] | str | None = None) -> Any:
        """Validate ``arguments`` against the tool schema and run the handler."""
        tool = self.tools.get(name)
        if tool is None:
            raise ToolError(f"unknown tool {name!r}; available: {self.names()}")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                raise ToolError(f"arguments for {name!r} are not valid JSON: {exc.msg}") from exc
        args = _validate(tool.input_schema, arguments or {}, name)
        return tool.handler(**args)


_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
}


def _validate(schema: dict[str, Any], args: Any, tool: str) -> dict[str, Any]:
    """Minimal JSON-Schema check (types, required, bounds, no extras) - enough for LLM-produced args."""
    if not isinstance(args, dict):
        raise ToolError(f"{tool}: arguments must be an object")
    props: dict[str, Any] = schema.get("properties", {})
    missing = [k for k in schema.get("required", []) if k not in args]
    if missing:
        raise ToolError(f"{tool}: missing required argument(s) {missing}")
    extra = sorted(set(args) - set(props))
    if extra and not schema.get("additionalProperties", False):
        raise ToolError(f"{tool}: unexpected argument(s) {extra}")
    out = {}
    for key, value in args.items():
        spec = props.get(key, {})
        expected = _JSON_TYPES.get(spec.get("type", ""), (object,))
        is_bool = isinstance(value, bool)
        if not isinstance(value, expected) or (is_bool and spec.get("type") in ("integer", "number")):
            raise ToolError(f"{tool}: {key!r} must be of type {spec.get('type')}")
        if "minimum" in spec and value < spec["minimum"]:
            raise ToolError(f"{tool}: {key!r} must be >= {spec['minimum']}")
        if "maximum" in spec and value > spec["maximum"]:
            raise ToolError(f"{tool}: {key!r} must be <= {spec['maximum']}")
        if "maxLength" in spec and isinstance(value, (str, list)) and len(value) > spec["maxLength"]:
            raise ToolError(f"{tool}: {key!r} is longer than {spec['maxLength']} characters")
        if "enum" in spec and value not in spec["enum"]:
            raise ToolError(f"{tool}: {key!r} must be one of {spec['enum']}")
        out[key] = value
    return out


def _hits(hits: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": h.id,
            "score": round(h.score, 4),
            "text": h.text,
            "metadata": {k: v for k, v in h.metadata.items() if not k.startswith("_")},
        }
        for h in hits
    ]


def retriever_tools(
    retriever: Retriever,
    *,
    prefix: str = "knowledge",
    description: str | None = None,
    max_k: int = 20,
    allow_write: bool = False,
    max_chars: int = 20_000,
) -> ToolKit:
    """``<prefix>_search`` (and, if ``allow_write``, ``<prefix>_add``) over a :class:`Retriever`."""

    def search(query: str, k: int = 5) -> list[dict[str, Any]]:
        return _hits(retriever.search(query, k=k))

    kit = ToolKit(
        [
            Tool(
                name=f"{prefix}_search",
                description=description
                or "Semantic search over the knowledge base. Returns the most relevant "
                "passages with similarity scores. Use it before answering factual questions.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural-language search query.",
                            "maxLength": max_chars,
                        },
                        "k": {"type": "integer", "description": "Number of results.", "minimum": 1, "maximum": max_k},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                handler=search,
            )
        ]
    )
    if allow_write:

        def add(text: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
            return {"id": retriever.add([text], metadata=[metadata or {}])[0]}

        kit.add(
            Tool(
                name=f"{prefix}_add",
                description="Add a passage to the knowledge base so it can be found by later searches.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "maxLength": max_chars},
                        "metadata": {"type": "object", "description": "Optional key/value tags."},
                    },
                    "required": ["text"],
                    "additionalProperties": False,
                },
                handler=add,
            )
        )
    return kit


def memory_tools(memory: SemanticMemory, *, prefix: str = "memory", max_k: int = 20, max_chars: int = 5_000) -> ToolKit:
    """``<prefix>_remember`` / ``<prefix>_recall`` over a :class:`SemanticMemory`."""

    def remember(text: str, tags: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"id": memory.remember(text, tags or {})}

    def recall(query: str, k: int = 5) -> list[dict[str, Any]]:
        return _hits(memory.recall(query, k=k))

    return ToolKit(
        [
            Tool(
                f"{prefix}_remember",
                "Save a durable fact, preference or observation for later recall.",
                {
                    "type": "object",
                    "properties": {"text": {"type": "string", "maxLength": max_chars}, "tags": {"type": "object"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
                remember,
            ),
            Tool(
                f"{prefix}_recall",
                "Recall previously saved memories relevant to a query.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "maxLength": max_chars},
                        "k": {"type": "integer", "minimum": 1, "maximum": max_k},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                recall,
            ),
        ]
    )


def router_tools(router: SemanticRouter, *, name: str = "route_request", max_chars: int = 5_000) -> ToolKit:
    """A tool that returns the best-matching routes for a request (useful for planner agents)."""

    def route(request: str, k: int = 3) -> dict[str, Any]:
        match = router.route(request)
        return {
            "match": match.name if match else None,
            "candidates": [{"route": n, "score": round(s, 4)} for n, s in router.top_k(request, k)],
        }

    return ToolKit(
        [
            Tool(
                name,
                "Find which capability/tool best handles a user request.",
                {
                    "type": "object",
                    "properties": {
                        "request": {"type": "string", "maxLength": max_chars},
                        "k": {"type": "integer", "minimum": 1, "maximum": 20},
                    },
                    "required": ["request"],
                    "additionalProperties": False,
                },
                route,
            )
        ]
    )
