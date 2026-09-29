"""Model Context Protocol server (``pip install "clmkit[mcp]"``).

Exposes a :class:`~clmkit.agents.tools.ToolKit` (knowledge search, memory, routing)
to any MCP client - Claude Desktop/Code, IDE agents, etc. - over stdio::

    clmkit mcp --index ./my_index --model Qwen/Qwen3-Embedding-0.6B

Example client config (``claude_desktop_config.json`` / ``.mcp.json``)::

    {"mcpServers": {"clmkit": {"command": "clmkit",
                               "args": ["mcp", "--index", "/abs/path/to/index", "--model", "hashing"]}}}

Supports both MCP Python SDK generations: 2.x (handlers passed to the ``Server``
constructor) and 1.x (decorator registration).
"""

from __future__ import annotations

import inspect
import json
from typing import Any

from clmkit.agents.tools import ToolError, ToolKit
from clmkit.utils import require


def _tool_result(toolkit: ToolKit, name: str, arguments: dict[str, Any] | None) -> tuple[str, bool]:
    """Run a tool; errors come back as readable text flagged ``is_error`` so the model can self-correct."""
    try:
        return json.dumps(toolkit.call(name, arguments or {}), ensure_ascii=False, default=str), False
    except ToolError as exc:
        return json.dumps({"error": str(exc)}), True


def build_server(toolkit: ToolKit, *, name: str = "clmkit") -> Any:
    """Create a low-level ``mcp`` server whose tools are exactly ``toolkit``'s tools."""
    require("mcp")
    from mcp import types as mcp_types
    from mcp.server.lowlevel import Server

    # camelCase field names (inputSchema, isError) are valid on both SDK generations:
    # 1.x declares them directly, 2.x accepts them as aliases. Typed as Any for that reason.
    types: Any = mcp_types

    def tools() -> list[Any]:
        return [
            types.Tool(name=t["name"], description=t["description"], inputSchema=t["input_schema"])
            for t in toolkit.to_anthropic()
        ]

    if "on_list_tools" in inspect.signature(Server.__init__).parameters:  # mcp >= 2

        async def on_list_tools(ctx: Any, params: Any) -> Any:
            return types.ListToolsResult(tools=tools())

        async def on_call_tool(ctx: Any, params: Any) -> Any:
            text, is_error = _tool_result(toolkit, params.name, params.arguments)
            return types.CallToolResult(content=[types.TextContent(type="text", text=text)], isError=is_error)

        return Server(name, on_list_tools=on_list_tools, on_call_tool=on_call_tool)

    server: Any = Server(name)  # mcp 1.x

    @server.list_tools()
    async def list_tools() -> list[Any]:
        return tools()

    @server.call_tool()
    async def call_tool(tool: str, arguments: dict[str, Any] | None) -> Any:
        text, is_error = _tool_result(toolkit, tool, arguments)
        content = [types.TextContent(type="text", text=text)]
        # 1.x only lets a handler flag an error by returning a full CallToolResult.
        return types.CallToolResult(content=content, isError=True) if is_error else content

    return server


def run_stdio(toolkit: ToolKit, *, name: str = "clmkit") -> None:  # pragma: no cover - needs a client
    import anyio
    from mcp.server.stdio import stdio_server

    server = build_server(toolkit, name=name)

    async def main() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(main)
