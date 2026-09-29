# Using clmkit in AI agents and ML flows

A contrastive encoder is the "semantic sense" of an agent. It decides what to remember, what to look up, and which tool fits a request. clmkit packages those uses as small, framework-agnostic parts.

## Patterns at a glance

| need | component | notes |
|---|---|---|
| knowledge lookup / RAG | `Retriever` + `retriever_tools()` | metadata filters, reranking, persistence |
| long-term memory | `SemanticMemory` + `memory_tools()` | recency decay, `forget_older_than` |
| tool / intent / sub-agent selection | `SemanticRouter` + `router_tools()` | abstains below threshold → fall back to the LLM |
| tool shortlisting for big toolsets | `SemanticRouter.top_k()` | give the LLM only the k most relevant tools |
| expose to any MCP client | `clmkit mcp` / `serve.mcp_server.build_server` | Claude Desktop, Claude Code, IDE agents |
| expose over HTTP | `clmkit serve` | OpenAI-compatible `/v1/embeddings`, `/v1/search`, `/v1/rerank` |
| classic ML features | `integrations.sklearn.EmbeddingTransformer` | classification, clustering, dedup |
| LangChain | `integrations.langchain.ClmkitEmbeddings` | drop-in `Embeddings` |

## Tool calling with any LLM

```python
from clmkit import Retriever, load_encoder
from clmkit.agents import SemanticMemory, memory_tools, retriever_tools

encoder = load_encoder("Qwen/Qwen3-Embedding-0.6B")
kb = Retriever(encoder); kb.add(documents)
kit = retriever_tools(kb, description="Search the company handbook.") + memory_tools(SemanticMemory(encoder))

tools_for_claude = kit.to_anthropic()   # [{"name", "description", "input_schema"}, ...]
tools_for_openai = kit.to_openai()      # [{"type": "function", "function": {...}}, ...]

# In your agent loop, when the model emits a tool call:
result = kit.call(tool_name, tool_arguments)   # validates types/bounds/extra keys; raises ToolError on bad input
```

`ToolError` messages are written for the model to read, so return them as the tool result and let the model correct its call. The MCP server does exactly that.

### Instructions for asymmetric models

Qwen3-Embedding is instruction-aware. Give each use its own instruction:

```python
SemanticMemory(encoder, instruction="Given a user message, retrieve facts previously stated about the user")
SemanticRouter(encoder, routes, instruction="Given a user request, retrieve the tool that can fulfil it")
Retriever(encoder, query_instruction="Given a support question, retrieve the help-center article that answers it")
```

## Routing

```python
from clmkit.agents import Route, SemanticRouter

router = SemanticRouter(encoder, [
    Route("billing", ["refund", "invoice", "charged twice"]),
    Route("security", ["reset password", "2FA", "account locked"], threshold=0.6),
    Route("maps", [], description="directions, navigation and travel times"),
], threshold=0.5, aggregation="max")

match = router.route(user_message)
if match is None:
    ...  # nothing confident: let the LLM decide
```

## Memory

```python
from clmkit.agents import SemanticMemory

memory = SemanticMemory(encoder, recency_half_life=7 * 24 * 3600)
memory.remember("User's name is Sam and they prefer metric units", {"type": "profile", "user": "u123"})
memory.recall("what units should I use?", k=3, filter={"user": "u123"})   # per-user isolation via filters
memory.forget_older_than(90 * 24 * 3600)
memory.save("memory_store")
```

## MCP

```bash
clmkit index --model Qwen/Qwen3-Embedding-0.6B --input docs.jsonl --output /abs/path/kb
```

```json
{"mcpServers": {"kb": {"command": "clmkit", "args": ["mcp", "--index", "/abs/path/kb"]}}}
```

Works with MCP Python SDK 1.x (≥ 1.28.1) and 2.x.

## Security notes for agent builders

- **Retrieved text is untrusted.** Documents and memories may contain prompt-injection attempts. Present tool results to the model as data (the default for tool results in the Claude and OpenAI APIs), and don't grant write tools (`allow_write=True`, `--allow-writes`) to agents that read untrusted content unless you need them.
- **Isolate tenants** with metadata filters (`filter={"user": ...}`) or separate indexes. Don't rely on the model to scope queries.
- The REST server binds to `127.0.0.1` by default. Set `CLMKIT_API_KEY` before exposing it anywhere else, and put it behind a proper gateway for rate limiting.
