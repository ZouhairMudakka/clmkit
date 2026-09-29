"""Give an AI agent semantic memory, a knowledge base and a tool router.

    python examples/04_agent_tools.py

`ToolKit.to_anthropic()` / `.to_openai()` produce tool definitions for any LLM API that
supports function calling; `ToolKit.call(name, args)` executes what the model asked for.
Below, a scripted "model" stands in for the LLM so the example runs offline.
"""

import json

from clmkit import HashingEncoder, Retriever
from clmkit.agents import Route, SemanticMemory, SemanticRouter, memory_tools, retriever_tools, router_tools

encoder = HashingEncoder(dim=1024)  # or load_encoder("Qwen/Qwen3-Embedding-0.6B")

knowledge = Retriever(encoder)
knowledge.add(
    [
        "Refunds are available within 30 days of purchase.",
        "Two-factor authentication is under Settings > Security > 2FA.",
        "The API rate limit is 600 requests per minute per key.",
    ]
)
memory = SemanticMemory(encoder, recency_half_life=7 * 24 * 3600)  # memories fade over ~a week
router = SemanticRouter(
    encoder,
    [
        Route("billing", ["refund", "invoice", "payment", "charge on my card"]),
        Route("security", ["password", "2FA", "login problem", "account locked"]),
        Route("developer", ["API", "rate limit", "SDK", "webhook"]),
    ],
    threshold=0.1,
)

tools = retriever_tools(knowledge) + memory_tools(memory) + router_tools(router)
print("Tools for the LLM (Anthropic format):")
print(json.dumps([t["name"] for t in tools.to_anthropic()], indent=2))

# A scripted agent turn: the "model" routes, remembers a preference, searches, recalls.
script = [
    ("route_request", {"request": "I was charged twice, can I get my money back?"}),
    ("memory_remember", {"text": "User prefers answers in bullet points", "tags": {"type": "preference"}}),
    ("knowledge_search", {"query": "refund window", "k": 1}),
    ("memory_recall", {"query": "how does the user like answers formatted?", "k": 1}),
    ("knowledge_search", {"query": "x", "k": 500}),  # invalid -> ToolError the model can read and fix
]
for name, args in script:
    try:
        result = tools.call(name, args)
    except ValueError as exc:
        result = {"error": str(exc)}
    print(f"\n> {name}({json.dumps(args)})\n{json.dumps(result, indent=2)[:400]}")
