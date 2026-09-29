# Serving clmkit (REST + MCP)

## REST (OpenAI-compatible)

```bash
pip install "clmkit[hf,serve]"
clmkit index --model Qwen/Qwen3-Embedding-0.6B --input examples/data/docs.jsonl --output my_index
export CLMKIT_API_KEY=change-me            # optional bearer auth (always set it off-localhost)
clmkit serve --index my_index --port 8000  # binds 127.0.0.1 by default
```

Any OpenAI SDK works against `/v1/embeddings`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="change-me")
vec = client.embeddings.create(model="clmkit", input=["hello world"]).data[0].embedding
```

clmkit-specific extras (plain HTTP):

```bash
curl -s localhost:8000/v1/search -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" -d '{"query": "reset password", "k": 3}'

curl -s localhost:8000/v1/embeddings -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d '{"input": "reset password", "input_type": "query", "dimensions": 256}'
```

Interactive docs: <http://127.0.0.1:8000/docs>.

## MCP (for Claude Desktop / Claude Code / IDE agents)

```bash
pip install "clmkit[hf,mcp]"
```

```json
{
  "mcpServers": {
    "clmkit-kb": {
      "command": "clmkit",
      "args": ["mcp", "--index", "/absolute/path/to/my_index"]
    }
  }
}
```

The agent then gets a `knowledge_search` tool backed by your index (add `--allow-writes`
to also expose `knowledge_add`).

## Using a remote embedding server as the encoder

Run the model with vLLM / TEI and point clmkit at it; indexing, search, agents and eval
work unchanged:

```bash
vllm serve Qwen/Qwen3-Embedding-8B --task embed --port 8001
clmkit index --model openai:Qwen/Qwen3-Embedding-8B --model-arg base_url=http://127.0.0.1:8001/v1 \
  --input examples/data/docs.jsonl --output my_index
```
