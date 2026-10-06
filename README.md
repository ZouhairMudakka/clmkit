# clmkit

> **Development preview — validation in progress.** Independent tests have found
> unresolved training, persistence and input-validation defects. This is not yet
> a verified public-alpha release. Read [current validation status](docs/VALIDATION_STATUS.md)
> before using the training or serving paths. Historical audit claims below do
> not override the current findings.

**A lightweight, pluggable skeleton framework for Contrastive Language Models.**
Encode, fine-tune, index, retrieve, rerank, evaluate and serve text-embedding models, and drop them into ML pipelines or AI agents.

[![CI](https://github.com/ZouhairMudakka/clmkit/actions/workflows/ci.yml/badge.svg)](https://github.com/ZouhairMudakka/clmkit/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Status: development preview](https://img.shields.io/badge/status-development_preview-orange)

---

A **contrastive language model (CLM)** maps text to vectors so that related texts land close together. It's trained by pulling matching pairs together and pushing non-matching ones apart (InfoNCE and friends). Examples include [Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-8B) (0.6B / 4B / 8B), E5, BGE and GTE. These models power semantic search, RAG, agent memory, tool routing, deduplication, clustering and classification.

`clmkit` gives you the whole loop in one small, readable codebase:

```
          ┌──────────── raw or fine-tuned ────────────┐
 text ──▶ │ Encoder (HF / Qwen3 / remote / hashing)   │ ──▶ vectors ──▶ Index ──▶ Retriever ──▶ Reranker
          └───────────────────────────────────────────┘                         │
                ▲                                                               ▼
   ContrastiveTrainer (InfoNCE · MRL · GradCache · LoRA)         Agents (memory · router · tools · MCP)
                ▲                                                  ML pipelines (sklearn · LangChain)
   data (JSONL) + hard-negative mining + evaluators               REST API (OpenAI-compatible)
```

## Highlights

- **Qwen3 presets.** Presets for 0.6B/4B/8B apply last-token pooling, left padding, the `Instruct: …\nQuery:` prompt and the `<|endoftext|>` pooling token. Independent CPU reference checks pass for the 0.6B embedding and reranker models at the tests' enforced tolerances; see [current evidence](docs/VALIDATION_STATUS.md). The 4B/8B models have not been exercised.
- **Experimental fine-tuning.** InfoNCE with in-batch and hard negatives, false-negative masking, CoSENT, triplet and Matryoshka losses. **GradCache** supports large contrastive batches through smaller forward passes, and **LoRA** is available through `peft`. Training has unresolved correctness findings; single-GPU 8B capacity has not been measured. See [validation status](docs/VALIDATION_STATUS.md).
- **Agent building blocks.** `SemanticMemory` with recency decay, `SemanticRouter` for tool/intent routing, and a `ToolKit` that exports tool definitions for **Anthropic** and **OpenAI**-style function calling, plus an **MCP server** for Claude Desktop, Claude Code and IDE agents.
- **ML pipelines.** A scikit-learn `EmbeddingTransformer`, a LangChain `Embeddings` adapter, a JSON/NumPy CLI, and IR metrics (nDCG/MRR/Recall/MAP) with BEIR loading.
- **Serving.** An OpenAI-compatible `/v1/embeddings` endpoint (so existing SDKs work), plus `/v1/search` and `/v1/rerank`. Bearer auth, request limits, and localhost binding by default.
- **Small and pluggable.** The core depends only on `numpy`, and heavy backends are lazy extras. A registry plus Python entry points let third-party packages add encoders, indexes, rerankers and losses. Use trusted model checkpoints and index snapshots; native FAISS and legacy checkpoint loading have separate trust requirements.

## Install

```bash
pip install "clmkit @ git+https://github.com/ZouhairMudakka/clmkit"            # numpy-only core
pip install "clmkit[hf] @ git+https://github.com/ZouhairMudakka/clmkit"        # + torch/transformers
pip install "clmkit[all] @ git+https://github.com/ZouhairMudakka/clmkit"       # + training, serving, MCP, YAML, sklearn
```

Extras: `hf`, `train`, `serve`, `mcp`, `faiss`, `yaml`, `sklearn`, `all`, `dev`. Python 3.10+.

## Quickstart

```python
from clmkit import Retriever, load_encoder

encoder = load_encoder("Qwen/Qwen3-Embedding-0.6B")   # or "Qwen/Qwen3-Embedding-8B", or "hashing" (no download)
retriever = Retriever(encoder)
retriever.add(["Paris is the capital of France.", "Bananas are rich in potassium."],
              metadata=[{"topic": "geo"}, {"topic": "food"}])

hits = retriever.search("What is France's capital city?", k=1)
print(hits[0].text, hits[0].score)
```

### Encoding (raw model, no fine-tuning)

```python
q = encoder.encode(["What is the capital of China?"], kind="query")            # instruction added for you
d = encoder.encode(["The capital of China is Beijing."], kind="document")      # documents: no instruction
q_code = encoder.encode("sort a list", kind="query", instruction="Given a question, retrieve code snippets")
small = encoder.encode(["..."], dim=256)                                         # Matryoshka truncation
```

Remote servers work the same way (vLLM, TEI, Ollama, OpenAI, or `clmkit serve`):

```python
encoder = load_encoder({"type": "openai", "model": "Qwen/Qwen3-Embedding-8B", "base_url": "http://localhost:8000/v1"})
```

### Fine-tuning Qwen3-Embedding

Data is JSONL with one example per line:

```json
{"query": "how do I reset my password", "positive": "Open Settings > Security ...", "negatives": ["To change your email ..."]}
```

```bash
clmkit mine  --model Qwen/Qwen3-Embedding-0.6B --train train.jsonl --corpus corpus.txt --output train_hn.jsonl
clmkit train --config configs/qwen3-embedding-8b-lora.yaml          # LoRA + GradCache + MRL on one GPU
clmkit eval  --model runs/qwen3-embedding-8b-lora/final --data eval.jsonl
```

The same flow from Python:

```python
from clmkit.training import ContrastiveTrainer, TrainConfig, LoraSettings
config = TrainConfig(output_dir="runs/x", batch_size=64, mini_batch_size=8,      # GradCache
                     lora=LoraSettings(r=16), matryoshka_dims=[4096, 1024, 256],
                     loss_kwargs={"temperature": 0.02, "false_negative_margin": 0.1})
ContrastiveTrainer(encoder, config, examples, evaluator=evaluator).train()
```

See [docs/TRAINING.md](docs/TRAINING.md) for data prep, hyper-parameters and hardware sizing.

### AI agents

```python
from clmkit.agents import SemanticMemory, SemanticRouter, Route, retriever_tools, memory_tools

tools = retriever_tools(retriever) + memory_tools(SemanticMemory(encoder, recency_half_life=7 * 86400))
tools.to_anthropic()          # -> tool definitions for Claude (Messages API / Agent SDK)
tools.to_openai()             # -> OpenAI-style function specs (also vLLM, Ollama, LiteLLM)
tools.call("knowledge_search", {"query": "refund policy", "k": 3})   # validated dispatch

router = SemanticRouter(encoder, [Route("billing", ["refund", "invoice"]), Route("security", ["password", "2FA"])])
router.route("I was charged twice")   # -> RouteMatch(name='billing', score=...)
```

MCP server for Claude Desktop or Claude Code: `clmkit mcp --index ./my_index` ([config](examples/07_serve_and_client.md)). More in [docs/AGENTS.md](docs/AGENTS.md).

### ML pipelines

```python
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from clmkit.integrations.sklearn import EmbeddingTransformer

clf = make_pipeline(EmbeddingTransformer("Qwen/Qwen3-Embedding-0.6B"), LogisticRegression()).fit(texts, labels)
```

### Serving

```bash
clmkit index --model Qwen/Qwen3-Embedding-0.6B --input docs.jsonl --output my_index
CLMKIT_API_KEY=secret clmkit serve --index my_index --port 8000
```

```python
from openai import OpenAI
OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="secret").embeddings.create(model="clmkit", input=["hi"])
```

## Supported models (presets)

| Model family | Pooling | Query prompt | Notes |
|---|---|---|---|
| `Qwen/Qwen3-Embedding-0.6B / 4B / 8B` | last token (`<\|endoftext\|>`) | `Instruct: {task}\nQuery:{q}` | MRL 32–1024 / 2560 / 4096 dims; independent 0.6B CPU check only |
| `Qwen/Qwen3-Reranker-0.6B / 4B / 8B` | LLM yes/no | `<Instruct>/<Query>/<Document>` | independent 0.6B CPU check only |
| `intfloat/e5-*`, `multilingual-e5-*-instruct`, `e5-mistral-7b-instruct` | mean / last token | `query: ` / `Instruct: …` | |
| `BAAI/bge-*-en-v1.5`, `bge-m3` | CLS | BGE retrieval prompt | |
| `thenlper/gte-*`, `sentence-transformers/all-*` | mean | – | |
| any other HF encoder | mean (override with `pooling=`) | configurable templates | |
| any OpenAI-compatible `/v1/embeddings` | server-side | client-side templates | vLLM, TEI, Ollama, OpenAI |

Fine-tuned checkpoints save a `clmkit_config.json`. Independent tests found that configured output dimensions do not survive save/reload; see the current validation status before relying on checkpoint round trips.

## Project layout

```
src/clmkit/
  encoders/      Encoder base, HF/Qwen3 encoder, OpenAI-compatible client, hashing baseline, presets
  index/         VectorIndex base, exact NumPy index, FAISS (flat/HNSW)
  training/      ContrastiveTrainer (GradCache, LoRA, MRL), TrainConfig, config-driven runner
  eval/          IR + STS metrics, RetrievalEvaluator, BEIR loader
  agents/        SemanticMemory, SemanticRouter, ToolKit (Anthropic/OpenAI tool specs)
  serve/         FastAPI app (OpenAI-compatible), MCP server
  integrations/  scikit-learn, LangChain
  losses.py · pooling.py · data.py · retrieval.py · rerank.py · registry.py · cli.py
configs/         ready-to-run training configs (Qwen3-Embedding-8B LoRA, 0.6B full)
examples/        7 runnable examples + sample data
docs/            DESIGN · TRAINING · AGENTS · GAP_ANALYSIS · AUDIT
```

## Quality

On the retained Windows CPU environment, 102 original tests pass and two opt-in real-model tests skip. Statement coverage is 96.81%; branch coverage is 87.33%. Ruff, mypy and bandit pass. A fresh NumPy-only wheel install passes 62 tests with 13 optional skips and two slow tests deselected. Hosted ordinary CI and the six-job installed-wheel matrix pass; independent contract tests still expose unresolved defects. Real 0.6B reference checks and a bounded CPU LoRA/save-reload probe pass. The fresh CPU dependency scan passes after updating setuptools, while the retained local environment still has cryptography advisories. See [current validation status](docs/VALIDATION_STATUS.md) and [validation harnesses](validation/README.md) for exact scope and run links.

## Documentation

- [Design & architecture](docs/DESIGN.md): principles, module map, data flow, extension points
- [Training guide](docs/TRAINING.md): data format, losses, GradCache, LoRA, hardware sizing for 8B
- [Agents & integrations](docs/AGENTS.md): memory, routing, tools, MCP, REST, LangChain, sklearn
- [Gap analysis & roadmap](docs/GAP_ANALYSIS.md): what's covered, what isn't, what's next
- [Audit report](docs/AUDIT.md): tests, static analysis, security review, real-model validation

## Contributing

Issues and PRs are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). Security reports go through [SECURITY.md](SECURITY.md).

## License

[Apache-2.0](LICENSE). Model weights are licensed separately by their authors (Qwen3-Embedding and Qwen3-Reranker are Apache-2.0).
