# clmkit

> **Public alpha — validated CPU scope.** Audited training, persistence and input
> contracts pass the corrected test suite and hosted checks. Read
> [validation status and limits](docs/VALIDATION_STATUS.md) before using training
> or serving paths. GPU/4B/8B capacity and production reliability remain unverified.

**A lightweight Python framework for contrastive embedding models.**
Encode, fine-tune, index, retrieve, rerank, evaluate and serve text-embedding models, and drop them into ML pipelines or AI agents.

[![CI](https://github.com/ZouhairMudakka/clmkit/actions/workflows/ci.yml/badge.svg)](https://github.com/ZouhairMudakka/clmkit/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Status: public alpha](https://img.shields.io/badge/status-public_alpha-orange)

---

Here, **CLM** is shorthand for a contrastively trained text-embedding model, not causal language modeling. Contrastive training pulls matching examples closer together and separates non-matches. The encoder turns texts into vectors; an index compares them to retrieve and rank existing items. Supported presets include [Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-8B) (0.6B / 4B / 8B), E5, BGE and GTE.

Use this workflow when an application repeatedly needs relevant items from a collection: support examples, documents, memories or routing candidates. Store document embeddings, encode each new query, and retrieve a shortlist. Search results can be the output themselves, or supply context for an LLM in a RAG application. Keyword search, metadata constraints and an optional reranker remain useful parts of that workflow; relevance must be evaluated on the intended task.

`clmkit` brings model conventions, retrieval, adaptation, evaluation and application interfaces into one readable codebase:

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
- **Experimental fine-tuning.** InfoNCE with in-batch and hard negatives, false-negative masking, CoSENT, triplet and Matryoshka losses. **GradCache** supports large contrastive batches through smaller forward passes, and **LoRA** is available through `peft`. Epoch coverage, numerical guards and weighted-loss fixes have focused regressions; single-GPU 8B capacity has not been measured. See [validation status](docs/VALIDATION_STATUS.md).
- **Agent building blocks.** `SemanticMemory` with recency decay, `SemanticRouter` for tool/intent routing, and a `ToolKit` that exports tool definitions for **Anthropic** and **OpenAI**-style function calling, plus an **MCP server** for Claude Desktop, Claude Code and IDE agents.
- **ML pipelines.** A scikit-learn `EmbeddingTransformer`, a LangChain `Embeddings` adapter, a JSON/NumPy CLI, and IR metrics (nDCG/MRR/Recall/MAP) with BEIR loading.
- **Serving.** An OpenAI-compatible `/v1/embeddings` endpoint (so existing SDKs work), plus `/v1/search` and `/v1/rerank`. Bearer auth, request limits, and localhost binding by default.
- **Small and pluggable.** The core depends only on `numpy`, and heavy backends are lazy extras. A registry plus Python entry points let third-party packages add encoders, indexes, rerankers and losses. HF loaders require safetensors; use trusted model repositories and complete index snapshots, especially native FAISS files.

## Install

The commands below pin the `v0.1.0a2` public alpha, including the audited fixes,
hybrid retrieval and tested templates. The
[release record](https://github.com/ZouhairMudakka/clmkit/releases/tag/v0.1.0a2)
identifies its exact commit, artifacts and checks; see also
[validation status](docs/VALIDATION_STATUS.md).

```bash
python -m pip install "clmkit @ git+https://github.com/ZouhairMudakka/clmkit@v0.1.0a2"       # NumPy-only core
python -m pip install "clmkit[hf] @ git+https://github.com/ZouhairMudakka/clmkit@v0.1.0a2"   # + torch/transformers
python -m pip install "clmkit[all] @ git+https://github.com/ZouhairMudakka/clmkit@v0.1.0a2"  # + training/serving/integrations
```

Use a fresh virtual environment (`python -m venv .venv`) and activate it before
installing. These commands require Git. The legacy
[v0.1.0a1 prerelease](https://github.com/ZouhairMudakka/clmkit/releases/tag/v0.1.0a1)
predates later security/correctness fixes and templates; its unchanged wheel
and source archives do not contain them. Use the new alpha above.

Extras: `hf`, `train`, `serve`, `mcp`, `faiss`, `yaml`, `sklearn`, `all`, `dev`. Python 3.10+.

HF support starts at torch 2.13, transformers 5.17, safetensors 0.8 and
PEFT 0.21.1 for adapters. Older stacks and legacy `.bin` weights are unsupported.
The `configs/` and `examples/` directories are repository assets: clone this repo
to use the recipe paths below:

```bash
git clone https://github.com/ZouhairMudakka/clmkit.git
cd clmkit
git checkout --detach v0.1.0a2
```

No PyPI release has been published; use the pinned Git URL above.

## Quickstart

This runs with the core install and downloads no model. Hashing is a lexical test
baseline, useful for checking the pipeline; it is not a trained semantic encoder.

```python
from clmkit import Retriever, load_encoder

encoder = load_encoder("hashing")
retriever = Retriever(encoder)
retriever.add(["Paris is the capital of France.", "Bananas are rich in potassium."],
              metadata=[{"topic": "geo"}, {"topic": "food"}])

hits = retriever.search("What is France's capital city?", k=1)
print(hits[0].text, hits[0].score)
```

### Encoding (raw model, no fine-tuning)

Install the `hf` extra first. This example downloads Qwen3-Embedding-0.6B;
the first download needs network access, disk space and sufficient RAM.

```python
encoder = load_encoder("Qwen/Qwen3-Embedding-0.6B")
q = encoder.encode(["What is the capital of China?"], kind="query")            # instruction added for you
d = encoder.encode(["The capital of China is Beijing."], kind="document")      # documents: no instruction
q_code = encoder.encode("sort a list", kind="query", instruction="Given a question, retrieve code snippets")
small = encoder.encode(["..."], dim=256)                                         # Matryoshka truncation
```

Remote servers work the same way (vLLM, TEI, Ollama, OpenAI, or `clmkit serve`):

```python
encoder = load_encoder({"type": "openai", "model": "Qwen/Qwen3-Embedding-8B", "base_url": "http://localhost:8000/v1"})
```

For CLI indexes, provide remote credentials through `OPENAI_API_KEY` or
`--model-arg api_key_env=YOUR_ENV_NAME`; use `HF_TOKEN` or a cached Hugging Face
login for private Hub models. Saved CLI encoder specs reject literal credentials,
custom headers, and URLs containing user info, query strings, or fragments before
loading a model. This also applies when serving with `--allow-writes --save-on-exit`.
Custom plugin configuration remains the caller's responsibility; arbitrary
plugin-specific secrets cannot be inferred reliably.

### Fine-tuning Qwen3-Embedding

Data is JSONL with one example per line:

```json
{"query": "how do I reset my password", "positive": "Open Settings > Security ...", "negatives": ["To change your email ..."]}
```

```bash
clmkit mine  --model Qwen/Qwen3-Embedding-0.6B --train train.jsonl --corpus corpus.txt --output train_hn.jsonl
clmkit train --config configs/qwen3-embedding-8b-lora.yaml --set data.train=train_hn.jsonl --set data.eval=eval.jsonl
clmkit eval  --model runs/qwen3-embedding-8b-lora/final --data eval.jsonl
```

Prepare a separate `eval.jsonl` with held-out pairs before running this sequence.
The 8B recipe is experimental; GPU capacity is unverified. Use the 0.6B config to
start smaller and adjust its data paths in the same way.

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

## Use-case templates

[Tested starter templates](docs/USE_CASES.md) show how to build:

- [Retail customer-service desk](docs/BUSINESS_USE_CASE.md): an end-to-end business
  workflow combining policy evidence, authorized order context, customer
  preferences and team routing into a staff review packet.

- [Support knowledge search](templates/support_search.py): tenant-scoped retrieval,
  source references, and index save/reload for FAQ search or RAG context.
- [Assistant memory](templates/assistant_memory.py): user-scoped facts and recall
  across sessions, with persistence and isolation checks.
- [Support routing](templates/support_routing.py): billing/account/delivery
  selection with an unmatched fallback; no business actions are executed.

They use synthetic data and a NumPy-only hashing baseline, so no model download
or API key is needed. Tests check workflow behavior; semantic accuracy must be
evaluated with your chosen encoder and data. Get the templates from the
`v0.1.0a2` checkout, following the
[setup and customization guide](docs/USE_CASES.md#get-the-templates).

## Retrieval and measured evidence

This alpha adds `BM25Retriever` and `HybridRetriever`, with
shared metadata eligibility, deterministic reciprocal rank fusion, and Python
update/delete APIs. It also adds label-aware effective training batches,
blockwise hard-negative scoring, and strict CLI index compatibility by default.
Hybrid remains in-memory and Python-only.

[Retrieval features and reproducible evidence](docs/RELEVANCE_EVIDENCE.md) documents
the APIs, limitations and bounded Codespaces commands for BANKING77, CLINC150,
SciFact and a direct Sentence Transformers workflow comparison. Read the
[results and costs](docs/RELEVANCE_RESULTS.md) and
[evidence-to-claims ledger](docs/CLAIMS_LEDGER.md) before choosing a retrieval
method. These public tasks do not establish private-customer accuracy, business
ROI or human developer productivity. The
[intent-matching recipes](templates/intent_matching/README.md) show full-gallery
indexing, adaptation/rebuilding and separately calibrated rejection.

On the fixed BANKING77 test, pretrained MiniLM reached **91.79% Hit@1**, versus
**78.96%** for BM25. Three adapted seeds averaged **92.72%**: a gain below the
registered two-point practical target. On SciFact, hybrid had the highest observed
nDCG@10 (**0.6865**) among the four tested methods. CLINC's development-calibrated
5% false-acceptance target did not transfer to test: dense retrieval accepted 7.5% of
unsupported requests. These results support evaluating each stage for the task;
they are MiniLM study results, not Qwen or customer-policy accuracy estimates.

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

Fine-tuned checkpoints save pooling, prompts and output dimension in `clmkit_config.json`.
Reload restores the saved dimension; `output_dim=None` explicitly selects native width.
Checkpoints contain weights/adapters, not resumable optimizer state. Retrieval snapshots
store a configuration fingerprint; supply `encoder_identity` for mutable/custom weights
and use `Retriever.load(..., strict=True)` to reject mismatches. Legacy snapshots warn
that only their old name/dimension check is available.

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
templates/       tested retail service desk, search, memory and routing starters
docs/            DESIGN · TRAINING · AGENTS · GAP_ANALYSIS · AUDIT
```

## Quality

The recorded alpha validation passed all 22 hosted jobs: standard CI, independent contracts, clean installed
wheels, minimum ML dependencies, and real Qwen 0.6B reference/training checks.
The independent review found no remaining P1 blocker in the documented CPU alpha
scope. See [current validation status](docs/VALIDATION_STATUS.md) for exact results,
historical run links, environment advisories and untested hardware. Earlier
coverage figures and Qwen smoke results describe their recorded commits, not every
future change. [Validation harnesses](validation/README.md) preserve the independent checks.
The subsequent [framework review](docs/FRAMEWORK_REVIEW.md) records newer fixes,
their verification, measured overhead and developer adoption limits.

## Documentation

- [Retrieval results and costs](docs/RELEVANCE_RESULTS.md): task comparisons, uncertainty and no-gain outcomes
- [Reproduce the study](docs/RELEVANCE_EVIDENCE.md): pinned public data, models, protocol and bounded CPU commands
- [Claims ledger](docs/CLAIMS_LEDGER.md): which evidence supports each public claim
- [Framework review](docs/FRAMEWORK_REVIEW.md): correctness fixes, measured efficiency, developer fit and prioritized gaps
- [Use cases & tested templates](docs/USE_CASES.md): choose a workflow, run it, customize it, and understand its test scope
- [Retail business workflow](docs/BUSINESS_USE_CASE.md): customer journey, working template, integration points and pilot measures
- [Design & architecture](docs/DESIGN.md): principles, module map, data flow, extension points
- [Training guide](docs/TRAINING.md): data format, losses, GradCache, LoRA, hardware sizing for 8B
- [Agents & integrations](docs/AGENTS.md): memory, routing, tools, MCP, REST, LangChain, sklearn
- [Gap analysis & roadmap](docs/GAP_ANALYSIS.md): what's covered, what isn't, what's next
- [Audit report](docs/AUDIT.md): tests, static analysis, security review, real-model validation
- [Audit remediation](docs/REMEDIATION.md): corrected contracts, acceptance evidence and remaining gaps

## Contributing

Issues and PRs are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). Security reports go through [SECURITY.md](SECURITY.md).

## License

[Apache-2.0](LICENSE). Model weights are licensed separately by their authors (Qwen3-Embedding and Qwen3-Reranker are Apache-2.0).
