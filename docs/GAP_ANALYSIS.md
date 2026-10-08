# Gap analysis & roadmap

> Historical builder assessment. The status labels below are not an independent
> release verdict. Read [current validation status](VALIDATION_STATUS.md) and the
> [updated remaining gaps](REMEDIATION.md#remaining-gaps) first.
> The later [framework review](FRAMEWORK_REVIEW.md) assesses developer adoption,
> measured overhead and current priorities. Historical scale estimates below are
> not capacity guarantees.

*Assessed against v0.1.0 (2026-09-30).* Legend: ✅ done and tested · 🟡 partial / workaround exists · ❌ not implemented.

## Current milestone update — 8 October 2026

The historical matrices and roadmap below retain the 30 September assessment;
their missing-feature labels are not the current implementation inventory.
The relevance-study source freeze
`e61cc76d6cf52f25422335d42eaf18dde44e3067` now implements:

- **BM25 and dense/BM25 reciprocal-rank fusion**, with shared metadata eligibility.
  These are Python-only, in-memory, single-writer APIs; learned sparse models,
  hybrid persistence and CLI/REST integration remain gaps.
- **Blockwise hard-negative scoring**, replacing the full query-by-corpus score
  matrix. Corpus embeddings and other model/merge buffers still remain resident.
- **Label-aware effective training batches**, including GradCache chunks, with
  explicit negatives disabled in this mode; this is not a multi-positive loss.
- **Stricter index provenance**, including strict CLI loading, declared immutable
  encoder identities and rejection of incompatible or legacy-index writes.
  Python loading still requires `strict=True` for strict checks; identities do not
  automatically detect arbitrary weight changes or make snapshots transactional.

See [APIs and reproduction](RELEVANCE_EVIDENCE.md) and the
[implementation review](RELEVANCE_REVIEW.md) for tests and remaining limits.
The [public study and application checks](RELEVANCE_RESULTS.md) now provide
held-out comparisons, measured costs, full-gallery recipes and a bounded HTTP
diagnostic. Adaptation improved BANKING77 modestly but missed the registered
practical target; hybrid/reranking gains depended on the task, and CLINC's 5%
development false-acceptance target did not transfer to test. These are public
proxy results, not customer-policy accuracy. The new alpha's exact candidate
checks are linked from its release record.

Remaining priorities are representative customer evaluation and calibration,
durable storage/recovery, hybrid CLI/REST integration, measured serving batching,
API reference documentation and human usability evidence. PyPI, optimizer resume,
multi-GPU work, Qwen 4B/8B execution and Arabic–English quality remain open.
Historical counts and scale estimates below do not establish those capabilities.

## 1. Requirements from the brief

The brief called for a skeleton framework for a **Contrastive Language Model**, open-sourced, usable **in an AI/ML flow, in an AI agent, or elsewhere**, perhaps with a **Qwen 8B** model, **fine-tuned or used raw**.

| Requirement | Status | Evidence / notes |
|---|---|---|
| Skeleton framework, pluggable | ✅ | ABCs for encoder, index, reranker; loss protocol; registries + entry-point plugins (tested) |
| Qwen ~8B contrastive model | ✅ | `Qwen/Qwen3-Embedding-8B` preset (last-token pooling, left padding, instruction template, `<\|endoftext\|>`, MRL 32–4096). The 0.6B sibling matches the model card to 1.5e-7. The 4B/8B presets share the identical code path and settings but **were not executed locally** (hardware) |
| Used raw | ✅ | `load_encoder("Qwen/…")`; also remote vLLM/TEI/Ollama via the OpenAI-compatible client |
| Fine-tuned | ✅ | InfoNCE (+hard negatives, false-negative masking, symmetric), CoSENT, triplet, Matryoshka; GradCache; LoRA; hard-negative mining; YAML configs; validated on real Qwen3-0.6B |
| AI/ML flow integration | ✅ | Python API, scikit-learn transformer, LangChain adapter, CLI (JSONL/`.npy`), REST |
| AI agent integration | ✅ | memory, router, tool specs (Anthropic/OpenAI), MCP server (SDK 1.x and 2.x), REST |
| "Different uses" | 🟡 | retrieval, RAG, reranking, routing, memory, classification and STS are covered; **clustering, deduplication and zero-shot classification helpers are not** (easy with the embeddings, but no first-class API) |
| Open source on GitHub | ✅ | Apache-2.0, CI, contributing/security docs, issue templates |
| Gap analysis, testing, auditing | ✅ | this document; 104 tests; [AUDIT.md](AUDIT.md) |

## 2. Capability matrix

### Modelling & inference
| Capability | Status | Gap / next step |
|---|---|---|
| HF encoders with pooling/prompt presets | ✅ | add more presets as models ship (e.g. newer Qwen/Gemma/NV embedders) |
| Instruction-aware queries, per-example instructions | ✅ | |
| Matryoshka truncation at inference | ✅ | |
| Remote encoders (OpenAI-compatible) | ✅ | no async client; no request concurrency |
| Rerankers: LLM yes/no (Qwen3-Reranker), cross-encoder, bi-encoder | ✅ | no listwise/LLM-generative rerankers |
| Quantised inference (int8/4-bit, ONNX, TensorRT) | ❌ | pass `model_kwargs` for bitsandbytes today; no first-class support or tests |
| Sparse / hybrid retrieval (BM25, SPLADE, BGE-M3 sparse) | ❌ | high value for RAG; add a `SparseEncoder` + score fusion (RRF) |
| Multi-vector (ColBERT-style late interaction) | ❌ | |
| Multimodal (CLIP-style, Qwen3-VL-Embedding) | ❌ | the `Encoder` interface fits; needs image inputs in `encode` |
| Contrastive *decoding* (generation-time) | ❌ | out of scope (different technique); could be a separate module |
| Long-document handling | 🟡 | `chunk_text` (char-based); no token-aware or semantic chunking |

### Training
| Capability | Status | Gap / next step |
|---|---|---|
| InfoNCE w/ in-batch + hard negatives, FN masking | ✅ | Qwen3 paper also adds query-query / doc-doc negatives: not implemented |
| CoSENT, triplet, Matryoshka | ✅ | no AnglE, no distillation losses (MarginMSE, KL from a reranker teacher) |
| GradCache | ✅ | gradient-equivalence test on tiny model |
| LoRA (peft), merge on save | ✅ | QLoRA (4-bit base) not wired/tested |
| bf16 autocast, gradient checkpointing | ✅ | **fp16 + GradScaler not supported** (rejected with a clear error) |
| Multi-GPU (DDP/FSDP/DeepSpeed) + cross-device negative gathering | ❌ | **top training gap for 8B full fine-tunes**; plan: `accelerate` + `all_gather` embeddings before the loss |
| Resume from checkpoint (optimizer/scheduler state) | ❌ | only weights/adapters are saved |
| Experiment tracking | 🟡 | `callbacks=[...]` hook; no built-in W&B/MLflow |
| Synthetic data generation (LLM-generated queries) | ❌ | common for domain adaptation; would pair with the mining CLI |
| Curriculum / task-homogeneous batching | ❌ | |

### Retrieval, storage & serving
| Capability | Status | Gap / next step |
|---|---|---|
| Exact NumPy index, FAISS flat/HNSW | ✅ | FAISS filtering is over-fetch + post-filter (may return < k under very selective filters) |
| Vector DB adapters (pgvector, Qdrant, Milvus, Weaviate, LanceDB) | ❌ | interface is ready (`VectorIndex`); adapters are the obvious community contributions |
| Incremental persistence / WAL | ❌ | `save()` rewrites the directory; fine for ≤ millions of docs |
| Metadata filtering | 🟡 | equality dict or Python predicate, evaluated in Python (O(N)) |
| REST API (OpenAI-compatible) | ✅ | global model lock → one forward at a time per process; no dynamic batching queue; no streaming ingest |
| Auth / limits | 🟡 | bearer token + size limits; **no rate limiting, no multi-key/tenant auth** (use a gateway) |
| Observability (Prometheus metrics, tracing) | ❌ | |
| MCP server | ✅ | stdio transport only; no streamable-HTTP transport yet |
| Docker image / Helm chart | ❌ | |

### Evaluation
| Capability | Status | Gap / next step |
|---|---|---|
| nDCG/MRR/Recall/Precision/MAP/Hit, STS Spearman/Pearson | ✅ | verified against hand computations and SciPy |
| BEIR-format loader | ✅ | |
| MTEB runner / leaderboard-comparable eval | ❌ | wrap `mteb` so users can report standard numbers |
| Latency / throughput benchmarks | ❌ | |

### Engineering
| Item | Status | Notes |
|---|---|---|
| Tests | ✅ | 104 total (102 pass by default + 2 opt-in real-model); 95% branch coverage |
| Static analysis | ✅ | ruff (incl. bandit rules), mypy, bandit, pip-audit in CI |
| CI matrix | ✅ | Linux/Windows/macOS × py3.10/3.13 core-only; Linux full with CPU torch |
| GPU CI | ❌ | real-model tests are opt-in (`CLMKIT_TEST_MODEL`, `CLMKIT_TEST_RERANKER`) |
| PyPI release | ❌ | install from GitHub for now; add a trusted-publishing workflow |
| Docs site (mkdocs + API reference) | ❌ | Markdown docs only |
| Async API | ❌ | agents built on asyncio must use a thread pool |

## 3. Positioning vs the ecosystem

| Project | Strength | How clmkit differs |
|---|---|---|
| **sentence-transformers** | the most complete training library (dozens of losses, multi-GPU, huge model zoo) | clmkit is smaller and agent-first: memory, routing, tool specs, MCP and serving are built in, and GradCache/false-negative masking are readable. For large-scale training features, sentence-transformers is ahead |
| **FlagEmbedding (BGE)** | reference training recipes for BGE models | model-family specific; clmkit is model-agnostic with presets |
| **ms-swift** | Qwen's official fine-tuning toolkit (supports Qwen3-Embedding) | heavier, broad LLM toolkit; clmkit focuses on the contrastive loop only |
| **txtai / Haystack / LlamaIndex** | full pipelines and orchestration | clmkit is a component you can plug into them (LangChain adapter today) |
| **semantic-router** | routing layer | clmkit's router is one module sharing the same encoder/eval stack |
| **vLLM / TEI** | high-throughput embedding serving | complementary: clmkit uses them as remote encoders |

## 4. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Upstream API churn (transformers v5, mcp 2.x already broke APIs once) | breakage on fresh installs | version-tolerant code paths (tested on mcp 1.28.1 + 2.2.0; transformers `dtype`/`torch_dtype`), CI on latest deps; consider upper bounds before 1.0 |
| Model-convention mistakes (pooling/EOS/prompt) | silently worse embeddings | presets + reference-equivalence tests; opt-in real-model tests |
| False negatives in mined data | degraded fine-tunes | positive-aware mining, FN masking, duplicate-free batching |
| Prompt injection via retrieved content | agent misbehaviour | documented; write tools off by default |

## 5. Roadmap (prioritised)

**P0: next release**
1. Multi-GPU training via `accelerate` with cross-device negative gathering, plus resume-from-checkpoint.
2. Hybrid retrieval: BM25 + dense with reciprocal-rank fusion.
3. PyPI trusted publishing + versioned docs.

**P1**
4. Vector DB adapters (pgvector, Qdrant) behind `VectorIndex`.
5. Distillation losses (reranker → embedder, MarginMSE/KL) and QLoRA.
6. MTEB runner and a small benchmark suite; GPU CI job for the real-model tests.
7. Async encoder/retriever API; dynamic batching in the server; Prometheus metrics.

**P2**
8. Multimodal encoder (Qwen3-VL-Embedding), multi-vector retrieval.
9. Clustering / dedup / zero-shot classification helpers.
10. Synthetic training-data generation with an LLM, and task-homogeneous batching.
11. Docker image; MCP streamable-HTTP transport.
