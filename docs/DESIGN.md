# Design & architecture

## Scope: what "CLM" means here

**Contrastive Language Model (CLM)**: a language model trained with a contrastive objective to produce text representations. Paired texts (query ↔ relevant passage, sentence ↔ paraphrase) are pulled together and unrelated ones pushed apart. In practice this means **bi-encoder embedding models** such as Qwen3-Embedding, E5, BGE and GTE, often paired with a **reranker** (cross-encoder or LLM-as-judge) for second-stage precision.

The name is also used for two other things that are **out of scope** for v0.1 (see the [gap analysis](GAP_ANALYSIS.md)):

- *Contrastive decoding*, an inference-time trick that contrasts an expert and an amateur generative LM. It's a different problem: it concerns generation, not representation.
- *CLIP-style multimodal contrastive models* (text ↔ image). The `Encoder` interface has no text-only assumptions beyond its input type, so a multimodal encoder is a natural extension (for example Qwen3-VL-Embedding).

## Design principles

1. **Skeleton, not monolith.** Every major piece is an interface with one or two solid implementations: `Encoder`, `VectorIndex`, `Reranker`, loss protocol, evaluator callable. Users swap pieces without forking.
2. **Numpy-only core; heavy backends are lazy.** `import clmkit` never imports torch. `transformers`, `peft`, `fastapi`, `faiss` and `mcp` load only when the component that needs them is built. CI enforces this.
3. **Correct by default for real models.** Pooling, padding side, prompt templates and EOS handling are *model-specific* and easy to get silently wrong, so they live in presets and are verified against reference implementations.
4. **Readable training loop.** No trainer framework. `ContrastiveTrainer.train()` reads top to bottom, and GradCache is ~40 lines.
5. **Safe persistence and serving.** No pickle (NumPy `allow_pickle=False` + JSON), `yaml.safe_load`, `trust_remote_code=False`, localhost binding, optional auth, request limits, writes off by default.
6. **Framework-agnostic agent integration.** Tools are plain JSON Schema + Python callables, exported to Anthropic/OpenAI formats and MCP. There's no dependency on any agent framework.

## Module map

```mermaid
flowchart LR
  subgraph core [numpy-only core]
    reg[registry.py<br/>Registry + entry points]
    enc[encoders/base.py<br/>Encoder ABC]
    hash[encoders/hashing.py]
    oai[encoders/openai_compat.py]
    idx[index/<br/>VectorIndex, NumpyIndex]
    ret[retrieval.py<br/>Retriever]
    data[data.py<br/>examples, batching, mining]
    ev[eval/<br/>metrics, evaluators, BEIR]
    ag[agents/<br/>memory, router, tools]
  end
  subgraph torch [torch extras]
    hf[encoders/hf.py<br/>HFEncoder + presets]
    pool[pooling.py]
    loss[losses.py]
    tr[training/<br/>ContrastiveTrainer]
    rr[rerank.py<br/>LLM yes/no, cross-encoder]
  end
  subgraph edge [integration edge]
    cli[cli.py]
    api[serve/app.py<br/>FastAPI]
    mcp[serve/mcp_server.py]
    skl[integrations/sklearn.py]
    lc[integrations/langchain.py]
  end
  enc --> hash & oai & hf
  hf --> pool
  tr --> loss & hf & data
  ret --> enc & idx & rr
  ag --> ret
  ev --> idx
  api & mcp & cli --> ret & ag
  skl & lc --> enc
  reg -.lazy.-> hf & idx & rr & loss
```

## Key abstractions

### `Encoder` (`encoders/base.py`)

Subclasses implement two things: `native_dim` and `_encode(formatted_texts, batch_size) -> ndarray`. The base class handles everything else, so every backend behaves identically:

- **asymmetric prompts**: `query_template` / `document_template` with `{instruction}` / `{text}`; per-call or per-example instructions;
- **Matryoshka truncation** (`dim=` per call or `output_dim` per encoder) followed by **L2 normalisation**, so dot product equals cosine;
- input validation, `str` vs `list[str]`, empty input, shape checks;
- `fingerprint()`, stored with indexes so querying with a different model raises a warning or error.

Implementations: `HFEncoder` (transformers), `OpenAICompatibleEncoder` (stdlib HTTP), and `HashingEncoder` (deterministic feature hashing, a zero-download baseline for tests, CI and demos).

### Presets (`encoders/presets.py`)

A `ModelPreset` is chosen by glob match on the model id, or by a `clmkit_config.json` next to a checkpoint. The Qwen3 preset encodes a subtle fact: the tokenizer's `eos_token` is `<|im_end|>`, but embeddings are pooled from `<|endoftext|>`. The encoder guarantees that token is last, without duplicating it when the tokenizer already appends it.

### `VectorIndex` (`index/`)

The index stores `(id, vector)` pairs only; text and metadata live in the `Retriever`. That keeps indexes swappable, since a vector DB adapter only needs `add/search/remove/save/load`. `NumpyIndex` is exact and chunked with argpartition top-k. `FaissIndex` supports flat and HNSW. Filtering is expressed as `allowed_ids`.

### `Retriever` (`retrieval.py`)

Encoder + index + document store, with an optional reranker. Metadata filters can be an equality dict or a predicate. Reranking over-fetches `rerank_candidates` (default `4k`) and keeps the first-stage score in `metadata["retrieval_score"]`. Persistence is a directory (`index/`, `documents.jsonl`, `retriever.json`).

### Training (`losses.py`, `training/`)

- All losses share `loss(query, positive, negatives=None, scores=None)`, so the trainer is loss-agnostic. `MatryoshkaLoss` wraps any of them.
- `info_nce` builds candidates from in-batch positives plus **all** hard negatives in the batch. It optionally masks suspected false negatives (candidates scoring more than `margin` above the positive) and optionally adds the symmetric document→query term.
- `iter_batches` keeps duplicate queries and positives out of the same batch, because a duplicate positive becomes a guaranteed false negative.
- **GradCache**: (1) embed all chunks without a graph, saving RNG state; (2) compute the full-batch loss on detached embeddings to get ∂L/∂embedding; (3) re-embed each chunk with a graph and back-propagate the cached gradient. Tests assert the resulting parameter gradients equal full-batch gradients.
- LoRA uses `peft`. Checkpoints can be adapter-only (small; `HFEncoder` auto-loads base + adapter) or merged (plain checkpoint).

### Agents (`agents/`)

- `SemanticMemory`: a `Retriever` plus a creation timestamp. Scores decay as `cos * 0.5 ** (age / half_life)`. It supports `forget` and `forget_older_than`.
- `SemanticRouter`: routes defined by example utterances and/or a description, aggregated by max, mean or centroid. Global and per-route thresholds let you abstain (`None`) and fall back to an LLM.
- `ToolKit`: JSON Schema tools with a minimal validator (types, required, bounds, `maxLength`, enum, no extra args). `ToolError` messages are safe to show to the model.

### Registry and plugins (`registry.py`)

Four registries: `ENCODERS`, `INDEXES`, `RERANKERS` and `LOSSES`. Built-ins that need optional deps are registered **lazily by dotted path**. Third-party packages register via entry points:

```toml
[project.entry-points."clmkit.encoders"]
my-encoder = "my_pkg.encoders:MyEncoder"
```

## Data flow: a search request

1. `Retriever.search(q)` calls `encoder.encode(q, kind="query", instruction=…)`. The template is applied, then tokenize → forward → pool → truncate → normalise.
2. `index.search(qvec, pool, allowed_ids=filter(metadata))` returns top-k `(id, score)` pairs.
3. The optional `reranker.rerank(q, hits)` scores pairs jointly and re-sorts them.
4. `SearchHit(id, score, text, metadata)` objects are returned. The REST API, MCP tools and agent tools are thin shells over this.

## Decisions & trade-offs

| Decision | Why | Cost |
|---|---|---|
| numpy-only core, lazy torch | usable in agents and services without a 2 GB dependency; fast imports | a registry indirection |
| Own trainer instead of HF `Trainer` / sentence-transformers | GradCache + in-batch-negative semantics are explicit; ~350 readable lines | no DDP/FSDP yet (see gap analysis) |
| Exact NumPy index by default | zero deps, exact | O(N) per query; memory depends on dimensions and query chunking (1M × 4096 float32 vectors alone need 16.384 GB) |
| NumPy JSON + `.npy` with pickle disabled | avoids pickle deserialization for this backend | native FAISS files still require trusted provenance |
| stdlib HTTP client for remote encoders | no `requests`/`httpx` dependency | fewer conveniences (we implement retry/backoff) |
| Refuse HTTP redirects in the remote client | `urllib` forwards `Authorization` across hosts | a server behind a redirect needs its final URL |
| Global lock around the model in the REST server | torch modules aren't guaranteed thread-safe | one forward at a time per process; multiple workers need an explicit shared-state/durability design |
| fp32 on CPU for `dtype="auto"` | transformers v5 defaults to the checkpoint dtype (bf16), which is slow and less precise on CPU | more RAM on CPU |
