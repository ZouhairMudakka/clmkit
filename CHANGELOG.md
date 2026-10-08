# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0a2] - 2026-10-08

Adds a reproducible public-data retrieval study and the framework features needed
to compare keyword, pretrained, adapted, hybrid and reranked retrieval. All 13
registered held-out comparisons completed; independent formulas verify their
ranking and rejection metrics. Results include modest adaptation gains below the
registered practical target and calibration limits, rather than a universal
quality claim. See [results and costs](docs/RELEVANCE_RESULTS.md).

- Add NumPy-only BM25 and exact dense/BM25 reciprocal rank fusion, shared metadata
  eligibility, deterministic ties, and staged document mutation APIs. Hybrid is
  currently Python-only and in-memory.
- Add relevance labels and effective-batch label exclusion for ordinary and
  GradCache training; explicit negatives remain unsupported in this mode.
- Score hard-negative mining in configurable query/corpus blocks, with optional
  same-label exclusions; full corpus embeddings still remain resident.
- Default CLI index loading to strict compatibility, expose immutable encoder
  identities and an explicit mismatch override, and reject incompatible writes.
- Add pinned public-data preparation, development/test seals, bounded relevance
  runners and a direct Sentence Transformers workflow comparison. Document
  reproduction in `docs/RELEVANCE_EVIDENCE.md` and task-specific outcome/claim
  limits in `docs/RELEVANCE_RESULTS.md` and `docs/CLAIMS_LEDGER.md`.
- Add strict full-gallery intent-matching recipes for pretrained/adapted BANKING77
  models and separately calibrated CLINC routing, plus a bounded authenticated
  loopback serving benchmark. Real CPU checks passed all three full-gallery
  build/query recipes, direct Sentence Transformers parity and 192 measured
  HTTP requests at concurrency 1/4/8/16; these are bounded diagnostics, not an SLA.

- Audit framework correctness, measured overhead and developer workflows; record
  adoption limits and priorities in `docs/FRAMEWORK_REVIEW.md`.
- Accelerate duplicate-heavy training batching while preserving order and coverage;
  skip embedding calls for empty searches and invalid document IDs.
- Preserve CLI evaluation instructions and batch sizes, refresh changed sklearn
  encoders, and make default configuration work with all three training losses.
- Handle large finite-vector normalization, preserve adapter base revisions and
  reject checkpoint directory reuse across incompatible formats.
- Improve unpublished-package installation hints and mined-data training recipes.
- Reject literal credentials in persisted CLI encoder specs; keep environment
  references and cached authentication supported. Existing snapshots are unchanged.
- Add a retail service-desk business template that prepares human-reviewed case
  packets from scoped policies, order facts, preferences and request routing.
- Add documented support search, scoped assistant memory and request routing
  templates, with deterministic behavior tests and installed-wheel CI coverage.

## [0.1.0a1] - 2026-10-07

First GitHub prerelease, covering the documented CPU alpha scope. NumPy-only
quickstart and release artifacts are checked in isolated installations. GPU,
4B/8B capacity and production reliability remain unverified; see
[validation status](docs/VALIDATION_STATUS.md).

- Correct duplicate-aware epoch coverage, weighted Matryoshka pairing and per-query evaluation instructions.
- Reject non-finite training losses/gradients before optimizer mutation; validate remote vector indices and values.
- Restore saved output dimensions and memory settings; validate stored IDs and version encoder fingerprints.
- Require safetensors for HF models and adapters; raise ML dependency floors and test the minimum stack in CI.
- Document bounded memory-decay ranking, single-writer snapshot trust and unverified GPU/large-model paths.
- Restrict source archives to explicit root-relative package/test files and verify their contents before distribution.

## [0.1.0] - 2026-09-30

Initial implementation (historical entry; no tagged or PyPI release was published on this date).

### Added
- `Encoder` interface with asymmetric prompt templates, Matryoshka truncation and normalisation.
- `HFEncoder` with presets for Qwen3-Embedding (0.6B/4B/8B), E5, BGE, GTE and MiniLM/MPNet. Qwen3 behaviour is verified against the model card.
- `OpenAICompatibleEncoder` (vLLM, TEI, Ollama, OpenAI) with retries and redirect refusal; `HashingEncoder` zero-download baseline.
- `NumpyIndex` (exact) and `FaissIndex` (flat/HNSW), with pickle-free persistence.
- `Retriever` with metadata filters, reranking and persistence; `chunk_text`.
- Rerankers: `LLMYesNoReranker` (Qwen3-Reranker, verified vs the reference), `CrossEncoderReranker`, `EncoderReranker`.
- Losses: InfoNCE (hard negatives, false-negative masking, symmetric), CoSENT, triplet, Matryoshka.
- `ContrastiveTrainer` with GradCache, LoRA (peft), bf16 autocast, gradient checkpointing, eval and best-checkpoint tracking; YAML/JSON config runner.
- Data utilities: JSONL I/O, duplicate-free batching, positive-aware hard-negative mining.
- Evaluation: nDCG/MRR/Recall/Precision/MAP/Hit, STS correlations, `RetrievalEvaluator`, BEIR loader.
- Agents: `SemanticMemory`, `SemanticRouter`, `ToolKit` with Anthropic/OpenAI tool formats.
- Serving: OpenAI-compatible REST API (FastAPI) and an MCP server (SDK 1.x and 2.x).
- Integrations: scikit-learn `EmbeddingTransformer`, LangChain `ClmkitEmbeddings`.
- CLI: `info`, `encode`, `index`, `search`, `mine`, `train`, `eval`, `serve`, `mcp`.
- Docs: design, training, agents, gap analysis and audit report; 7 examples; CI.
