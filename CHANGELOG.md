# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## Unreleased — audit remediation (2026-10-07)

- Correct duplicate-aware epoch coverage, weighted Matryoshka pairing and per-query evaluation instructions.
- Reject non-finite training losses/gradients before optimizer mutation; validate remote vector indices and values.
- Restore saved output dimensions and memory settings; validate stored IDs and version encoder fingerprints.
- Require safetensors for HF models and adapters; raise ML dependency floors and test the minimum stack in CI.
- Document bounded memory-decay ranking, single-writer snapshot trust and unverified GPU/large-model paths.

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
