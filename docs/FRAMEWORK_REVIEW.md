# Framework efficiency and developer experience review

Review started 7 October 2026 against `19421a5`, after the public alpha and tested
use-case templates. CLM means contrastive text embeddings in this review.

## Audit plan and acceptance criteria

1. **Establish the baseline.** Run the existing no-download CPU suite, preserve
   the prior real-model evidence, and record the checked commit and environment.
2. **Review correctness and failure paths.** Inspect model/data contracts,
   retrieval/filtering, persistence, training, serving and integrations. Turn
   reproducible supported-path defects into regression tests before fixes.
3. **Measure framework overhead.** Use bounded, repeatable CPU workloads for
   batching, indexing and retrieval. Separate encoder costs from framework costs;
   publish workload sizes and before/after measurements for optimizations.
4. **Exercise developer journeys.** Check installation, first query, remote/local
   model selection, persisted indexes, agent integration and fine-tuning against
   the public API and documentation. Assess boilerplate and error messages.
5. **Prioritize gaps.** Distinguish alpha defects from missing scale/production
   features. Evaluate fitness for prototyping, application developers, researchers
   and non-developer users without inferring quality from hashing tests.
6. **Verify and review.** Run targeted regression tests, static checks and hosted
   installed-package/platform checks. Obtain an independent final review after
   testing, then record results and remaining limitations here.

Acceptance: no newly demonstrated supported-path defect is silently dismissed;
fixes preserve tested behavior; performance claims include reproducible evidence;
usability recommendations distinguish implemented capabilities from proposed
work. Any unresolved issue states its practical impact and workaround.

## Scope and resource limits

Use the retained development environment and no-download fixtures locally. Avoid
recreating large model caches after cleanup. No production systems or customer
data are used. GPU/CUDA, 4B/8B capacity, live vendor interoperability, held-out
semantic quality and commercial time savings require separate evidence. Hosted
CPU checks may validate clean installations; that does not establish GPU support.

## Findings and verification

The framework is a useful **developer-oriented alpha for text embeddings**. Its
small Python surface, NumPy-only core, optional backends and executable templates
reduce integration work for prototypes and controlled pilots. This is an
architectural assessment backed by workflow tests, not a measured productivity
study. It is not yet a durable vector service or a turnkey business application.

### Findings corrected in this review

| Finding | Impact before correction | Correction and regression evidence |
|---|---|---|
| CLI evaluation overrides record instructions | Evaluated a different query task from the data; could mislead model comparison. | Preserve each instruction unless explicitly overridden; capture actual encoder inputs. |
| CLI evaluation/mining ignore batch size | User memory controls do not take effect. | Forward the argument to both workflows; verify actual encoding batches. |
| sklearn caches an obsolete encoder | `set_params` reports new settings while output still uses old dimensions. | Invalidate cached encoder after encoder/spec parameter changes; verify transformed output. |
| Default training kwargs belong only to InfoNCE | Selecting CoSENT or triplet raises a constructor error. | Delegate defaults to each loss; run a tiny CPU training step for each objective. InfoNCE retains temperature 0.05. |
| Large finite float32 vectors normalize to zero | Overflow silently corrupts similarity scores. | Recompute overflowing norms in float64; test independent normalized-vector expectations and index retrieval. |
| Reusing one directory for adapter and merged checkpoints | Stale files make a newly saved checkpoint unloadable. | Reject incompatible formats before mutation; verify existing bytes/model and separate-directory reload parity. |
| Adapter-only saves lose base revision | Reload can use a different base checkpoint. | Save resolved base commit when available, otherwise requested revision; restore unless explicitly overridden. Local/mock tests verify the loader argument. |
| CLI reload ignores supplied model kwargs without a model | User intent is silently lost. | Reject the ambiguous invocation with guidance. |
| Empty retrieval and invalid inserts still encode | Waste local inference or remote requests for a known empty/failing operation. | Check scope/IDs first; counted-encoder tests prove zero model calls while retaining input validation. |
| Duplicate-aware batching repeatedly scans deferred examples | Large duplicate groups create substantial Python overhead. | Earliest-compatible-batch placement with successor maps; compare exact order against an independent stable-greedy oracle, including shuffling, partial batches and early yielding. |
| Installation/training guidance has broken handoffs | Unpublished PyPI command fails; mined data is not selected by the next command. | Checkout/Git installation guidance and explicit training/evaluation data overrides; mark 8B configuration unverified on hardware. |

The reproducible failures above are P2 correctness/usability or performance
findings. Priorities describe practical impact, not a count of failing assertions.

**Credential persistence:** CLI-created indexes previously stored literal model
credentials in their reconstructible `encoder_spec`. Treat accidental credential
disclosure through sharing an index as a P1 concern. Index creation and writable
server save-on-exit now reject built-in literal keys/tokens, nonempty custom
headers and credential-bearing base URLs before model loading. Environment-variable
references and Hugging Face cached authentication remain supported. Regression
tests verify rejection before inference, no secret in output/error, and persistence
of the environment-variable name without its value. Custom plugin options remain
the caller's responsibility; this guard is not a general-purpose secret scanner.
Existing snapshots are not rewritten or scrubbed by this change. Check previously
shared snapshots and rotate any exposed credentials. No actual credential exposure
was observed in this audit; the reproduction used a synthetic token.

### Measured efficiency

Windows 11, Python 3.13.1, NumPy 2.4.6; single-thread BLAS settings. Five alternating
before/after measurements in the same process against baseline `19421a5`, with
batch size 64 and no shuffle. Both implementations consume every resulting batch.

| Synthetic batching workload | Before median | After median | Measured speedup |
|---|---:|---:|---:|
| 4,000 examples sharing one positive | 1.546 s | 9.84 ms | 157x |
| 20,000 examples, 10% sharing one positive | 307 ms | 25.8 ms | 11.9x |
| 20,000 distinct examples | 11.1 ms | 9.29 ms | 1.19x |

These numbers measure batch construction, not model training speed. Gains depend
on duplicate structure; they are not a universal complexity guarantee. The new
maps trade memory for fewer scans: an earlier separately traced 4,000-shared
fixture increased incremental allocations from approximately 0.23 MB to 1.05 MB.
This is `tracemalloc` allocation evidence, excluding imports, fixtures and total
process RSS. The benchmark stays below a 128 MiB traced-allocation guard.

The same bounded harness exercises exact search (2,000 vectors x 64 dimensions,
32 queries), selective filtering, and an offline hashing evaluator. These are
sanity workloads, not evidence of production scale or semantic quality. Raw paired
samples are in [the evidence file](validation-evidence/2026-10-07-framework/performance-paired.json).
Reproduce with [the benchmark script](../validation/benchmark_framework.py); its
header includes the baseline extraction and paired-comparison commands.

### Where costs still grow

- **In-memory exact search:** float32 vectors alone require approximately
  `4 * documents * dimensions` bytes, before text, metadata and search temporaries.
  For example, one million 1,024-dimensional vectors require about 4.1 GB decimal
  for vectors alone. NumPy searches allocate query-chunk-by-corpus score arrays;
  selective filters also copy the selected vectors. Choose bounded query chunks
  and measure resident memory on the intended corpus.
- **Incremental ingestion:** NumPy index additions concatenate the vector matrix.
  Many single-document additions repeatedly copy prior data. Bulk additions are
  the supported mitigation; durable incremental ingestion needs another backend.
- **Mining:** hard-negative mining retains all embeddings and the full
  query-by-corpus similarity/sorting arrays. Encoding batch size does not cap those
  arrays. Corpus-blocked scoring with bounded top-k is the next major optimization.
- **Filtering and serving:** metadata predicates scan documents in Python. FAISS
  filtering uses bounded over-fetch and may return fewer than `k` eligible hits.
  REST serializes model forwards; remote requests are synchronous. High concurrency
  requires load tests and a queue/batching or serving architecture.
- **Persistence:** snapshots remain trusted, nontransactional, single-writer
  directories. They provide no WAL, crash-atomic publication or multi-process
  coordination. Checkpoint saves contain weights/adapters, not resumable optimizer
  state.

The raw-vector estimate also agrees with FAISS's documented flat-index storage;
its HNSW index adds graph memory. This does not establish clmkit end-to-end memory
capacity. [FAISS index documentation](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes).

## Efficiency for developers and users

| Intended user/workflow | Assessment | What adoption still requires |
|---|---|---|
| Python developer, embedding search or RAG prototype | Good fit: encoder, index, retrieval and reranking share a small interface; offline quickstart needs no model download. | Pick a real encoder, preserve asymmetric prompts, evaluate held-out queries and choose storage for actual data volume. |
| Agent developer, scoped memory or routing | Good starting point: tested tenant/user wrappers, tool schemas, MCP and abstention templates. | Authenticate scope upstream, calibrate thresholds, handle retention and concurrent writes. Templates do not provide an authorization system. |
| Researcher, small contrastive fine-tuning experiment | Useful: composable losses, LoRA, GradCache, metrics and config runner. | Dataset design and leakage checks; independent quality baseline; GPU evidence if relevant. No optimizer resume or distributed training. |
| Business developer, support/service-desk pilot | Useful template: tested policy citations, exact order ownership, routing and human review. | Replace synthetic fixtures, integrate real systems, measure retrieval accuracy and task outcomes. No commercial ROI claim is established. |
| Nontechnical end user | Limited direct fit. This is a Python library and CLI. | A developer must provide the application UI, connectors and operational controls. |
| Large-scale production operator | Requires significant additional infrastructure. | Durable backend, concurrent-write design, observability, load/failure testing and deployment-specific security. |

Strengths are the optional dependency boundaries, pluggable contracts, shared
query/document conventions and small runnable examples. All eight applicable
offline examples/templates ran successfully in this review. Their synthetic
tests establish workflow behavior; real semantic effectiveness remains a property
of the chosen model, domain data and evaluation protocol.

Remaining integration friction is concrete:

- A Python-created snapshot does not automatically contain the reconstructible
  CLI encoder spec. CLI search/serve/MCP needs `--model` and relevant options.
- Strict fingerprint validation and custom `encoder_identity` are Python API
  controls; the CLI currently warns on mismatch and has no strict identity flag.
- CLI index creation exposes NumPy/FAISS, while custom registry indexes require
  Python integration. Shared `--batch-size` parsing in search/serve/MCP is not a
  runtime service-batching control.
- There is no PyPI release or generated API-reference site. The GitHub alpha
  artifact is immutable; fixes documented here are post-alpha repository changes.

For teams selecting a framework, compare this compact integration layer with the
requirements of their actual training workflow. Sentence Transformers documents
a broader established training workflow around models, datasets, losses, training
arguments, evaluators and a trainer. This review does not benchmark productivity
or model quality against that library. [Official training overview](https://www.sbert.net/docs/sentence_transformer/training_overview.html).

## Prioritized remaining work

1. **Before a real pilot:** use held-out domain queries, calibrate routing/no-match
   thresholds, validate access scope and measure latency/RSS on target data. Verify
   the actual remote server/model if using the API backend.
2. **Next developer-experience increment:** documented Python-to-CLI lifecycle,
   strict CLI snapshot identity, command-specific resource flags and an API
   reference. Publish a new tested alpha when ready to distribute these fixes.
3. **Next scale increment:** blockwise hard-negative mining/top-k, bulk or durable
   ingestion, a vector-database adapter, hybrid retrieval and token-aware chunking.
4. **Training/operations expansion:** optimizer resume, distributed training,
   target GPU tests, dynamic serving batches, observability and recovery/load tests.

Do not infer priority from model size alone. A small domain pilot can use a hosted
encoder and the current retrieval interface; a large local model does not remove
the storage, evaluation and operations gaps.

## Verification record

- Baseline at `19421a5`: 306 tests plus six subtests passed; two model-download
  cases deselected.
- Final combined local run: **427 tests plus six subtests passed**, with two
  model-download cases deselected, in 34.89 seconds. This includes the ordinary
  suite and independent scientific/core/numerical-gradient contracts.
- Ruff lint and formatting pass (66 files), mypy passes (37 source files), and
  Bandit passes. Eight applicable offline examples/templates pass. These counts
  overlap focused regression suites and must not be added to them.
- The independent final agent review found no remaining P1/P2 blocker in these
  changes. Its optional benchmark-metadata clarification was applied without
  altering measured samples. This is a second-agent review, not external certification.
- All **22 hosted jobs passed** on audited source revision
  `24ea404e8e6121d4b4016484805f1584c852abb1`:
  [standard CI](https://github.com/ZouhairMudakka/clmkit/actions/runs/37620279556)
  **10/10**,
  [independent validation](https://github.com/ZouhairMudakka/clmkit/actions/runs/37620279557)
  **9/9**, and
  [Qwen 0.6B CPU reference/training checks](https://github.com/ZouhairMudakka/clmkit/actions/runs/37620337728)
  **3/3**. These runs cover clean installed wheels, the minimum ML stack, actual
  MCP stdio, embedding/reranker references and bounded LoRA training/reload.
- Local environment: Windows/Python 3.13, retained virtual environment inheriting
  system packages; CPU torch 2.13, transformers 5.17, PEFT 0.21.1. It is not a clean
  install or a fresh dependency-security verdict. The hosted runs above provide
  separate clean-environment checks on the named source revision.
- Historical real Qwen 0.6B evidence remains associated with its recorded commits
  in [validation status](VALIDATION_STATUS.md). Neither the historical checks nor
  the new CPU checks establish GPU/CUDA or 4B/8B support or held-out quality gains.
