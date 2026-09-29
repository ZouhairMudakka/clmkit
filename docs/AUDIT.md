# Audit report: clmkit v0.1.0

*Date: 2026-09-30. Environment: Windows 11, Python 3.13.1, CPU-only torch 2.13, transformers 5.17, peft 0.21, fastapi 0.142 / starlette 1.7, mcp 2.2.0 (and 1.28.1), faiss-cpu 1.15.*

## Summary

| Check | Tool | Result |
|---|---|---|
| Unit + integration tests | pytest 9 | **102 passed**, 2 skipped (opt-in real-model tests; both pass when enabled, see below) |
| Coverage | pytest-cov (branch) | **95%** overall |
| Core without optional deps | clean venv with only `numpy` + `pytest` | `import clmkit` does not load torch; 62 passed, 15 skipped (optional backends) |
| Lint / style | ruff 0.16 (E, F, W, I, B, UP, SIM, C4, RUF, **S (bandit rules)**, PT, RET, PIE) + ruff format | clean |
| Types | mypy 2.3 (`check_untyped_defs`, `warn_unused_ignores`) | clean, 37 files |
| Security static analysis | bandit 1.9 | 0 issues (3 reviewed false positives annotated with justification) |
| Dependency vulnerabilities | pip-audit 2.x | **0 known vulnerabilities** in clmkit's resolved dependency closure (89 packages) |
| Real model: Qwen3-Embedding-0.6B | vs official model-card scores | max abs diff **1.5e-7** |
| Real model: Qwen3-Reranker-0.6B | vs model-card reference code | max abs diff **4.7e-10** (fp32) |
| Real model: fine-tuning | LoRA + GradCache + MRL on Qwen3-Embedding-0.6B (CPU) | trains, evaluates, checkpoints, reloads |

## Real-model validation

Model weights were downloaded at pinned revisions, used offline, and deleted afterwards.

**Qwen3-Embedding-0.6B** (`97b0c614…`). The model card's transformers example prints
`[[0.7645568, 0.1414251], [0.1354974, 0.5999550]]` for its 2 queries × 2 documents. clmkit (`load_encoder(...)`, `kind="query"` / `"document"`, default instruction) produced
`[[0.7645570, 0.1414250], [0.1354980, 0.5999550]]`, a maximum absolute difference of 1.5e-7 (float32 round-off). At 256 Matryoshka dimensions the ranking is preserved. This confirms the prompt template, `<|endoftext|>` handling, left padding and last-token pooling. The test is automated as `tests/test_real_model.py` (`CLMKIT_TEST_MODEL=Qwen/Qwen3-Embedding-0.6B pytest -m slow`).

**Qwen3-Reranker-0.6B** (`e61197ed…`). The model card prints no reference scores, so the card's code was run verbatim next to `LLMYesNoReranker` on matched and mismatched pairs. The input token ids were identical. In fp32 the scores agree to 4.7e-10. With transformers v5's default bf16 load, the reference code reports `1.0` while clmkit reports `0.99954`. The fp32 truth is `0.99950`: the reference applies `log_softmax` to bf16 logits, and clmkit upcasts to fp32 first, so clmkit's bf16 result is the more accurate one. This is automated as `CLMKIT_TEST_RERANKER=Qwen/Qwen3-Reranker-0.6B pytest -m slow`.

**Fine-tuning.** `clmkit train` on `configs/qwen3-embedding-0.6b-full.yaml`, with LoRA r=8 (2.29M trainable params, 0.38% of the model), GradCache chunk 3 of batch 6, MRL dims [1024, 256], ran for 6 steps on CPU. It evaluated and saved the best and final adapter-only checkpoints, and the checkpoint reloaded through `load_encoder` with the correct Qwen3 settings. On the tiny sample set the base model already scores 1.0 on every metric, so this run validates the *pipeline*, not a quality gain. Learning is validated on the tiny model in the test suite: MRR@5 improves by more than 0.2, and loss falls by more than 50%.

**Not executed:** Qwen3-Embedding-4B/8B and Qwen3-Reranker-4B/8B. The dev machine has no GPU and 8 GB RAM. They use the same code path and presets as the verified 0.6B models, differing only in `mrl_range`.

## Defects found and fixed during the audit

| # | Severity | Finding | Fix | Regression test |
|---|---|---|---|---|
| 1 | **High** | *Qwen3 pooling token pitfall.* The Qwen3-Embedding tokenizer's `eos_token` is `<\|im_end\|>`, but embeddings are pooled from `<\|endoftext\|>`. A generic "append EOS" implementation would pool from the wrong token and silently degrade quality. | Preset names the pooling token explicitly; append only when absent | `test_eos_is_appended_once_and_is_the_pooled_token`; real-model test |
| 2 | **High** | *MCP SDK 2.x break.* `mcp` 2.x removed the decorator API (`Server.list_tools()`), so the MCP server would crash on every fresh install. | Version-tolerant `build_server` (constructor handlers on 2.x, decorators on 1.x) | `test_mcp_server_lists_and_calls_tools`, run against mcp 1.28.1 and 2.2.0 |
| 3 | **Medium (security)** | *API-key leak on redirect.* `urllib` follows redirects and forwards the `Authorization` header, including to other hosts, so a malicious or misconfigured endpoint could harvest the key. | Remote client refuses all redirects | `test_redirects_are_refused_so_the_api_key_cannot_leak` |
| 4 | **Medium (security)** | *Vulnerable dependency floors.* Minimum versions allowed starlette/h11/mcp releases with published CVEs. | Raised floors: `starlette>=1.3.1`, `h11>=0.16`, `mcp>=1.28.1`, `fastapi>=0.115` | pip-audit in CI |
| 5 | Medium | *BEIR loader dropped data.* A headerless `qrels.tsv` lost its first judgement (header detection assumed numeric query ids). | Detect the header by a non-numeric score column; error on malformed rows | `test_load_beir` |
| 6 | Low | *O(n²) batcher.* Duplicate-avoiding batching rescanned all pending examples per batch. | Single streaming pass with a small retry queue | `test_iter_batches_is_linear_time_on_large_input` (20k examples) |
| 7 | Low | *AP@k definition.* Normalised by `min(|rel|, k)`, not trec_eval's `|rel|`. | Normalise by all relevant documents | `test_binary_metrics_hand_computed` |
| 8 | Low | *Asserts as runtime checks* (removed under `python -O`). | Explicit exceptions | bandit B101 clean |
| 9 | Low | *`zip()` without `strict`* could silently truncate on length mismatches. | `strict=True` on all 15 call sites | ruff B905 clean |
| 10 | Info | FastAPI cannot resolve locally defined request models under `from __future__ import annotations`. | Module intentionally omits it (commented) | serve tests |
| 11 | Info | transformers v5 loads checkpoints in their stored dtype (bf16) by default: slow and less precise on CPU. | `dtype="auto"` resolves to fp32 on CPU and bf16 on capable CUDA | `test_truncation_mrl_dtype_and_misc` |

## Security design review

| Area | Control |
|---|---|
| Deserialisation | No pickle anywhere. Indexes use `.npy` loaded with `allow_pickle=False` plus JSON; a pickled array is rejected (tested). Configs use `yaml.safe_load` (a `!!python/object` payload is rejected, tested). |
| Remote code | `trust_remote_code=False` by default; `revision=` supported for pinning hub commits. |
| Network client | http(s) schemes only (`file://` rejected, tested); redirects refused; retries with backoff; keys read from env vars, never included in `repr`; warning when sending a key over plain HTTP to a non-local host. |
| REST server | Binds `127.0.0.1` by default (warns if exposed without a key); optional bearer auth with constant-time comparison; limits on batch size, text length, `k`, and instruction length; writes disabled unless `--allow-writes`; `/health` is the only unauthenticated route. |
| Agent tools | JSON-Schema validation of LLM-produced arguments (types, required, bounds, `maxLength`, enums, unexpected keys); errors returned as tool errors. |
| Secrets in repo | `.gitignore` excludes `.env*`, weights, caches and run outputs. |

### Residual risks and limitations

- **Prompt injection through retrieved content** is inherent to RAG and memory. It's documented in [AGENTS.md](AGENTS.md); write tools are off by default.
- **No rate limiting or multi-tenant auth** in the REST server. Deploy behind a gateway.
- **Hub downloads are trust-on-first-use** unless `revision` is pinned.
- **The REST server serialises inference** with a lock (correctness over throughput). Scale with multiple workers.
- **4B/8B presets are untested on real weights** (see above).
- **CI has not yet produced a run** at the time of writing; it runs on the first push.

## Note on the development environment

`pip-audit` over the developer's *global* Python environment flagged 18 unrelated packages (for example pypdf, pillow, authlib, urllib3, cryptography, werkzeug). None of them is in clmkit's dependency closure. They're recorded here only to explain why auditing a shared environment differs from auditing a fresh install, and CI audits a fresh install.

## How to reproduce

```bash
pip install -e ".[dev,train,serve,mcp,faiss,sklearn]"
ruff check src tests && ruff format --check src tests && mypy
pytest -q --cov=clmkit
bandit -q -r src/clmkit
pip-audit --skip-editable
CLMKIT_TEST_MODEL=Qwen/Qwen3-Embedding-0.6B CLMKIT_TEST_RERANKER=Qwen/Qwen3-Reranker-0.6B pytest -m slow tests/test_real_model.py
```
