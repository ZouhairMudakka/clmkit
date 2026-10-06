# Audit remediation — 7 October 2026

The independent audit found two high-priority supported-use defects: duplicate-aware
training could omit examples, and dependency declarations admitted an unsafe legacy
checkpoint-loading combination. Product commit `7bb7a07` fixes these and the narrower
scientific, persistence and remote-response findings. Commit `234c294` changes only
CI packaging-tool preparation. The independent review found no remaining P1
implementation blocker within the documented CPU alpha scope.

| Area | Correction and acceptance evidence |
|---|---|
| Training coverage | Epochs consume their actual deterministic duplicate-aware batch plan; explicit step budgets retain precedence. Tests cover duplicate patterns, seeds, repeated epochs and short/long step limits. |
| Numerical safeguards | Loss, representation/model gradients and clipping norm are checked before optimizer mutation. Ordinary and GradCache failure tests compare parameters, optimizer/scheduler state and counters. |
| Weighted loss/evaluation | Matryoshka weights remain paired with their dimensions; invalid dimensions/weights reject. Evaluation preserves individual query instructions. |
| Model loading | Supported floors are torch 2.13, transformers 5.17, PEFT 0.21.1 and safetensors 0.8. All HF loaders require safetensors; explicit adapters resolve safe weights and matching config. Harmless legacy fixtures verify rejection. |
| Checkpoint reload | Native/truncated output dimensions, explicit overrides, adapter-only and merged saves have tiny-model round trips. Real 0.6B inference and adapter-update/save/reload checks run in CI. |
| Persistence | Memory settings persist; snapshots validate IDs, vector finiteness and format versions. FAISS settings and native-index consistency are checked after trusted loading. |
| Encoder compatibility | Versioned fingerprints include built-in vector-producing configuration. Caller-supplied immutable identity covers mutable/custom weights. Legacy snapshots warn about shallow identity. |
| API boundaries | Remote indices must exactly cover integer input positions; malformed JSON, shapes and non-finite vectors reject. Shared encoding also rejects float32 overflow and invalid hidden tail components. |

The extra-high independent review found two remaining boundary omissions during
remediation (snapshot version and custom-encoder finiteness); both were fixed and
independently rechecked before final hosted validation. Tests were not marked xfail.
The memory-decay oracle explicitly tests the supported bounded candidate policy
and demonstrates that a larger pool can recover its global-best counterexample.

See [validation status](VALIDATION_STATUS.md) for the final run verdict and
[preserved evidence](validation-evidence/2026-10-07-remediation/) for per-commit
results, package versions, numerical tolerances and artifact digests.

## Remaining gaps

| Gap | Alpha disposition / next validation |
|---|---|
| GPU/CUDA, 4B/8B, multi-GPU | Experimental recipes only; execute on suitable isolated GPU hardware before asserting capacity or numerical parity. A CPU VPS does not close these gaps. |
| Held-out quality | No generalization claim. Use uncontaminated task splits, multiple seeds and meaningful baselines before claiming improvement. Reranker prompt formatting still shares constants with its reference. |
| Heavy duplicate sampling | Counting is correct but duplicate-heavy batching remains quadratic. Benchmark and redesign before large repeated-query workloads. Singleton InfoNCE needs additional contrastive candidates. |
| Global decayed memory ranking | Default `4*k` candidate pool is approximate; use an adequate pool and exact backend for a global guarantee. FAISS filtering remains bounded. |
| Identity and storage trust | Mutable/custom weights require `encoder_identity`. Safetensors is not repository authentication; native FAISS artifacts must be trusted. Snapshots require a single writer and complete files; no transaction/recovery guarantee. |
| Production serving | Authentication-derived tenant scope, gateway/TLS, load limits, durable shared state, monitoring and recovery remain deployment work. Multiple writable workers do not share the in-memory store. |
| Training resume / broader retrieval | Optimizer resume, distributed training, hybrid retrieval and database connectors remain optional roadmap items. |
| Distribution / maintenance | The subsequent v0.1.0a1 GitHub prerelease packages this work; no PyPI publication. Release preparation upgrades and SHA-pins the Actions dependencies. Fresh scans do not repair the retained laptop environment's disclosed cryptography advisories. |

No client production host was used. The personal Codespace from earlier testing
remains stopped; the corrected candidate was verified using local tests and hosted CI.
