# Independent validation

These probes preserve contracts proposed in the October 2026 audit. Several
deliberately fail against the initial implementation. They are not marked xfail:
a passing ordinary suite alone must not imply that these findings are resolved.

- `test_scientific_contracts.py`: independent objective/gradient checks and tiny-model defects.
- `test_core_independent.py`: synthetic retrieval, evaluation and persistence probes.
- `run_protocol.py`: real loopback HTTP and MCP stdio exchanges.
- `additional_probes.py`: actual LangChain integration and bounded transport checks.
- `run_installed.py`: wheel built from sdist, installed into a new NumPy-only environment.
- `prepare_reference.py`: resolve/record a model revision for manual 0.6B reference CI.
- `check_reference_result.py`: fail if the intended reference test skipped or did not pass.
- `real_training_smoke.py`: bounded Linux CPU LoRA step, finite-parameter/update checks and adapter save/reload parity; run with `timeout 1200s python -u validation/real_training_smoke.py --threads 4`.

The two original reference tests use different acceptance tolerances: embedding
model-card scores allow 2e-3; reranker parity allows 1e-6. Historical smaller
observed differences are not the tests' enforced tolerances. These checks do not
prove training quality, 4B/8B support, or GPU execution. They run separately to
bound model memory and disk usage. Model weights are never uploaded as artifacts.

Core probes include proposed stricter policies (for example exact decayed memory
top-k) that may instead be resolved with an explicit documented alpha scope.
Failure counts are not counts of release blockers. See `docs/VALIDATION_STATUS.md`.
