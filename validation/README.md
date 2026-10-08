# Independent validation

These probes preserve contracts proposed in the October 2026 audit and the
subsequent evidence study. Some were designed to expose failures in the initial
implementation. They are not marked xfail: a passing ordinary suite alone does
not establish that every independent contract or model-quality claim is verified.

For the alpha package and examples, use the `v0.1.0a2` tag as described in the
[installation guide](../README.md#install). Run validation commands from that
matching Git checkout; the wheel installs the library, not this tooling. The
source distribution includes the evidence runners and templates, while the Git
checkout also carries the broader audit tooling and documentation.

- `test_scientific_contracts.py`: independent objective/gradient checks and tiny-model defects.
- `test_core_independent.py`: synthetic retrieval, evaluation and persistence probes.
- `run_protocol.py`: real loopback HTTP and MCP stdio exchanges.
- `additional_probes.py`: actual LangChain integration and bounded transport checks.
- `benchmark_framework.py`: bounded offline CPU workloads and optional paired
  baseline/current batching measurements; see its header for reproduction commands.
- `run_installed.py`: wheel built from sdist, installed into a new NumPy-only environment;
  checks archive contents, the README quickstart, and tests including the
  repository use-case templates copied outside the source checkout.
- `prepare_reference.py`: resolve/record a model revision for manual 0.6B reference CI.
- `check_reference_result.py`: fail if the intended reference test skipped or did not pass.
- `real_training_smoke.py`: bounded Linux CPU LoRA step, finite-parameter/update checks and adapter save/reload parity; run with `timeout 1200s python -u validation/real_training_smoke.py --threads 4`.

The two original reference tests use different acceptance tolerances: embedding
model-card scores allow 2e-3; reranker parity allows 1e-6. Historical smaller
observed differences are not the tests' enforced tolerances. These checks do not
prove training quality, 4B/8B support, or GPU execution. They run separately to
bound model memory and disk usage. Model weights are never uploaded as artifacts.

The memory-decay probe now checks the documented bounded candidate pool and a
larger pool that recovers its global-best fixture. Exact global decayed top-k is
not promised by the default approximate policy. No tests are marked xfail.
Failure counts are not counts of release blockers. See the
[validation status](../docs/VALIDATION_STATUS.md).

## Relevance study and workflow verification

Follow the [evidence protocol](../docs/RELEVANCE_EVIDENCE.md) for the authorized
Codespaces workflow, split controls, resource bounds and reproduction commands:

- `evidence_data.py` prepares pinned public artifacts and checksummed manifests.
- `evidence_run.py` locks the protocol, runs individual evaluations/training,
  and seals development choices before held-out evaluation.
- `evidence_suite.py` runs the registered jobs sequentially and retains retry history.
- `evidence_report.py` verifies prediction artifacts and recomputes metrics
  against pinned judgments before producing a report.
- `evidence_dx.py` compares bounded workflows with pinned weights; it does not
  measure human productivity.
- `benchmark_serving.py` measures a bounded authenticated CPU loopback workload;
  it does not establish production capacity or a service-level agreement.

The [intent-matching recipes](../templates/intent_matching/README.md) apply the
prepared galleries to pinned-model retrieval, adapted index rebuilding and
separate CLINC rejection. Their offline fixture tests and the NumPy-only
installed-wheel checks establish contracts, not real-model relevance.

Read the [recorded study evidence](../docs/validation-evidence/2026-10-08-relevance/README.md)
and [independent review](../docs/RELEVANCE_REVIEW.md) for completion status and
limitations. The recorded study source is
`e61cc76d6cf52f25422335d42eaf18dde44e3067`; the `v0.1.0a2` release version does
not replace that source identity or the hashes bound by its protocol and seal.
BANKING77 same-intent matching, SciFact evidence retrieval and CLINC out-of-scope
rejection have different meanings and denominators. None establishes general
customer-service accuracy or authorizes business actions. No model weights or
raw benchmark texts belong in public validation reports.
