# Validation status — 7 October 2026

The [framework review](FRAMEWORK_REVIEW.md) records post-alpha correctness and
efficiency changes, their verification, and developer fit.
The historical remediation counts below refer only to their named commits.

**The corrected framework meets the documented CPU public-alpha criteria.**
The two P1 blockers and narrower supported-contract findings are resolved. This
assessment does not establish production readiness, GPU capacity or quality gains.
The first GitHub prerelease is `v0.1.0a1`; its release page records the tagged commit
and release-candidate CI results. That legacy release predates the latest
security/correctness fixes and templates; use the audited source revision in the
[install instructions](../README.md#install). No PyPI package is published. The remediation
evidence below refers to its recorded commits and remains distinct from release
packaging/quickstart checks.

## Relevance milestone in progress — 7 October 2026

Draft [PR #4](https://github.com/ZouhairMudakka/clmkit/pull/4) adds hybrid retrieval,
label-aware training, blockwise mining and a registered public-data study.
The independently reviewed source freeze is
`e61cc76d6cf52f25422335d42eaf18dde44e3067`; all 10
[standard CI jobs](https://github.com/ZouhairMudakka/clmkit/actions/runs/37672464025)
and nine [independent validation jobs](https://github.com/ZouhairMudakka/clmkit/actions/runs/37672464030)
passed on that revision, alongside all three
[Qwen 0.6B reference jobs](https://github.com/ZouhairMudakka/clmkit/actions/runs/37673746828).
The [retained status receipts](validation-evidence/2026-10-07-relevance/)
record their per-job outcomes. These results do not cover later commits automatically.

The additive examples passed a combined local run of 594 tests, with two opt-in
model tests skipped, plus lint, formatting and type checks. Extending the serving
benchmark to all four planned concurrency levels passed 17 focused tests.
The independent review identified and corrected an installed-test packaging
omission. The next hosted run passed all 10 standard CI jobs but exposed an
unmocked optional AnyIO import in the new NumPy-only serving test. An isolated
import guard reproduced that failure and verified its correction (12 passes,
five expected HTTPX skips); another hosted run must verify clean installation.

The bounded compact-model pilot passed in the personal Codespace. Development
training/comparisons have started, but no completed held-out result is claimed
here. See [reproduction instructions](RELEVANCE_EVIDENCE.md) and
[review findings](RELEVANCE_REVIEW.md). This milestone is not yet a new alpha
release or evidence of relevance improvement.

## Earlier verified framework revision — 7 October 2026

Audited source revision `24ea404e8e6121d4b4016484805f1584c852abb1` includes the
framework review fixes, credential-persistence guard and tested use-case templates.
All **22 hosted jobs passed on this same revision**:

| Validation | Result |
|---|---|
| [Standard CI](https://github.com/ZouhairMudakka/clmkit/actions/runs/37620279556) | 10/10 jobs: Linux/Windows/macOS core, full CPU Python 3.10/3.12, lint/type checks and security scan. |
| [Independent validation](https://github.com/ZouhairMudakka/clmkit/actions/runs/37620279557) | 9/9 jobs: independent contracts, six installed NumPy-only wheel jobs, actual MCP 1.x stdio and minimum supported ML stack. |
| [Qwen 0.6B reference checks](https://github.com/ZouhairMudakka/clmkit/actions/runs/37620337728) | 3/3 jobs: embedding and reranker CPU references, bounded LoRA training and reload. |

The framework review also records **427 local tests plus six subtests passed**,
with two model-download cases deselected. Local and hosted counts overlap and
must not be added together. These checks do not establish GPU/CUDA or 4B/8B
capacity, held-out quality gains, production reliability or commercial ROI.

## Verified remediation — 7 October 2026

Product commit `7bb7a07` fixes epoch truncation, pre-update numerical guards,
weighted Matryoshka pairing, output-dimension reload, mixed instructions, memory
settings, encoder fingerprints, saved-store integrity and malformed remote output.
HF/PEFT loaders require safetensors; supported minimum ML versions are torch 2.13,
transformers 5.17, PEFT 0.21.1 and safetensors 0.8.

Local combined validation passes **253 tests and six subtests**, with two opt-in
real-model checks skipped. Actual protocol probes pass **44/44**; additional
LangChain/transport probes pass **10/10**. Ruff, format, mypy and Bandit pass.
The combined run collected before one final mocked model-revision test was added;
that test passed separately, as did the final snapshot-version and custom-encoder
guards. The full exact-commit hosted checks below include these final additions.

Workflow-only commit `234c2942c256055a09e79eb0c2b241fe029ea153` refreshes packaging
tools before every CPU model installation. On that commit all **22 hosted jobs pass**:

| Validation | Result |
|---|---|
| [Standard CI](https://github.com/ZouhairMudakka/clmkit/actions/runs/37533089986) | 10/10 jobs: Linux/Windows/macOS core, full CPU Python 3.10/3.12, static checks and fresh dependency scan. |
| [Independent validation](https://github.com/ZouhairMudakka/clmkit/actions/runs/37533090000) | 9/9 jobs: scientific 14/14, core 20 methods plus six subtests, protocol 44/44, additional 10/10; six installed-wheel jobs; MCP 1.x; minimum ML stack with passing audit. |
| [Real Qwen 0.6B](https://github.com/ZouhairMudakka/clmkit/actions/runs/37533195934) | 3/3 jobs: guarded embedding/reranker references and one-step LoRA with finite updates and adapter reload parity. |

The exact minimum ML job pins torch 2.13.0+cpu, transformers 5.17.0, PEFT 0.21.1
and safetensors 0.8.0; transitive packages are freshly resolved. It reports 236
passes and 11 optional skips, with slow tests excluded. Each initially preserved
installed-wheel artifact reports 135 passes and 17 optional skips (two slow tests
deselected); the final six-job wheel matrix also passes on unchanged product code.

The extra-high independent review found no remaining P1 implementation blocker.
See [remediation and remaining gaps](REMEDIATION.md) and
[retained per-commit evidence](validation-evidence/2026-10-07-remediation/).
Private vulnerability reporting is enabled and verified. Reference tolerances
remain embedding `atol=2e-3` and reranker `atol=1e-6`; no general quality claim follows.

The first remediation runs passed behavior checks but retained setuptools 78.1.0
in contract/reference environments. The final refreshed artifacts record 84.0.0;
dedicated current/minimum security jobs pass. Older results remain qualified in
the evidence directory. The laptop's inherited cryptography advisories are not
silently reclassified as fixed by these fresh hosted results.

Memory decay explicitly ranks within a bounded semantic candidate pool (default
four times `k`); the independent oracle now tests both that documented limitation
and a larger pool recovering its global-best fixture. No failure was hidden with
xfail. Mutable weights need caller-provided `encoder_identity`; fingerprints are
configuration checks, not authentication. Saves remain trusted, nontransactional,
single-writer snapshots. These scope limits persist after remediation.

## Historical baseline (before remediation)

The original Windows CPU suite passes (102 tests, two real-model tests skipped).
A newly downloaded NumPy-only environment passes 62 installed-wheel tests; 13
optional tests skip and two real-model tests are deselected. Existing static and
package checks pass. Independent contract probes expose unresolved defects.

### Baseline release findings

- Epoch-based training can omit examples when duplicate avoidance expands the batch count.
- Declared dependency floors permit a known unsafe legacy checkpoint-loading combination.
- Non-finite loss/gradient handling, weighted Matryoshka pairing, embedding dimension
  persistence and mixed-instruction evaluation require correction.
- Memory configuration, encoder identity, saved-store integrity and remote output
  validation have narrower defects or require explicit scope restrictions.
- The retained development environment includes an old cryptography dependency
  with known advisories. Fresh dependency scans must be evaluated independently.

The independent workflow intentionally remains red while its contract assertions
fail. The existing CI badge alone is not a release-readiness verdict.

### Baseline evidence limits

Hosted checks have now executed. [Ordinary CI](https://github.com/ZouhairMudakka/clmkit/actions/runs/37529157417)
passes all ten jobs, including the original platform matrix and security scan.
[Independent validation](https://github.com/ZouhairMudakka/clmkit/actions/runs/37529157324)
passes six installed-wheel matrix jobs and MCP 1.x, but its contract job remains
red. On the same product source, preserved scientific JUnit reports 6 passes/8
failures, core JUnit 7 passes/19 failing cases, protocol 38/6 and additional
integration probes 6/4. These overlapping assertion counts are not unique bugs.

[Reference checks](https://github.com/ZouhairMudakka/clmkit/actions/runs/37527734885)
pass for both Qwen 0.6B models, using recorded immutable revisions. The embedding
test enforces 2e-3 tolerance and the reranker 1e-6; these results do not reproduce
the builder's more precise historical numerical claims. A separate personal
Codespace (4 CPUs, 16 GB RAM) passes one rank-2 LoRA step on two synthetic pairs,
finite-parameter checks, actual adapter updates, and adapter save/reload parity
(observed query/document errors both zero). Total probe time including download
was 47.84 seconds; peak process RSS was about 5.04 GiB. This is execution evidence,
not held-out quality improvement. [Preserved evidence](validation-evidence/2026-10-07/).

The first clean CPU install pulled vulnerable setuptools 78.1.0 from PyTorch's
CPU package index as a torch dependency (four records, two advisory IDs). Updating
setuptools to 84.0.0 from PyPI cleared the Codespace scan; CI now updates packaging
tools before installing CPU torch. Both before/after scans are retained. This
does not fix the separate vulnerable versions permitted by the project's declared
torch/transformers minimums.

4B/8B capacity, CUDA, multi-GPU training and production reliability have not been
validated. Treat large-model recipes as experimental. Persistence is intended
for trusted, complete, single-writer snapshots; FAISS native indexes and external
model checkpoints must not be described as universally safe to load.
