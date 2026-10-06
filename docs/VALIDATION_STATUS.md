# Validation status — 7 October 2026

This repository is published for development and validation. **It is not yet a
verified public-alpha release.** No release tag is approved by these results.

The original Windows CPU suite passes (102 tests, two real-model tests skipped).
A newly downloaded NumPy-only environment passes 62 installed-wheel tests; 13
optional tests skip and two real-model tests are deselected. Existing static and
package checks pass. Independent contract probes expose unresolved defects.

## Unresolved release findings

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

## Evidence limits

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
