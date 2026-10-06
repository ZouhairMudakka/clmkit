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

Hosted CI results are pending until actual runs complete. A configured workflow
is not execution evidence. Real Qwen 0.6B results from the original builder are
historical and were not reproduced in the constrained local October 7 run.
Manual reference workflows are provided for that verification. They do not yet
exercise real-model training/save/reload or held-out quality improvement.

4B/8B capacity, CUDA, multi-GPU training and production reliability have not been
validated. Treat large-model recipes as experimental. Persistence is intended
for trusted, complete, single-writer snapshots; FAISS native indexes and external
model checkpoints must not be described as universally safe to load.
