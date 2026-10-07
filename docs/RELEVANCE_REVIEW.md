# Relevance workflow review — 7 October 2026

This review covers the additions described in [the reproducible evidence guide](RELEVANCE_EVIDENCE.md).
It evaluates implementation and experiment integrity; it does not certify business accuracy or production readiness.

## Changes under review

- NumPy-only BM25 and exact dense/BM25 reciprocal-rank fusion, with the same metadata eligibility applied before both shortlists.
- Label-aware contrastive batches that keep same-intent examples out of each other's in-batch negatives, including across GradCache chunks.
- Blockwise hard-negative scoring, with aligned corpus labels and deterministic tie handling.
- Model/index provenance checks at mutation and persistence boundaries, and strict CLI loading by default.
- Pinned public datasets, a registered development/test protocol, three training seeds, rejection calibration and reproducible result verification.

An independent Astra Extra High subagent reviewed the implementation after the initial local regression run. This is an AI-assisted code review, not an external certification. No model or dataset downloads were needed for its independent reproductions.

## Findings and corrections

| Finding | Correction and verification |
|---|---|
| A permissive legacy-index load could save old vectors under a replacement encoder fingerprint | Legacy snapshots retain an unknown-provenance marker. Reading remains possible with warnings; nonempty legacy indexes must be rebuilt before additions or saves. Independently reproduced and verified. |
| The test gate did not recheck changed development artifacts after sealing | Test access now checks the sealed development results/predictions, training reports, checkpoints and rejection thresholds before opening test queries. Modified/deleted artifact regressions and an independent corruption probe verify rejection. |
| A final report could accept corrupt or incomplete prediction files | Results record prediction hashes and byte counts. Reports decode the files, require exact query coverage and valid hits, and recompute ranking/rejection metrics from checksum-verified judgments. A strengthened independent probe also rejects corrupt gzip whose metadata hash was updated to match. |
| Hybrid retrieval did not receive the protocol's BM25 parameters | Both lexical branches now receive the registered `k1` and `b`. A nondefault-parameter regression covers the wiring. The initial pilot used matching defaults and was unaffected. |
| A timed-out job left a directory that prevented resuming | Explicit resume archives incomplete output and logs before retrying. Completed results require matching protocol/source identities; prior failures and durations remain in the final report. Archive paths are constrained to the experiment directory. |

The reviewer found no further blocker in batch assembly, training step limits, mining order, checkpoint serialization, ranking formulas or rejection denominators. Independent follow-up ran 82 focused regressions; after the final history correction, all 23 report regressions passed. These sets overlap and should not be added together as a total test count.

Hosted testing separately exposed an evidence-helper import difference between `pytest` and `python -m pytest`, and a package inventory mismatch. The test support path is now explicit; source archives include only the named evidence modules, and isolated wheel tests copy those helpers without adding them to the library wheel.

## Remaining gaps

- Hybrid retrieval is a Python, in-memory API. Durable storage, CLI/REST integration and concurrent writes remain separate work.
- Hybrid searches currently recompute the encoder configuration fingerprint on each query. This is measurable framework overhead; the study includes it rather than assuming it is free.
- Blockwise mining bounds the score matrix, not the complete resident corpus embeddings or model memory.
- Label-aware training uses one example per label per effective batch and disables explicit hard negatives. It is not a multi-positive loss or a solution to noisy labels.
- Immutable weight identity is caller-supplied in the product API. Arbitrary in-place weight changes cannot be inferred from configuration alone.
- Public intent labels and relevance judgments provide reproducible proxy tasks. They cannot establish a client's policy accuracy, business savings, Arabic–English quality or production reliability.
- The compact CPU study does not validate Qwen 4B/8B execution, CUDA, distributed training or large-corpus serving capacity.
- A direct Sentence Transformers comparison can measure workflow parity and overhead. It cannot establish human developer productivity from script length or setup counts.

The source freeze was cleared for the development study. Held-out claims require a complete sealed development matrix and an independently recomputed final report; review clearance alone is not such a result.

## Additive example and serving review

The independent reviewer also checked the intent-matching recipe and loopback HTTP benchmark, with all 28 offline fixture tests passing. The recipe checks the complete prepared gallery, encoder identity and configuration before reload; CLINC rejection thresholds cannot transfer to other datasets or models. The serving benchmark uses a loopback listener, an ephemeral bearer token and cleanup of its own server child.

The new serving test helper was initially missing from the explicit source archive and isolated installed-test helper list. That inclusion was corrected. Hosted NumPy-only tests then exposed an optional AnyIO import in the mocked server-constructor test. The test now supplies that mock too; an isolated import guard reproduced the original failure and verified 12 passing tests with five expected HTTPX skips after the fix. Hosted installation verification remains the release gate. The combined local suite passed 594 tests with two opt-in real-model tests skipped; extending the concurrency matrix then passed 17 focused tests. Linting, formatting and type checks also passed.

Serving measurements are closed-loop diagnostics at specified concurrency levels, not capacity or SLA evidence. Passing fixture tests does not establish a successful real-model recipe or measured HTTP performance; those require the separate cloud runs.
