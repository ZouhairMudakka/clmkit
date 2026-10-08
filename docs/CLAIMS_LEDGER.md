# Evidence-to-claims ledger — draft

**8 October 2026.** All 13 registered held-out comparisons completed and the
independent metrics retry passed all 13. This ledger separates implemented
mechanics, measured outcomes and pending evidence. It does not announce a new alpha release.
Use the [results draft](RELEVANCE_RESULTS.md) for denominators, uncertainty and
interpretation, the [protocol](validation-evidence/2026-10-08-relevance/protocol.json)
for registered choices, and the [development JSON](validation-evidence/2026-10-08-relevance/report-dev-oct8/report.json)
for development values and artifact hashes. Held-out claims below use the
[test JSON](validation-evidence/2026-10-08-relevance/report-test-oct8/report.json)
and [independent retry receipt](validation-evidence/2026-10-08-relevance/independent-metrics-oct8-retry1.json).

Product/evidence execution source is
`e61cc76d6cf52f25422335d42eaf18dde44e3067`; later examples, packaging and
documentation have their own revision-specific checks. Historical test counts
overlap and must not be added together. The
[validation status](VALIDATION_STATUS.md) identifies the revisions actually checked.

## Implemented behavior

| ID | Supported claim | Evidence | Boundary |
|---|---|---|---|
| F01 | NumPy-only BM25 and exact dense/BM25 reciprocal-rank fusion are implemented with shared metadata eligibility and deterministic ties. | [Implementation](../src/clmkit/hybrid.py), [tests](../tests/test_hybrid_retrieval.py), [API guide](RELEVANCE_EVIDENCE.md#lexical-and-hybrid-retrieval) | Python-only, in-memory, single-writer; no hybrid snapshot/CLI/REST or learned sparse model. Similarity is not authorization. |
| F02 | Label-aware sampling admits at most one training pair for each label in an effective batch, including across GradCache chunks. | [Trainer](../src/clmkit/training/trainer.py), [tests](../tests/test_label_aware_training.py) | Requires labels, may shrink batches, disables explicit negatives. Not a multi-positive loss or a guarantee against noisy labels. |
| F03 | Hard-negative scores are computed in query/corpus blocks with optional same-label exclusions. | [Data utilities](../src/clmkit/data.py), [tests](../tests/test_label_aware_training.py) | Corpus embeddings/model/merge buffers remain resident; bounded candidate inspection may return fewer negatives. No measured end-to-end speedup claim from implementation alone. |
| F04 | CLI index loading is strict by default; incompatible/legacy vectors cannot silently be relabeled through add/save. | [Retriever](../src/clmkit/retrieval.py), [CLI](../src/clmkit/cli.py), [lifecycle tests](../tests/test_index_lifecycle_evidence.py) | Python loading still needs `strict=True`. Generic identities are caller-supplied; arbitrary weight changes are not detected automatically. Snapshots are nontransactional. |
| F05 | The experiment binds prepared inputs, source, model/checkpoint identity, thresholds and development artifacts before test access; reports recheck predictions and metrics. | [Runner](../validation/evidence_run.py), [reporter](../validation/evidence_report.py), [review](RELEVANCE_REVIEW.md) | Integrity controls do not establish scientific truth or immunity to public-benchmark pretraining exposure. Evidence scripts are source-checkout assets. |
| F06 | Full-gallery intent recipes enforce strict reload and separate CLINC calibration. | [Recipe](../templates/intent_matching/README.md), [tests](../tests/test_intent_matching_template.py), [real-model receipt](validation-evidence/2026-10-08-relevance/aux-oct8/receipt.json) | Real-model build and fresh-process reload completed for 8,006-document BANKING frozen/adapted and 15,000-document CLINC galleries. BANKING labels are not resolution truth; CLINC thresholds are not permission to act. |

## Held-out observations

The following wording applies to the fixed registered systems and datasets.
Metric verification and final independent claim review passed within the documented CPU alpha scope. Publication and exact-candidate checks are recorded separately. The
first checker stopped after seven BANKING runs on a CLINC BM25 schema assumption
(`model: null` in the result versus `model: minilm` in its threshold). A strict
explicit-model correction passed the complete retry. Preserve that failed receipt;
the correction did not change models, thresholds or predictions.

| ID | Supported held-out wording | Evidence and qualification |
|---|---|---|
| T01 — BANKING adaptation | “Mean adapted Hit@1 was 92.72%, versus 91.79% frozen: +0.93 percentage points, paired grouped 95% interval [+0.29,+1.56].” | 3,080 queries, 3,070 groups, 2,000 bootstrap draws, conditional on all three trained seeds. Seeds 42/1729/2026 scored 92.89%/92.60%/92.66%. Mean and interval upper bound are below the registered two-point target. Excluding 42 training-overlap queries leaves 3,038: +0.94 points, interval [+0.27,+1.59]. |
| T02 — SciFact retrieval | “Hybrid had the highest observed test nDCG@10 (0.6865), compared with BM25 0.6647, frozen dense 0.6451 and rerank 0.6804.” | 300 queries, full 5,183-document corpus and original multiple judgments. Hybrid Recall@10/@100: 0.8179/0.9543; rerank had highest Hit@1, 56.67%. Dense top-20 shortlist hit/recall: 84.67%/83.73%. No overlap queries; no paired superiority interval or universal superiority claim. |
| T03 — CLINC rejection | “Frozen development thresholds yielded test OOS false acceptance of 6.70% BM25 and 7.50% dense, with in-scope coverage 53.80% and 87.67%.” | All 4,500 in-scope and 1,000 OOS queries retained. Accepted in-scope accuracy: 91.95%/91.25%, excluding accepted OOS. The development 5% target is not a test guarantee. Separate ranking sensitivity excludes 52 queries, leaving 4,448 in-scope; thresholds are never transferred to BANKING. |
| T04 — BANKING method choice | “Frozen dense outperformed hybrid and reranking on primary test Hit@1 in this configuration.” | Frozen 91.79%, hybrid 91.17%, rerank 90.49%; BM25 78.96%. Recommend frozen dense as a simple starting baseline here, weighing adaptation's modest gain against cost. This does not prescribe a universal retrieval architecture. |
| T05 — intent-level limitations | “Average adaptation gains conceal label-level regressions.” | [Descriptive breakdown](validation-evidence/2026-10-08-relevance/banking77-intent-breakdown.json): `pending_transfer` 77.50% → 73.33% mean adapted; `balance_not_updated_after_bank_transfer` 80.00% → 75.83%. Weak frozen labels `top_up_failed` and `card_payment_not_recognised` improved to 83.33% and 81.67%. Fixed-seed means are not an ensemble; these aggregates do not explain confusion causes. |

The [results draft](RELEVANCE_RESULTS.md) gives all method tables, full denominators,
overlap sensitivity and measured costs. SciFact favors comparing hybrid with
lexical and dense baselines; BANKING does not support adding hybrid or reranking
by default. Warm retrieval quantiles exclude reranking. Paired component sums
are not end-to-end HTTP latency, and these CPU timings do not establish an SLA.

## Development observations only

All rows below refer to the **development split**, not held-out performance.
There are 13/13 completed development comparisons and three completed training
seeds. The report remains labeled `partial_development` because it is not
held-out evidence. The interrupted SciFact attempt and retry remain in its history.

| ID | Permitted development wording | Evidence and qualification |
|---|---|---|
| D01 | “On BANKING77 development queries, mean adapted Hit@1 across three registered seeds was 92.46%, versus 91.29% frozen: +1.17 percentage points, paired grouped 95% interval [+0.38,+1.97].” | 1,997 queries, 1,984 groups, 2,000 bootstrap draws; conditional on seeds 42/1729/2026. **Below the registered two-point practical target.** No held-out improvement claim. |
| D02 | “Hybrid and reranking did not improve BANKING77 primary development Hit@1 over the frozen baseline in this configuration.” | Dev Hit@1: frozen 91.29%, hybrid 89.93%, rerank 91.14%. Retain the no-gain result and distinguish supporting metrics. |
| D03 | “On SciFact development queries, BM25/dense/hybrid/rerank nDCG@10 was 0.7099/0.6702/0.7185/0.7047.” | Full 5,183-document corpus, 162 development claims and original multiple judgments. Descriptive comparison without a paired superiority interval; not scientific truth or universal method superiority. |
| D04 | “At CLINC development-selected thresholds, both methods had 5% OOS false acceptance; in-scope coverage was 55.53% BM25 and 86.13% dense.” | Threshold selection and measurement share 3,000 in-scope/100 OOS dev queries. Accepted in-scope accuracy: 93.64%/92.14%. These are calibration outcomes, not independent test confirmation. |
| D05 | “The registered training runs completed 200 steps each, with recorded training times 444.7, 434.7 and 477.5 seconds for seeds 42, 1729 and 2026.” | [Training records](validation-evidence/2026-10-08-relevance/training/). Four-core CPU Codespace; excludes later index rebuilding and unknown interrupted-attempt duration. Not total study time, a GPU claim or production SLA. |

The [results draft](RELEVANCE_RESULTS.md) records full tables, actual reranked
shortlist coverage, CLINC overlap sensitivity and timing conventions. For support
intents, same-label relevance is a proxy. For SciFact, relevance is judged evidence
retrieval. Neither establishes accuracy on a customer's business policies.

## Auxiliary observations

All eight [auxiliary jobs](validation-evidence/2026-10-08-relevance/aux-oct8/receipt.json)
passed at source `7e6a7f5bc75f2e2a7e527304a816472947783e63`, with the frozen
core and auxiliary script identities recorded separately from the relevance study.

| ID | Supported wording | Evidence and qualification |
|---|---|---|
| X01 — framework workflow comparison | “On 100 development documents and 20 queries, clmkit and direct Sentence Transformers/NumPy produced matching ranked IDs, filters and reload rankings, with vectors equal within declared tolerances.” | [DX report](validation-evidence/2026-10-08-relevance/aux-oct8/dx.json). Both rejected wrong immutable identities. Three alternating repetitions: median encoding/index/search 0.7183 s versus 0.6356 s, about 13% descriptive overhead. Loading/persistence excluded from this timing. Same-process DX reloads; no human productivity or speedup claim. |
| X02 — HTTP behavior | “All 192 measured authenticated loopback requests succeeded at client concurrency 1/4/8/16; throughput stayed near 68–71 requests/s while p95 increased from 23.09 to 269.03 ms.” | [HTTP report](validation-evidence/2026-10-08-relevance/aux-oct8/serving.json). 48 requests per level, no failures; unauthenticated check and four warmups separate. One short fixed query, one ascending repetition, closed-loop clients, shared CPU host and serialized encoder forwards. No maximum-capacity or SLA claim. |
| X03 — real-model recipes | “Frozen and adapted BANKING indexes were built from all 8,006 gallery documents and strictly reloaded in fresh query processes; CLINC used all 15,000 gallery documents and its matching sealed threshold.” | [Commands/results receipt](validation-evidence/2026-10-08-relevance/aux-oct8/receipt.json). Adapted seed 42 rebuilt into a separate fresh directory with checkpoint identity; CLINC threshold 0.6381174325942994. No action executed. The frozen synthetic lost-card example returned `card_about_to_expire`, so successful execution must not be presented as universally correct intent prediction. |

## Pending claims and acceptance fields

| ID | Status | Evidence required before final wording |
|---|---|---|
| R01 — reviewed release | **Review passed; publication recorded separately** | [Independent review](RELEVANCE_REVIEW.md) found no remaining CPU-alpha blocker. The [v0.1.0a2 release record](https://github.com/ZouhairMudakka/clmkit/releases/tag/v0.1.0a2) must retain exact-revision checks and durable artifacts before release completion is claimed. Preserve v0.1.0a1. |

For each completed row retain the exact public sentence, split/query/corpus
counts, protocol/source/model identities, result and prediction hashes,
uncertainty/limitations, failure history, reviewer disposition and durable
artifact link. Pending means no completion or quantitative outcome is asserted.

## Claims this evidence does not support

- Accuracy on private customer policies, verified ticket resolution, business
  savings, autonomous-action eligibility or a security/authorization guarantee.
- Universal superiority over Sentence Transformers, other retrieval methods,
  workflow frameworks, LLMs or commercial systems.
- Human developer productivity improvement based on setup steps, code length or
  one scripted workflow comparison.
- Production readiness, concurrent-write durability, maximum serving capacity,
  crash recovery or an SLA.
- GPU/CUDA, Qwen 4B/8B capacity, distributed training, Arabic–English retrieval
  quality or absence of benchmark exposure during pretraining.

The current additions remain alpha capabilities with documented adoption limits.
Public installation/version statements must identify the exact revision that
contains them. The README targets v0.1.0a2; historical v0.1.0a1 does not include these additions.
