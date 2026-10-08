# Public retrieval study — results

**8 October 2026: all 13 registered held-out comparisons completed, and the
independent metrics retry passed all 13.** BANKING77 adaptation improved mean
Hit@1 by **0.93 percentage points**, with a paired grouped 95% interval
**[+0.29,+1.56] points**, below the registered two-point practical target. Hybrid
had the highest observed SciFact nDCG@10. CLINC out-of-scope false acceptance
exceeded its 5% development calibration target on held-out queries.

All eight auxiliary workflow, HTTP and real-model recipe jobs also passed.
The final independent review found no remaining blocker within the CPU public-alpha scope; see the [review](RELEVANCE_REVIEW.md). The [held-out report](validation-evidence/2026-10-08-relevance/report-test-oct8/REPORT.md)
and [full-precision JSON](validation-evidence/2026-10-08-relevance/report-test-oct8/report.json)
retain all methods, artifact hashes, timing and overlap sensitivity. The generator
reports `status: complete` and `held_out_evidence: true`. The
[independent retry receipt](validation-evidence/2026-10-08-relevance/independent-metrics-oct8-retry1.json)
records the separate metric check; see execution history below for its prior failure.

## Tasks and registered configuration

| Task | Fixed gallery/corpus | Development queries | Official held-out queries | Meaning of relevance |
|---|---:|---:|---:|---|
| BANKING77 | 8,006 training examples | 1,997 | 3,080 | Same-intent matching proxy; not actual resolutions or verified duplicate tickets |
| CLINC150 full | 15,000 in-scope training examples | 3,000 in-scope + 100 OOS | 4,500 in-scope + 1,000 OOS | Supported-intent matching and separately calibrated rejection |
| BEIR SciFact | 5,183 documents | 162, derived from BEIR training claims | 300 | Judged scientific evidence retrieval, preserving multiple relevant documents |

SciFact evaluates neither scientific truth nor medical decisions. BANKING77 and
CLINC are support-intent proxies, not tests of private customer policy, business
outcomes or action authorization. CLINC OOS examples never enter its gallery.
Dataset rights, revisions, transformations and groups are in the [manifests](validation-evidence/2026-10-08-relevance/manifests/).

Execution source was frozen at `e61cc76d6cf52f25422335d42eaf18dde44e3067`.
The [protocol](validation-evidence/2026-10-08-relevance/protocol.json) pins
`sentence-transformers/all-MiniLM-L6-v2` at
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41` and the MS MARCO MiniLM cross-encoder
at `233902d25c440f23af6f7d6e94d2946bac0bee0a`. Qwen is outside this first wave.
The artifact gate was sealed before held-out evaluation.

BM25 uses the fixed tokenizer, `k1=1.5`, `b=0.75`; hybrid uses reciprocal-rank
fusion with constant 60 and candidate depth 100. Retrieval returns up to 100
candidates. Reranking changes only the first 20 dense candidates; ranks 21–100
retain retrieval order. Candidate lists are bound by hashes. All methods within
a task use the same full corpus and eligible queries.

BANKING77 adaptation uses 200 steps per seed 42, 1729 and 2026, learning rate
`2e-5`, batch size 32, and label-aware sampling without explicit negatives. The
registered practical target is a two-absolute-percentage-point Hit@1 gain.
See [reproduction and limitations](RELEVANCE_EVIDENCE.md) and the
[implementation review](RELEVANCE_REVIEW.md).

## BANKING77 — held-out intent matching

| Method | Test Hit@1 | Test Hit@5 | Test MRR@10 |
|---|---:|---:|---:|
| BM25 | 78.96% | 93.60% | 0.8513 |
| Frozen dense | 91.79% | 97.05% | 0.9406 |
| Hybrid | 91.17% | 97.37% | 0.9384 |
| Dense + reranker | 90.49% | 97.11% | 0.9340 |
| Adapted, seed 42 | 92.89% | 97.21% | 0.9473 |
| Adapted, seed 1729 | 92.60% | 97.08% | 0.9460 |
| Adapted, seed 2026 | 92.66% | 97.08% | 0.9462 |

Mean adapted Hit@1 is **92.72%**, versus **91.79%** frozen: **+0.93 points**.
All three registered seeds improved, with Hit@1 ranging from 92.60% to 92.89%.
The grouped 95% interval **[+0.29,+1.56] points** uses 2,000 bootstrap samples
and 3,070 groups across 3,080 queries. It is conditional on these trained seeds
and supplied duplicate groups, not every source of training uncertainty. Both
the mean gain and interval upper bound are below the two-point practical target.

The fixed lexical grouping rule flags **42** test queries overlapping training.
The official score retains them. Excluding them leaves **3,038** queries and
3,028 groups: frozen Hit@1 **91.74%**, mean adapted **92.68%**, gain **+0.94
points**, interval **[+0.27,+1.59]**. The practical-target conclusion is unchanged.
The rule cannot detect every semantic paraphrase or unknown pretraining exposure.

Hybrid and reranking did not improve primary held-out Hit@1 over frozen dense.
The reranked top-20 shortlist contained a same-intent positive for **98.90%** of
queries; its mean recall of all same-intent gallery examples was **16.61%**.
These denominators differ. Start with frozen dense for this support-intent use
case, retaining BM25 as a cheap lexical comparison; justify adaptation's modest
gain against training and maintenance costs. The study does not justify adding
hybrid or reranking universally.

The [descriptive per-intent breakdown](validation-evidence/2026-10-08-relevance/banking77-intent-breakdown.json)
shows uneven gains. Among the weakest frozen labels, `top_up_failed` improved
from **75.00% to 83.33%** mean adapted Hit@1 and `card_payment_not_recognised`
from **77.50% to 81.67%**. However, `pending_transfer` fell from **77.50% to
73.33%**, and `balance_not_updated_after_bank_transfer` from **80.00% to 75.83%**.
These are descriptive label outcomes averaged across fixed seeds, not an ensemble
or new tuning experiment. They identify areas for later error review; the
aggregate does not establish which competing intents caused each error.

## CLINC150 — held-out rejection at frozen thresholds

Ranking uses all **4,500 in-scope queries**. Rejection retains all **5,500 queries**,
including **1,000 OOS** examples.

| Method | Test in-scope Hit@1 | Test OOS false acceptance | Test in-scope coverage | Accuracy among accepted in-scope |
|---|---:|---:|---:|---:|
| BM25 | 83.18% | 6.70% (67/1,000) | 53.80% (2,421/4,500) | 91.95% |
| Frozen dense | 88.02% | 7.50% (75/1,000) | 87.67% (3,945/4,500) | 91.25% |

Thresholds remain **18.14321605391373** for BM25 and **0.6381174325942994** for
dense. Each was selected on development data to maximize in-scope coverage
subject to at most 5% OOS false acceptance; both accepted 5 of 100 development OOS
examples. **That calibration target is not a held-out guarantee.** Dense accepts
more supported queries and more OOS queries than BM25. Accepted in-scope accuracy
excludes accepted OOS examples; it is not accuracy among every accepted query.

The overlap rule flags **52** held-out in-scope queries. Separate ranking
sensitivity retains **4,448** in-scope queries: BM25 Hit@1 **83.03%**, dense
**87.93%**. Rejection figures above keep the official 4,500/1,000 denominators;
they are not overlap-filtered rejection estimates. Thresholds are specific to
their method/model/gallery: never reuse them for BANKING77. Scores are not
calibrated probabilities, and rejection is not permission to act.

## SciFact — held-out scientific evidence retrieval

All **300** queries retain original multi-document judgments. No test query
overlaps training under the fixed grouping rule.

| Method | Test Hit@1 | Test nDCG@10 | Test Recall@10 | Test Recall@100 |
|---|---:|---:|---:|---:|
| BM25 | 54.33% | 0.6647 | 0.7849 | 0.8892 |
| Frozen dense | 50.33% | 0.6451 | 0.7833 | 0.9250 |
| Hybrid | 54.67% | 0.6865 | 0.8179 | 0.9543 |
| Dense + reranker | 56.67% | 0.6804 | 0.8029 | 0.9250 |

Hybrid has the highest observed nDCG@10 and recall; reranking has the highest
Hit@1. BM25 exceeds frozen dense nDCG@10 at substantially lower observed CPU
cost. Compare hybrid against both baselines for scientific retrieval; this
ordering is metric- and task-dependent, with no paired superiority interval
reported here.

The dense top-20 reranking shortlist had **84.67%** hit rate and **83.73%** mean
recall. Reranking cannot recover judged documents outside that shortlist; its
unchanged top-100 membership explains equal Recall@100 to dense. Judgments may
be incomplete; unjudged documents are not established negatives.

## Development history

All 13 development comparisons and three training seeds completed. Their
[development report](validation-evidence/2026-10-08-relevance/report-dev-oct8/REPORT.md)
and [JSON](validation-evidence/2026-10-08-relevance/report-dev-oct8/report.json)
remain parameter-selection evidence, labeled `partial_development` and
`held_out_evidence: false`. They are not pooled with test results.

Development BANKING mean adaptation gain was **+1.17 points**, interval
**[+0.38,+1.97]**, also below target; frozen/hybrid/rerank Hit@1 was
91.29%/89.93%/91.14%. CLINC development coverage was 55.53% BM25 and 86.13%
dense at the selected 5% false-acceptance operating points. SciFact development
nDCG@10 was 0.7099/0.6702/0.7185/0.7047 for BM25/dense/hybrid/rerank. The fixed
overlap rule flagged no BANKING or SciFact development queries and 35 CLINC
queries; separately labeled sensitivity is retained in its JSON.

## Costs, execution history and limits

Records identify Linux, Python 3.12.3, four CPU cores, Torch 2.14.1+cpu,
Transformers 5.19.0 and NumPy 2.5.3. Torch threads are four and interop threads
one; NumPy BLAS settings are not explicitly pinned. Timing is host-dependent.

The three 200-step training runs took **444.7, 434.7 and 477.5 seconds** for seeds
42, 1729 and 2026. BANKING frozen dense index building took 36.5 seconds and
batched retrieval 33.6 seconds for all 3,080 queries; reranking added 533.4
seconds. Adapted indexes still require rebuilding after training.

| SciFact method | Index build seconds | Batched retrieval seconds, 300 queries | Additional reranking seconds | Warm retrieval p95 milliseconds | Process peak RSS MiB |
|---|---:|---:|---:|---:|---:|
| BM25 | 1.0 | 0.33 | — | 1.76 | 312.1 |
| Frozen dense | 294.3 | 4.19 | — | 133.40 | 768.4 |
| Hybrid | 298.1 | 4.67 | — | 144.26 | 835.0 |
| Dense + reranker | 293.8 | 3.82 | 351.32 | 138.73 | 768.5 |

Load, index build, batched retrieval, reranking and training are separate fields.
Warm p50/p95 uses 30 queries and **excludes reranking**. Paired sums of separately
measured retrieval/rerank durations give p95 **366.5 ms** for BANKING and
**1,363.5 ms** for SciFact; these exclude model swap and first-stage batching and
are not measured end-to-end HTTP latency. Memory is process peak RSS, not index
size or incremental method memory.

The final SciFact development rerank attempt was interrupted when the Codespace
stopped, then completed on retry. History records the prior suite and job as
`running (previous invocation)`. Unknown interrupted duration is not included
in successful-run timings or a claimed total study cost.

The first independent checker verified seven BANKING runs, then failed on its
CLINC BM25 schema assumption: the threshold records `model: minilm`, while the
BM25 result records `model: null`. The checker was corrected to require the
explicit registered threshold model; the retry passed all 13 comparisons.
The prior failed receipt remains part of the evidence history. This checker
repair did not retune the models, thresholds or held-out predictions.

## Auxiliary workflow, HTTP and recipe evidence

The [eight-job receipt](validation-evidence/2026-10-08-relevance/aux-oct8/receipt.json)
records separate execution at source `7e6a7f5bc75f2e2a7e527304a816472947783e63`,
with the frozen core tree, script hashes, model/checkpoint identities and corpus
hashes retained. These measurements ran after relevance evaluation.

The [direct workflow comparison](validation-evidence/2026-10-08-relevance/aux-oct8/dx.json)
used a deterministic BANKING development subset of **100 documents and 20
queries**, the same pinned MiniLM, mean pooling, normalization and length 128.
Document/query vectors agreed within `atol=2e-5`, `rtol=2e-4`, with maximum
absolute differences **2.98e-8/1.49e-8**. Top-10 ranked IDs, filtered ranked IDs,
filter eligibility and both implementations' reload rankings matched. Both
rejected the wrong immutable model identity.

Across three alternating-order repetitions, median document-encoding/indexing
plus query-encoding/search time was **0.7183 seconds for clmkit** versus **0.6356
seconds for direct Sentence Transformers/NumPy**: about **13% descriptive
overhead**, not a speedup. The clmkit timing includes full candidate fetch and
canonical tie sorting. Loading and persistence are outside this timing summary;
both DX reload checks occur in the same process. This tiny scripted comparison
does not measure human productivity or an isolated dependency footprint.

The [authenticated loopback HTTP report](validation-evidence/2026-10-08-relevance/aux-oct8/serving.json)
records **192/192 successful measured requests**, no failures or unstarted
requests, and an unauthenticated-request check. Four warmup requests are separate.

| Concurrent clients | Successful measured requests | Requests/second | Request p50 ms | Request p95 ms |
|---|---:|---:|---:|---:|
| 1 | 48 | 68.48 | 12.96 | 23.09 |
| 4 | 48 | 69.99 | 55.26 | 66.70 |
| 8 | 48 | 71.47 | 112.45 | 142.35 |
| 16 | 48 | 70.49 | 198.19 | 269.03 |

Throughput was roughly flat while latency rose. This was one short fixed
synthetic query, one ascending-order repetition, closed-loop clients and one CPU
server process on the same Codespace. Encoder forwards use a shared lock; worker
and numerical-library thread limits were four. This checks the observed serving
behavior, not maximum capacity, varied-query performance or a production SLA.

The recipe jobs built **all 8,006 BANKING gallery documents** with frozen MiniLM,
then rebuilt them with adapted seed 42 into a **separate fresh index directory**.
Independent query processes strictly reloaded each snapshot with its matching
immutable identity. The separate CLINC recipe built **all 15,000 in-scope
documents** and a fresh query process applied its matching sealed dense
threshold, **0.6381174325942994**. The receipt preserves commands, process IDs,
counts, ranked IDs/scores and identities; no action was executed.

Successful recipe execution is not semantic validation of every example. For
the retained synthetic lost-card query, frozen BANKING returned
`card_about_to_expire` first, whereas adapted returned `lost_or_stolen_card`.
That single demonstration is neither a quality estimate nor a reason to retune
after test exposure. The CLINC balance example was accepted; it does not add an
independent rejection-rate estimate to the held-out evaluation above.

## Review and release record

| Deliverable | Status | Required fields before claiming completion |
|---|---|---|
| Final result/claim review | **Passed within the documented CPU alpha scope** | [Independent review](RELEVANCE_REVIEW.md), including limitations and checker repair history |
| Immutable alpha and candidate checks | **Recorded separately** | The [v0.1.0a2 release record](https://github.com/ZouhairMudakka/clmkit/releases/tag/v0.1.0a2) is the authority for publication, exact tagged revision, hosted checks and downloadable artifacts; this study alone does not assert those gates passed. |

Further tuning after these test outcomes is exploratory. Preserve no-gain
findings, all seeds and failed/interrupted attempts. Qwen 4B/8B, CUDA,
Arabic–English retrieval, production reliability, private-customer accuracy,
business ROI and human developer productivity remain outside this evidence.
The [claims ledger](CLAIMS_LEDGER.md) governs public wording.
