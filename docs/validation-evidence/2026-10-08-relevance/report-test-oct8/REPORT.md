# Compact public benchmark evidence

Sealed held-out test results.

Completed evaluation runs: 13/13.

## banking77

Same-intent support matching proxy; not verified duplicate tickets or retrieved resolutions.

| Method | Hit@1 | nDCG@10 | Recall@10 | Queries | Warm p95 (s) | Peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 0.7896 | 0.6546 | 0.0638 | 3080 | 0.0023 | 323.1445 |
| dense | 0.9179 | 0.8703 | 0.0904 | 3080 | 0.1260 | 681.2852 |
| hybrid | 0.9117 | 0.8355 | 0.0850 | 3080 | 0.1428 | 683.8203 |
| rerank | 0.9049 | 0.8682 | 0.0903 | 3080 | 0.1267 | 717.1680 |
| adapted-42 | 0.9289 | 0.8953 | 0.0940 | 3080 | 0.1329 | 678.7500 |
| adapted-1729 | 0.9260 | 0.8941 | 0.0939 | 3080 | 0.1220 | 679.2344 |
| adapted-2026 | 0.9266 | 0.8946 | 0.0938 | 3080 | 0.1329 | 679.2305 |

Queries overlapping training under the fixed grouping rule: 42. Official scores above retain them; separately labelled filtered scores are in report.json.

## clinc150

Supported-intent routing and explicit out-of-scope rejection; not document answer absence.

| Method | Hit@1 | nDCG@10 | Recall@10 | Queries | Warm p95 (s) | Peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 0.8318 | 0.7600 | 0.0739 | 5500 | 0.0033 | 399.8633 |
| dense | 0.8802 | 0.8509 | 0.0843 | 5500 | 0.1348 | 760.6562 |

Queries overlapping training under the fixed grouping rule: 52. Official scores above retain them; separately labelled filtered scores are in report.json.

- bm25: OOS false acceptance 6.70%; in-scope coverage 53.80%; accepted in-scope accuracy 0.919454770755886. All OOS and in-scope queries remain in their denominators.
- dense: OOS false acceptance 7.50%; in-scope coverage 87.67%; accepted in-scope accuracy 0.9125475285171103. All OOS and in-scope queries remain in their denominators.

## scifact

Retrieval of judged scientific evidence; not scientific truth or medical decision accuracy.

| Method | Hit@1 | nDCG@10 | Recall@10 | Queries | Warm p95 (s) | Peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 0.5433 | 0.6647 | 0.7849 | 300 | 0.0018 | 312.0820 |
| dense | 0.5033 | 0.6451 | 0.7833 | 300 | 0.1334 | 768.3867 |
| hybrid | 0.5467 | 0.6865 | 0.8179 | 300 | 0.1443 | 835.0195 |
| rerank | 0.5667 | 0.6804 | 0.8029 | 300 | 0.1387 | 768.5273 |

Queries overlapping training under the fixed grouping rule: 0. Official scores above retain them; separately labelled filtered scores are in report.json.

## Registered adaptation comparison

Mean over seeds [42, 1729, 2026]: Hit@1 change +0.93 percentage points; paired grouped 95% interval [+0.29, +1.56] points.
Outcome: **improved_below_practical_target**. Practical target: 2.00 points.

## Resource costs and incomplete work

Training, model loading, index rebuilding, retrieval and reranking costs are recorded separately in report.json. Warm p95 above measures retrieval only, including for reranked methods. Memory is process peak RSS.

- Training seed 42: 200 steps; 444.7 seconds.
- Training seed 1729: 200 steps; 434.7 seconds.
- Training seed 2026: 200 steps; 477.5 seconds.
- Missing: none.
- Recorded failures/timeouts: suite-dev: running (previous invocation); dev-scifact-rerank: running (previous invocation).

## Interpretation limits

- Development findings are parameter-selection evidence; only the sealed full test matrix is held-out evidence.
- Public benchmark exposure during model pretraining is unknown.
- CPU Codespaces timing is observational and host-dependent; single sequential runs do not establish an SLA.
- Warm latency quantiles use a small query sample; batched throughput is a separate measurement.
- Reranker component durations are measured separately; their sum is not a measured end-to-end request.
- Bootstrap intervals are conditional on the registered trained seeds and supplied duplicate groups.
- Overlap filtering is a sensitivity analysis under the documented lexical grouping rule, not a replacement score.
- CLINC rejection thresholds apply only to their registered gallery/model and do not validate BANKING77 rejection.
- No results establish private-customer accuracy, business ROI, developer productivity or production readiness.

Checksum inventory: 75 artifacts, 115669898 bytes. Only hashes/counts and aggregate results are exported; no query texts or model weights.
