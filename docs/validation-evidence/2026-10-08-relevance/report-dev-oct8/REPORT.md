# Compact public benchmark evidence

**PARTIAL DEVELOPMENT REPORT — not held-out test evidence.**

Completed evaluation runs: 13/13.

## banking77

Same-intent support matching proxy; not verified duplicate tickets or retrieved resolutions.

| Method | Hit@1 | nDCG@10 | Recall@10 | Queries | Warm p95 (s) | Peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 0.7762 | 0.6505 | 0.0588 | 1997 | 0.0025 | 294.0742 |
| dense | 0.9129 | 0.8627 | 0.0815 | 1997 | 0.1393 | 664.2969 |
| hybrid | 0.8993 | 0.8319 | 0.0772 | 1997 | 0.1897 | 671.0273 |
| rerank | 0.9114 | 0.8662 | 0.0817 | 1997 | 0.1491 | 735.3242 |
| adapted-42 | 0.9254 | 0.8892 | 0.0845 | 1997 | 0.1514 | 662.9766 |
| adapted-1729 | 0.9244 | 0.8887 | 0.0845 | 1997 | 0.1418 | 662.4023 |
| adapted-2026 | 0.9239 | 0.8883 | 0.0844 | 1997 | 0.1442 | 662.1836 |

Queries overlapping training under the fixed grouping rule: 0. Official scores above retain them; separately labelled filtered scores are in report.json.

## clinc150

Supported-intent routing and explicit out-of-scope rejection; not document answer absence.

| Method | Hit@1 | nDCG@10 | Recall@10 | Queries | Warm p95 (s) | Peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 0.8320 | 0.7562 | 0.0735 | 3100 | 0.0035 | 337.0234 |
| dense | 0.8833 | 0.8533 | 0.0845 | 3100 | 0.1332 | 700.5469 |

Queries overlapping training under the fixed grouping rule: 35. Official scores above retain them; separately labelled filtered scores are in report.json.

- bm25: OOS false acceptance 5.00%; in-scope coverage 55.53%; accepted in-scope accuracy 0.936374549819928. All OOS and in-scope queries remain in their denominators.
- dense: OOS false acceptance 5.00%; in-scope coverage 86.13%; accepted in-scope accuracy 0.9214396284829721. All OOS and in-scope queries remain in their denominators.

## scifact

Retrieval of judged scientific evidence; not scientific truth or medical decision accuracy.

| Method | Hit@1 | nDCG@10 | Recall@10 | Queries | Warm p95 (s) | Peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 0.6111 | 0.7099 | 0.8282 | 162 | 0.0024 | 309.9922 |
| dense | 0.5556 | 0.6702 | 0.7731 | 162 | 0.1571 | 768.6133 |
| hybrid | 0.6173 | 0.7185 | 0.8287 | 162 | 0.1565 | 835.1562 |
| rerank | 0.6111 | 0.7047 | 0.7917 | 162 | 0.1396 | 766.1367 |

Queries overlapping training under the fixed grouping rule: 0. Official scores above retain them; separately labelled filtered scores are in report.json.

## Registered adaptation comparison

Mean over seeds [42, 1729, 2026]: Hit@1 change +1.17 percentage points; paired grouped 95% interval [+0.38, +1.97] points.
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

Checksum inventory: 47 artifacts, 62401988 bytes. Only hashes/counts and aggregate results are exported; no query texts or model weights.
