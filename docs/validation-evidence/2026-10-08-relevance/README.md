# Relevance study evidence — 8 October 2026

All **13 development comparisons, three training seeds and 13 held-out comparisons**
completed. The separate independent metric checker passed all 13 held-out runs,
and all eight real-model application checks passed. Read the
[results and costs](../../RELEVANCE_RESULTS.md) and
[independent review](../../RELEVANCE_REVIEW.md) for interpretation.

| Evidence | Contents |
|---|---|
| [Protocol](protocol.json), [seal](seal.json), [manifests](manifests/) | Fixed choices, model/source identities, dataset rights and split transformations |
| [Development report](report-dev-oct8/REPORT.md) and [JSON](report-dev-oct8/report.json) | All 13 development runs, three training seeds and interruption history |
| [Held-out report](report-test-oct8/REPORT.md) and [JSON](report-test-oct8/report.json) | All 13 test comparisons, grouped adaptation uncertainty, overlap sensitivity and costs |
| [Intent breakdown](banking77-intent-breakdown.json) | Descriptive per-intent BANKING77 scores and source result hashes |
| [Independent retry](independent-metrics-oct8-retry1.json) | Independent ranking/rejection formulas, coverage and artifact checks; 13/13 pass |
| [Original verification attempt](independent-metrics-oct8.json) | Seven successful comparisons, then a checker schema mismatch; preserved rather than overwritten |
| [Application receipt](aux-oct8/receipt.json), [DX](aux-oct8/dx.json), [HTTP](aux-oct8/serving.json) | Commands, source identities, parity/timing checks and full-gallery fresh-process recipes |
| [Verification procedures](verification/) | Dated tools for this recorded Codespaces layout, with explicit source/artifact guards |

The study source is `e61cc76d6cf52f25422335d42eaf18dde44e3067`. Auxiliary source
and immutable core-tree identities are recorded separately. The original metric
checker expected the BM25 calibration and result files to share their model-field
convention. The corrected checker explicitly validates each schema; results and
thresholds were unchanged. The interrupted development SciFact run and successful
retry remain in the suite history. Final packaging also corrected a false positive
for CLINC's numeric intent label `text`; raw dataset-text checks remain enforced.

The generated tables' CLINC query counts are total coverage: **3,100 development
inputs = 3,000 rankable in-scope + 100 OOS**, and **5,500 test inputs = 4,500 rankable
in-scope + 1,000 OOS**. Ranking excludes no-positive queries; rejection retains all
scope-labeled queries in its separate denominators. Scores are not probabilities.

Reranker table p95 measures retrieval only. Successful-run durations exclude the
unknown interrupted-attempt duration. The independent metric checker does not
independently recompute bootstrap draw quantiles or timing. HTTP results cover one
repeated embedding query in a short closed-loop test; they are not capacity/SLA
evidence. DX parity is not human productivity evidence.

Compact reports here contain no model weights or raw query/document text.
The [v0.1.0a2 release assets](https://github.com/ZouhairMudakka/clmkit/releases/tag/v0.1.0a2)
retain the ranked-ID/score archive, inventory and SHA256 checksums. Pinned data
preparation reproduces the omitted datasets; dataset licenses and notices remain
separate from clmkit's code license.
