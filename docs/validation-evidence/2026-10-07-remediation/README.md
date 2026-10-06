# Remediation evidence — 7 October 2026

Tested product commit: `7bb7a075054e79ce3ea0e58fa5c2c170fb0ff20d`.
Later documentation-only commits do not change the tested source or workflows.

The first runs below established behavior at `7bb7a07`. Their contract and reference
environments still included setuptools 78.1.0 from the CPU package index, although
the dedicated security/minimum jobs were clean. Workflow-only commit `234c294`
refreshes packaging tools in all CPU model jobs; its follow-up evidence is stored
under `refreshed/`. The older artifacts are retained with this qualification.

Final runs on `234c2942c256055a09e79eb0c2b241fe029ea153` all succeeded:
[CI 37533089986](https://github.com/ZouhairMudakka/clmkit/actions/runs/37533089986)
(10 jobs), [independent 37533090000](https://github.com/ZouhairMudakka/clmkit/actions/runs/37533090000)
(9 jobs), and [real models 37533195934](https://github.com/ZouhairMudakka/clmkit/actions/runs/37533195934)
(3 jobs). Final contract, minimum-stack and both reference requirement files record
setuptools 84.0.0. Their downloaded artifacts and refreshed training result have
separate digest/provenance entries in `refreshed/artifact-manifest.json`. The final
wheel jobs' API step outcomes are retained; their unchanged-product detailed
wheel artifacts below are from the initial remediation run, not the final rerun.

| Hosted run | Result |
|---|---|
| [CI 37532318169](https://github.com/ZouhairMudakka/clmkit/actions/runs/37532318169) | All 10 jobs passed: platform/core, full CPU, lint/types and security. |
| [Independent 37532318161](https://github.com/ZouhairMudakka/clmkit/actions/runs/37532318161) | All 9 jobs passed: six installed wheels, independent contracts, exact minimum ML stack and MCP 1.x. |
| [Real models 37532430967](https://github.com/ZouhairMudakka/clmkit/actions/runs/37532430967) | All 3 jobs passed: embedding, reranker and bounded LoRA/save-reload. |

Raw run/job API records are retained alongside downloaded artifact contents.
`artifact-manifest.json` records SHA-256 values verified against GitHub's displayed
artifact digests. Only JSON, XML, text and logs were extracted; model weights,
wheel binaries and sdists are not committed here. Original artifact archives
remain subject to GitHub's retention period.

The independent artifacts report 14 scientific cases and 26 core aggregate cases,
all passing. Core aggregation includes subtests: there are 20 test methods plus
six passing subtests. Protocol is 44/44; additional integrations are 10/10.
Minimum ML JUnit reports 236 passed and 11 optional skips, with slow tests excluded;
the exact stack is torch 2.13.0+cpu, transformers 5.17.0, PEFT 0.21.1 and safetensors
0.8.0. Other transitive packages are freshly resolved, not all pinned to their floors.
Each installed wheel JUnit reports 135 passed and 17 optional skips; two slow tests
are deselected. The fresh security and minimum-ML pip-audit steps succeeded.

Each real reference artifact contains one required pass and one unrelated skip,
with an explicit guard checking that the selected test passed. Embedding tolerance
is `atol=2e-3`, reranker `atol=1e-6` (both NumPy default `rtol=1e-7`). Training is
one rank-2 LoRA step on two synthetic pairs, with nonzero adapter changes and finite
parameters. Reload errors observed were zero; accepted tolerance was `atol=1e-5,
rtol=1e-4`. These are CPU execution checks, not benchmark quality or GPU evidence.

`local-junit.xml`, `local-tests.log`, `protocol.json` and `additional.json` preserve
the earlier Windows working-tree run: 253 passes, two real-model skips, six passing
subtests, protocol 44/0 and additional 10/0. That combined collection predates the
last Hub-pinning test and two shared-boundary guards; it must not be presented as
the final committed full-suite count. Focused checks and the exact-commit hosted
runs cover those additions. The local environment inherits system packages and
still has previously disclosed cryptography advisories; fresh hosted scans are
separate evidence.

Private vulnerability reporting was enabled and the GitHub API returned
`{"enabled": true}` on 7 October 2026. Secret scanning and push protection were
already enabled. No client production host or paid Codespace was used for this
remediation; the earlier personal Codespace remains stopped.
