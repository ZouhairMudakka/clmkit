# Hosted and Codespaces evidence

Collected October 7, 2026 in Dubai (UTC timestamps may show October 6).
The product source is unchanged from `472b3f3450cf829113fdb6a15702180d50b3f7fa`.

- `references/`: artifacts from [run 37527734885](https://github.com/ZouhairMudakka/clmkit/actions/runs/37527734885), checkout `9bb0bc2`. Both intended tests actually pass; the unrelated model test skips in each matrix job. Each folder records model revision, resolved packages, JUnit and the skip guard result.
- `contracts/`: artifacts from [run 37527635133](https://github.com/ZouhairMudakka/clmkit/actions/runs/37527635133), checkout `9bb0bc2`. Expected unresolved contracts fail; no xfail or success override is used.
- `codespace-training.json`: actual 4-core/16-GB Linux Codespace, Python 3.12.3, checkout `dc8858b`. One step/two synthetic pairs/rank-2 LoRA/FP32/chunk size one. Records source hashes, immutable model SHA, parameters, updates, resource use and reload errors. No quality-improvement claim.
- `pip-audit.json` and `resolved-requirements.txt`: clean Codespace environment before packaging-tool correction; setuptools 78.1.0 from the PyTorch CPU index has two distinct advisories (four duplicate records).
- `pip-audit-after.json`, `resolved-requirements-after.txt` and `update-setuptools.log`: after installing setuptools 84.0.0 from PyPI, no known vulnerabilities reported. Product dependency declarations were not changed.

Final configuration `71cc32e` has [ten passing ordinary CI jobs](https://github.com/ZouhairMudakka/clmkit/actions/runs/37529157417).
Its [independent workflow](https://github.com/ZouhairMudakka/clmkit/actions/runs/37529157324)
has seven passing jobs (six wheel environments plus MCP 1.x) and the failing
contract job. The earlier contract artifacts above retain detailed findings on
the same source. The project remains an unverified development preview.

Weights, adapters, virtual environments and account/billing details are excluded.
