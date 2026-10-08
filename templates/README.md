# Tested use-case templates

Copy one of these scripts into your project or run it from the repository root:

```bash
python templates/support_search.py
python templates/assistant_memory.py
python templates/support_routing.py
python templates/retail_service_desk.py
```

Install clmkit first, using the
[setup and use-case guide](../docs/USE_CASES.md).
Use the `v0.1.0a2` alpha tag for both these files and the installed core.
The four commands above print JSON and need no API credentials or model downloads.

| Script | Reusable entry points | Demo checks |
|---|---|---|
| `retail_service_desk.py` | `RetailServiceDesk.prepare_case` | Business workflow: scoped order lookup, routing, policy evidence, preferences and escalation |
| `support_search.py` | `build_retriever`, `search_support` | Refund article with source reference, tenant scope, no-match response, strict reload |
| `assistant_memory.py` | `ScopedMemory` | Separate users' facts, required trusted scope, snapshot reload |
| `support_routing.py` | `build_router`, `select_route` | Billing/account/delivery selection, abstention, no actions executed |

Data in these four scripts is synthetic. Hashing is a lexical test baseline, not a semantic-quality
claim. The builders accept a clmkit `Encoder`; replacing it requires your own
quality evaluation and threshold calibration. Identity scope must come from an
authenticated application, not a model or an unverified request.

The two persistence demos use temporary directories and remove their snapshots
when finished. For an application, call `save` with a directory you manage;
snapshots require trusted inputs and one writer and are not atomic.

To run the behavior and command-line tests from a repository checkout:

```bash
python -m pip install pytest
python -m pytest -q tests/test_use_case_templates.py tests/test_retail_service_desk.py
```

The wheel supplies `clmkit`; these standalone recipes are separate repository
assets included in the `v0.1.0a2` source distribution. The full documentation is
available in the matching Git checkout. See the full guide for intended uses, expected results, customization,
and what the tests do not establish.

The [retail business guide](../docs/BUSINESS_USE_CASE.md)
also covers the business problem, customer journey, integration points and pilot
success measures.

## Public-data intent matching

The [intent-matching recipes](intent_matching/README.md) use full prepared
BANKING77/CLINC150 training galleries and a pinned model. They require HF
dependencies and the authorized Codespaces data-preparation workflow. BANKING77
labels are an intent relevance proxy; CLINC rejection uses its own sealed
development threshold. The recipes support adapted-checkpoint index rebuilding
and separate-process strict reload, and never execute business actions.

Read the [evidence protocol](../docs/RELEVANCE_EVIDENCE.md) and
[verification guide](../validation/README.md) before interpreting outputs.
Offline fixture tests establish mechanics, not semantic quality or business
accuracy. Supporting evidence scripts ship in the source distribution; datasets,
model weights and generated indexes do not.
