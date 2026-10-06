# Tested use-case templates

Copy one of these scripts into your project or run it from the repository root:

```bash
python templates/support_search.py
python templates/assistant_memory.py
python templates/support_routing.py
```

Install clmkit first, using the
[setup and use-case guide](https://github.com/ZouhairMudakka/clmkit/blob/main/docs/USE_CASES.md).
These templates were added after v0.1.0a1 and work with that released core.
They print JSON and need no API credentials or model downloads.

| Script | Reusable entry points | Demo checks |
|---|---|---|
| `support_search.py` | `build_retriever`, `search_support` | Refund article with source reference, tenant scope, no-match response, strict reload |
| `assistant_memory.py` | `ScopedMemory` | Separate users' facts, required trusted scope, snapshot reload |
| `support_routing.py` | `build_router`, `select_route` | Billing/account/delivery selection, abstention, no actions executed |

All data is synthetic. Hashing is a lexical test baseline, not a semantic-quality
claim. The builders accept a clmkit `Encoder`; replacing it requires your own
quality evaluation and threshold calibration. Identity scope must come from an
authenticated application, not a model or an unverified request.

The two persistence demos use temporary directories and remove their snapshots
when finished. For an application, call `save` with a directory you manage;
snapshots require trusted inputs and one writer and are not atomic.

To run the behavior and command-line tests from a repository checkout:

```bash
python -m pip install pytest
python -m pytest -q tests/test_use_case_templates.py
```

The wheel supplies `clmkit`; these standalone recipes are separate repository
assets. See the full guide for intended uses, expected results, customization,
and what the tests do not establish.
