# Contributing to clmkit

Thanks for helping! clmkit aims to stay **small, readable and correct**, so please keep changes focused.

## Development setup

```bash
git clone https://github.com/ZouhairMudakka/clmkit && cd clmkit
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU is enough for the test suite
pip install -e ".[dev,train,serve,mcp,faiss,sklearn]"
```

## Before opening a PR

```bash
ruff check src tests && ruff format src tests
mypy
pytest -q --cov=clmkit
```

- Add tests with every change. The suite needs no network or downloads: HF models are built from tiny configs (see `tests/conftest.py`). Real-model checks go in `tests/test_real_model.py` behind the `slow` marker and an environment variable.
- Keep `import clmkit` free of torch. Put torch/transformers imports inside functions or in modules registered lazily in `registry.py`. CI checks this.
- Don't add pickle-based persistence, `trust_remote_code=True` defaults, or network calls at import time.
- Public APIs need docstrings. User-facing changes go in `CHANGELOG.md`.

## Good first contributions

See [docs/GAP_ANALYSIS.md](docs/GAP_ANALYSIS.md#5-roadmap-prioritised). Self-contained items include:

- a `VectorIndex` adapter for pgvector / Qdrant / LanceDB;
- presets for new embedding models (with a reference-equivalence test);
- a BM25 sparse retriever and reciprocal-rank fusion;
- distillation losses (MarginMSE, KL from a reranker).

## Plugins without a PR

Third-party packages can register components via entry points, with no changes to clmkit:

```toml
[project.entry-points."clmkit.encoders"]
my-encoder = "my_pkg.encoders:MyEncoder"
```

Groups: `clmkit.encoders`, `clmkit.indexes`, `clmkit.rerankers`, `clmkit.losses`.

## Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
