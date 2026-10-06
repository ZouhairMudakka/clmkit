# Use cases and tested starter templates

Start with the workflow you need, then select an encoder and evaluate it on your
own data. clmkit supplies embedding and retrieval components; an application
supplies authentication, business rules, and any language-model generation.

| Use case | What clmkit provides | Starting point | Validation scope |
|---|---|---|---|
| Support FAQ or internal knowledge search | Ranked passages, metadata filters, persistent index | [Support search template](../templates/support_search.py) | Synthetic retrieval, source references, tenant scope and reload tests |
| Personal assistant memory | Store and retrieve facts, metadata scope, persistence | [Assistant memory template](../templates/assistant_memory.py) | Two-user separation and reload tests |
| Support request or tool routing | Route scoring, thresholds, abstention | [Support routing template](../templates/support_routing.py) | Known intents, unmatched requests and selection-only behavior |
| Retrieval-augmented generation (RAG) | Context retrieval and optional reranking before an LLM call | Support search template, then [RAG example](../examples/03_rag_pipeline.py) | Template tests cover retrieval; generated answers are not evaluated |
| Classification or clustering | Embedding features for an ML pipeline | [scikit-learn example](../examples/06_sklearn_pipeline.py) | Requires extra dependencies and a task-specific evaluation |
| Domain-specific embedding training | Pair loading, losses, hard negatives, LoRA and evaluation | [Training guide](TRAINING.md) | CPU contracts and Qwen 0.6B smoke checks; no demonstrated quality improvement |

The three templates use synthetic data and the **NumPy-only hashing encoder**.
It is a lexical pipeline baseline, not a trained semantic model. Passing these
tests establishes the demonstrated workflow behavior, not accuracy on real
customer questions, multilingual text, or paraphrases.

## Get the templates

These templates were added **after v0.1.0a1**. Use the current repository for the
template files and install the released core in a fresh virtual environment:

```bash
git clone https://github.com/ZouhairMudakka/clmkit.git
cd clmkit
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Then install and run:

```bash
python -m pip install "clmkit @ git+https://github.com/ZouhairMudakka/clmkit@v0.1.0a1"
python templates/support_search.py
python templates/assistant_memory.py
python templates/support_routing.py
```

Each command prints JSON, needs no API key or model download, and uses temporary
directories for demo snapshots. They neither contact a customer service nor
execute business actions. The Python scripts can be copied into another project;
they use public clmkit APIs and the standard library. Templates are repository
assets, not modules installed by the wheel. Source archives built after this
change include them; the existing v0.1.0a1 archives are unchanged.

## 1. Support knowledge search and RAG context

Use [support_search.py](../templates/support_search.py) for a help center,
employee handbook, or product documentation. It builds an index from synthetic
articles, queries one tenant's articles, and returns text plus stable source
references. Its demo saves and reloads the index using strict encoder checks.

Adapt `build_retriever(encoder)` to load your passages and their metadata. This
template expects `tenant_id`, `title`, `source_url`, and `revision` metadata keys.
Keep stable document IDs so a caller can trace every passage to its origin.
Call `search_support(...)` with a `trusted_tenant_id`
obtained from your application's authenticated session. A filter narrows search;
it does not authenticate the caller.

Expected behavior: the refund question selects the relevant refund article;
another tenant's similar article is excluded; an unknown tenant yields no
results; reloading preserves the retrieved IDs and source references. A minimum
score can reject weak matches, but a cosine score is not a probability that a
document answers the question.

For RAG, pass the selected passages and source references to your generation
model as untrusted context. This template stops at retrieval: it does not create
an answer, verify citations in generated prose, or decide authorization. An empty
result should trigger a clarification or explicit no-answer path in your app.

Before deploying: evaluate recall@k/MRR on held-out support questions, check that
retrieved sources support the answers, and test the no-answer policy. Evaluate
the full generation step separately if you add one.

## 2. Assistant memory scoped to a user

Use [assistant_memory.py](../templates/assistant_memory.py) to recall user
preferences or facts across sessions. `ScopedMemory` wraps `SemanticMemory` so
reads and writes require a `trusted_user_id`; callers cannot accidentally omit
the metadata scope through this wrapper. The demo stores synthetic facts for two
users and checks recall before and after saving a snapshot.

Expected behavior: each user sees only their own facts even when another user's
fact closely matches the question. A user with no stored memories gets an empty
list. Persistence preserves this separation. The wrapper assumes its trusted ID
comes from your server; accepting that ID from a model or arbitrary request body
would bypass that assumption.

Adapt the stored facts and choose what your app is allowed to remember. Decide
how users can inspect, correct, and delete memory. Retrieval similarity alone
does not decide whether a fact is still true or appropriate to store.

Use trusted snapshots with a single writer. Snapshot writes are not atomic and
this template is not a concurrent database. If you enable recency decay, it
reorders a bounded candidate pool; see the [agent guide](AGENTS.md) for that limit.

## 3. Support request and tool routing

Use [support_routing.py](../templates/support_routing.py) to shortlist billing,
account, or delivery handlers before an agent chooses its next action.
`build_router(encoder)` defines the example utterances and
`select_route(router, request)` returns a decision. An unmatched request falls
back to further clarification or human/LLM review.

Expected behavior: the demo's billing, account, and delivery requests select
their corresponding routes; unrelated and empty requests abstain. The threshold
is chosen for this tiny hashing demonstration. It is not a calibrated confidence
level and must be retuned when you change the encoder, utterances, or domain.
Pass your chosen value as `build_router(encoder, threshold=your_threshold)`;
omitting it runs the tiny demo calibration and raises an error if that dataset
cannot be separated. Search and memory recall also accept `min_score` overrides.

This template **selects a handler only**. It does not issue refunds, change
accounts, send messages, or call external tools. Keep authorization and any
required user confirmation in the application that executes those operations.

Before deploying: label representative requests, measure routing errors and
abstention rate, and include ambiguous and out-of-domain requests. Give expensive
or consequential actions a separate application-level decision check.

## Switch to a trained encoder

The template builders accept an `Encoder`. Install the `hf` extra, then pass a
trained model instead of the hashing baseline:

```bash
python -m pip install "clmkit[hf] @ git+https://github.com/ZouhairMudakka/clmkit@v0.1.0a1"
```

```python
from clmkit import load_encoder

encoder = load_encoder("Qwen/Qwen3-Embedding-0.6B")
# In a template: pass encoder to build_retriever, ScopedMemory, or build_router.
```

This downloads model weights and requires additional memory and disk space.
Existing Qwen 0.6B reference checks cover encoder behavior; **the three use-case
datasets have not been quality-benchmarked with that model**. Rebuild stored
vectors when changing encoders and retune thresholds on held-out examples.
For custom mutable weights, rebuild snapshots after weight changes or extend the
wrappers to manage explicit encoder identity using the underlying `Retriever`.
`ScopedMemory` does not expose an `encoder_identity` parameter. See the
[validation status](VALIDATION_STATUS.md) for fingerprint limits.

An OpenAI-compatible embedding endpoint can also supply vectors. Remote API
compatibility does not establish local CUDA correctness or GPU memory capacity.
See [serving and clients](../examples/07_serve_and_client.md) for setup.

## Run the template tests

With the released core installed and the current repository checked out:

```bash
python -m pip install pytest
python -m pytest -q tests/test_use_case_templates.py
```

Tests exercise application behavior, persistence, data scope, abstention, and
the scripts' JSON command-line outputs. Standard CI runs them across Linux,
Windows and macOS. The installed-wheel matrix also runs them outside the source
checkout with only NumPy and pytest installed, covering Linux Python 3.10–3.13
and Windows/macOS Python 3.13. Tests use no external model or paid service.

The test assertions are a starting point for your own acceptance dataset. Add
realistic failures and held-out queries before claiming semantic accuracy or
production readiness. Broader hardware and deployment limits remain in
[VALIDATION_STATUS.md](VALIDATION_STATUS.md).
