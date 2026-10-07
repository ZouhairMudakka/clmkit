# Retrieval features and reproducible evidence

The features below are **unreleased additions in the current source checkout**.
They are absent from the older audited install revision linked in the README.
This page documents APIs and the experiment protocol; it reports no new model
quality, speed, business savings or developer productivity result.

## Lexical and hybrid retrieval

```python
from clmkit import BM25Retriever, HashingEncoder, HybridRetriever

lexical = BM25Retriever(k1=1.5, b=0.75)
hybrid = HybridRetriever(HashingEncoder(dim=256), candidate_depth=100, rrf_k=60)
texts = ["SKU-X refund receipt", "SKU-Y delivery tracking"]
ids = ["receipt", "tracking"]
metadata = [{"owner": "alice", "sku": "SKU-X"}, {"owner": "bob", "sku": "SKU-Y"}]
for retriever in (lexical, hybrid):
    retriever.add(texts, ids=ids, metadata=metadata)
    hits = retriever.search("refund SKU-X", k=5, filter={"owner": "alice"})
    retriever.update("receipt", "SKU-X refund status")  # retains metadata
    retriever.delete(["tracking"])
```

Hashing is a no-download plumbing fixture, not a trained semantic encoder. Use a
pinned encoder evaluated for your task when testing semantic retrieval.

BM25 uses lowercased Unicode word tokens, positive Robertson IDF, no stemming or
stopword removal, and each query term once. Statistics cover the whole corpus;
filters restrict candidates without changing IDF. Empty or unknown lexical
queries return no results. Document updates/deletions rebuild lexical statistics.

Hybrid uses exact cosine retrieval over normalized vectors. Each branch retains
up to `max(k, candidate_depth)` eligible candidates, then sums
`1 / (rrf_k + rank)` with ranks starting at one. IDs are deduplicated; score ties
use ascending ID in both branches and fusion. A lexical zero-match query can
still return dense candidates. Scores are not calibrated probabilities.

A metadata equality mapping or predicate is evaluated once per document per
search call, and the same eligibility applies before either shortlist. Predicates
can express date/amount ranges and compound conditions; derive ownership scope
from authenticated application context. Text similarity does not enforce exact
SKU, amount or date constraints. Exact search avoids approximate-index underfill
but its cost still grows with the corpus.

These classes own their corpus; `get()` and result metadata return detached
copies. Change documents through `add`, `update` and `delete`. Hybrid re-encodes
only changed documents and stages updates before activation. Encoder configuration
changes require rebuilding; arbitrary in-place weight changes are not detected.
The API is currently Python-only, in-memory and single-writer: no hybrid snapshot,
CLI/REST integration, concurrent-write or crash-recovery guarantee is provided.

## Label-aware training and blockwise mining

```python
from clmkit.data import ContrastiveExample
from clmkit.training import TrainConfig

examples = [
    ContrastiveExample("card missing", "I lost my card", label="lost_card"),
    ContrastiveExample("transfer delayed", "My transfer has not arrived", label="pending_transfer"),
]
config = TrainConfig(
    output_dir="runs/intent",
    batch_size=32,
    mini_batch_size=8,
    avoid_same_label=True,
    max_negatives=0,
)
# ContrastiveTrainer(encoder, config, examples).train()
```

Supply a nonempty label for every query/positive pair. The sampler allows at most
one pair per label across the **entire effective contrastive batch**, including
all GradCache chunks, while avoiding repeated texts. It may produce smaller
batches when labels conflict. This is conservative sampling, not a multi-positive
loss. Labels can still be noisy; different intents are not guaranteed semantic
negatives. Keep validation/test examples outside the gallery and training pairs.

Explicit negatives are unsupported in this mode: use `max_negatives=0`. The
Python miner can exclude same-label corpus candidates, but its negative labels
are not carried into the training loss, so combining those mined negatives with
label-aware training remains unsupported.

```python
from clmkit.data import mine_hard_negatives

mined = mine_hard_negatives(
    encoder, examples, corpus_texts,
    corpus_labels=corpus_labels,  # aligned with corpus_texts
    query_block_size=128,
    corpus_block_size=4096,
    num_negatives=4,
)
```

Mining computes score blocks rather than a full query-by-corpus score matrix.
The full corpus embeddings remain resident; temporary merge/sort buffers and
model memory also count toward peak process memory. Ties use first corpus
occurrence. The miner inspects a bounded candidate pool before exclusions and
can return fewer negatives than requested. Labelled examples require aligned
corpus labels; conflicting labels for identical text are rejected. Block sizes
and corpus labels are Python API options, not new `clmkit mine` CLI flags.

## Strict model/index lifecycle

Use an immutable identity for local/fine-tuned weights and preserve it at load:

```bash
clmkit index --model ./checkpoint --encoder-identity sha256:YOUR_CHECKPOINT_DIGEST \
  --input documents.jsonl --output index-v2
clmkit search --index index-v2 --encoder-identity sha256:YOUR_CHECKPOINT_DIGEST \
  --strict-index "refund policy"
```

`--strict-index` is the CLI default for index loading in search, serve and MCP.
`--allow-encoder-mismatch` explicitly permits querying with a warning; it does not
make vectors compatible, and incompatible additions/saves remain rejected.
Python callers should use `Retriever.load(..., strict=True, encoder_identity=...)`.
Legacy snapshots lack the full encoding fingerprint. They remain readable with
warnings but must be rebuilt into a new retriever before adding documents or
saving, even when the model name and dimension match.
The identity is caller-supplied, not automatically computed by the CLI. After
training or changing vector conventions, rebuild into a separate directory,
validate it, then activate it while preserving the prior complete snapshot.
Multi-file snapshots are not transactional storage.

## Data provenance and interpretation

[`validation/evidence_data.py`](../validation/evidence_data.py) pins input artifacts,
verifies source hashes and writes checksummed prepared files plus manifests.
Existing frozen files cannot silently change. Retain source notices when sharing
derived data; clmkit's code license does not replace dataset licenses.

| Dataset | Pinned source and rights | What relevance means |
|---|---|---|
| BANKING77 | [PolyAI source](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b), [CC BY 4.0 notice](https://github.com/PolyAI-LDN/task-specific-datasets/blob/57ec275d8078af65b7731c2a98be812d844a6d6b/LICENSE) | Same-intent examples; not verified duplicate tickets, resolutions or production outcomes |
| CLINC150 full | [Source](https://github.com/clinc/oos-eval/tree/828f8093932c8fe6ca7936c3d2e52903b1c523de), [CC BY 3.0 notice](https://github.com/clinc/oos-eval/blob/828f8093932c8fe6ca7936c3d2e52903b1c523de/LICENSE) | Crowdworker intent matching and explicit out-of-scope rejection |
| BEIR SciFact | [BEIR archive/checksum catalog](https://github.com/beir-cellar/beir/wiki/Datasets-available), archive MD5 `5f7d1de60b170fc8027bb7898e2efca1`; [pinned original notices](https://github.com/allenai/scifact/blob/68b98a56d93e0f9da0d2aab4e6c3294699a0f72e/LICENSE.md) | Judged scientific evidence retrieval; claims/annotations CC BY 4.0, abstracts ODC-By 1.0, original code Apache-2.0 |

BANKING77 validation is derived from training with intent stratification and
token-multiset duplicate grouping. CLINC preserves official splits and excludes
OOS examples from its gallery. SciFact keeps the full corpus and multiple-positive
judgments; development claims come from grouped BEIR training claims. Duplicate
audits disclose the grouping rule's limits. Public model pretraining exposure is
unknown. These datasets do not establish performance on a customer's policies,
actual business records, Arabic queries or production infrastructure.

## Reproduce the development study

Run from this source checkout inside the authorized CPU Codespace, with its
environment active and `clmkit[train]` installed. The optional workflow comparison
also requires Sentence Transformers. Record exact installed versions, source
commit and any source changes. The scripts reject model/dataset work outside
`CODESPACES=true`; setting that flag locally does not authorize local downloads.
Do not use the client's production host.

```bash
python -m validation.evidence_run preflight
for dataset in banking77 clinc150 scifact; do
  python -m validation.evidence_data prepare --dataset "$dataset" \
    --output "/workspaces/evidence-data/$dataset"
done
python -m validation.evidence_run lock
python -m validation.evidence_suite pilot --max-seconds 3600
```

Preflight requires at least 3 GiB free disk before model loading. Review the
bounded pilot's resource use before proceeding. The protocol records model pins,
token limits, gallery manifests, methods, seeds and thresholds. Current defaults
use MiniLM, BANKING77 training seeds 42/1729/2026, 200 steps, no mined negatives,
top-100 retrieval, top-20 reranking and RRF with constant 60. Qwen is a separate
optional diagnostic, outside the first-wave registered method matrix.

After accepting a compute budget, run the sequential development suite. This
example bounds each invocation to one hour; `--resume` reuses matching completed
results, and individual evaluation/training jobs are capped at 45/60 minutes.

```bash
python -m validation.evidence_suite dev --max-seconds 3600 --resume
timeout 600s python -m validation.evidence_dx \
  --data-dir /workspaces/evidence-data/banking77 \
  --output /workspaces/evidence-results/dx.json \
  --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --max-length 128
```

The DX runner compares identical pinned MiniLM weights through clmkit and direct
Sentence Transformers plus NumPy on 100 corpus documents/20 development queries.
It checks vector/rank/filter parity and same-process save/reload/identity failure,
alternates warmed timings, and records workflow steps and installed versions.
It is not a human productivity study, isolated dependency-footprint measurement,
cold-start benchmark or fresh-process durability test. Export compact JSON reports,
not model caches or unreviewed dataset copies.

Test remains sealed until the complete development matrix, three registered
training runs, model identities and CLINC validation thresholds pass the seal:

```bash
python -m validation.evidence_run seal --output /workspaces/evidence-results/seal.json
python -m validation.evidence_suite test --max-seconds 3600 --resume
```

Review development results and freeze all choices **before** issuing these test
commands. The seal binds the protocol, dataset manifests, source, checkpoints,
thresholds and development results; changing them requires renewed development
validation. After viewing test results, further tuning is exploratory.

Report BANKING77 intent-proxy Hit@1, SciFact nDCG@10/Recall and CLINC rejection
separately. CLINC validation selects a threshold targeting at most 5% OOS false
acceptance; test behavior can differ and the threshold does not transfer to a
BANKING77 gallery. Keep missing predictions and no-match cases in their stated
denominators. Publish unsuccessful runs and no-gain outcomes, paired uncertainty,
seed variation and measured costs alongside any future result. The proposed
two-percentage-point BANKING77 adaptation target is a hypothesis, not a result.
