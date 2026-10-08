# Training guide

`clmkit` fine-tunes any `HFEncoder`, including Qwen3-Embedding 0.6B/4B/8B, E5, BGE and GTE, with contrastive objectives.

```bash
python -m pip install "clmkit[train,yaml] @ git+https://github.com/ZouhairMudakka/clmkit@v0.1.0a2"
clmkit train --config configs/qwen3-embedding-0.6b-full.yaml --set train.max_steps=20 --set encoder.device=cpu
```

Run recipe paths from the repository root; check out `v0.1.0a2` as described in
the [README](../README.md#install). The 0.6B recipe uses bundled sample data.
The 8B recipe requires your own data and its GPU capacity is unverified. For a
mining-to-training workflow, explicitly pass the mined file and held-out split:

```bash
clmkit mine --model Qwen/Qwen3-Embedding-0.6B --train train.jsonl --corpus corpus.txt --output train_hn.jsonl
clmkit train --config configs/qwen3-embedding-8b-lora.yaml --set data.train=train_hn.jsonl --set data.eval=eval.jsonl
clmkit eval --model runs/qwen3-embedding-8b-lora/final --data eval.jsonl
```

## 1. Data

One JSON object per line:

```json
{"query": "how do I reset my password",
 "positive": "Open Settings > Security and choose 'Reset password'.",
 "negatives": ["To change your email, open Settings > Account."],
 "instruction": "Given a support question, retrieve the help-center answer"}
```

| field | required | notes |
|---|---|---|
| `query` (alias `anchor`, `question`) | yes | |
| `positive` (alias `pos`, `document`) | yes | |
| `negatives` (alias `neg`, `hard_negatives`) | no | string or list; the batch uses `min(available, max_negatives)` per example |
| `instruction` | no | overrides the model's default query instruction (Qwen3 is instruction-aware) |
| `score` | no | graded label for `loss: cosent` |
| `label` | no | non-empty string identifying the shared relevance class of the query and positive; distinct from the numeric `score` |

Tips:

- **Write task-specific instructions** for Qwen3. The Qwen authors report 1–5% gains over no instruction. Write them in English even for multilingual data.
- **Mine hard negatives with a good model**, and filter false negatives. `clmkit mine` skips the positive, optionally skips the top-N, and drops candidates scoring above `0.95 × sim(query, positive)` (positive-aware mining).
- **Deduplicate.** Batches are built so the same query or positive never appears twice in a batch. A duplicate would become a false in-batch negative.
- Hold out an eval split (`data.eval` or `data.eval_fraction`) and pick checkpoints with `metric_for_best: ndcg@10`.

### Label-aware intent pairs

The label-aware options in this section are part of the `v0.1.0a2` alpha API.
Ordinary unlabeled examples
and explicit-negative training retain their existing behavior: `label` defaults
to `None`, and `avoid_same_label` defaults to `False`.

For intent matching, two different texts with the same label can form a positive
pair. Give the pair one string label:

```json
{"query": "I lost my payment card", "positive": "How can I replace a missing card?", "label": "lost_card"}
```

With an already configured trainable `encoder`, enable label-aware sampling:

```python
from clmkit.data import load_examples
from clmkit.training import ContrastiveTrainer, TrainConfig

examples = load_examples("intent-train.jsonl")
trainer = ContrastiveTrainer(
    encoder,
    TrainConfig(
        output_dir="runs/intent",
        batch_size=32,
        mini_batch_size=8,
        avoid_same_label=True,
        max_negatives=0,
    ),
    examples,
)
```

Every example must have a label. The sampler permits at most one example per
label across the **whole effective loss batch**, including all GradCache chunks.
It also keeps duplicate query/positive texts out of that batch, even if
`avoid_duplicates=False`. Direct calls to `training_step` validate these
conditions before computing gradients. There may be fewer examples than the
requested batch size when labels conflict; a singleton InfoNCE batch without
explicit negatives has no contrastive learning signal.

Use `max_negatives=0` for this workflow. Explicit negatives would enter every
query's candidate set, and their labels are not carried into the loss. The
trainer therefore rejects labelled examples containing explicit negatives when
`avoid_same_label=True`, unless `max_negatives=0` disables those negatives.
Labels alone do not enable this protection; the flag must be set. Do not label
all out-of-scope requests as one positive semantic class.

See the [support-intent recipes](../templates/intent_matching/README.md) for
training-gallery preparation, adapted checkpoint use, and full index rebuilding.
Their public dataset labels are relevance proxies; follow the
[evidence protocol](RELEVANCE_EVIDENCE.md) for split and evaluation controls.

### Blockwise mining and label exclusions

The Python mining API supports both label exclusions and independently bounded
query/corpus score blocks. Given labelled `examples` and `training_gallery`
records containing `text` and `label`, use:

```python
from clmkit.data import mine_hard_negatives

corpus = [row["text"] for row in training_gallery]
corpus_labels = [row["label"] for row in training_gallery]
mined = mine_hard_negatives(
    encoder,
    examples,
    corpus,
    corpus_labels=corpus_labels,
    num_negatives=4,
    batch_size=32,
    query_block_size=128,
    corpus_block_size=4096,
)
```

`corpus_labels` must align with the input corpus before deduplication. If any
example has a label, supply labels for every example and every corpus text.
Same-label candidates are excluded; conflicting labels for identical text are
rejected. Existing negatives must occur in the corpus with a known label
different from their own example's label. Use only the training gallery for
pair construction and mining; keep validation and test texts out of both.

Mining preserves the example's label, but it does **not** attach negative-label
metadata usable by the trainer. Its output cannot be used as explicit negatives
with label-aware training; `max_negatives=0` still disables them. The example
above is a mining call, not a change to that training restriction. Unlabeled
mining and training continue to support explicit negatives.

`batch_size` controls encoder batches. `query_block_size` and
`corpus_block_size` control score computation; their defaults are 128 and 4096.
This avoids a full query-by-corpus score matrix, but corpus embeddings remain
resident, along with the current query/positive block and retained candidates.
Measure total process memory as well as the score-buffer size.

Ties follow first occurrence in the deduplicated corpus. The miner inspects at
most `skip_top + 3 * num_negatives + 1` ranked candidates before exclusions, so
it may return fewer negatives than requested. The positive-aware score filter
and `skip_top` still apply. Aligned labels and block-size controls are Python
API options; the CLI mining command does not expose them.

## 2. Objectives

| `loss` | use when | key options (`loss_kwargs`) |
|---|---|---|
| `infonce` (default) | query/positive pairs, optional hard negatives | `temperature` (0.01–0.05), `false_negative_margin` (e.g. 0.1), `symmetric`, `in_batch_negatives` |
| `cosent` | pairs with graded similarity `score` (STS-style) | `scale` (20) |
| `triplet` | exactly one hard negative matters | `margin` |

`matryoshka_dims: [4096, 1024, 256]` trains the selected embedding widths jointly.
Choose dimensions supported by your model and evaluate each width separately;
fine-tuning does not guarantee that every truncated width retains its prior quality.

**Temperature.** Lower values sharpen the softmax. Values of 0.01–0.02 suit strong LLM-based embedders; 0.05 is a safe default for smaller models.

## 3. Batch size, GradCache and memory

Larger batches supply more in-batch negatives, but quality gains depend on the data
and activation memory grows with batch size. `mini_batch_size` enables **GradCache**:

1. embed the whole batch in chunks of `mini_batch_size` without keeping graphs;
2. compute the full-batch loss and its gradient w.r.t. every embedding;
3. re-embed each chunk with a graph and back-propagate the cached gradient.

Tiny-model tests check gradient equivalence with the full batch. GradCache reduces
retained activation memory by adding a no-gradient forward pass; full-batch
embeddings and the loss still consume memory. Actual time and peak memory depend
on the model, loss, chunk size and hardware. Enable `gradient_checkpointing: true`
on top for long sequences and measure the resulting tradeoff.

## 4. LoRA for 4B / 8B

```yaml
train:
  lora: {r: 16, alpha: 32, dropout: 0.05, target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]}
  learning_rate: 1.0e-4
  merge_lora_on_save: true
```

- Checkpoints saved during training (`checkpoint-N/`, `best/`) are **adapter-only**, a few MB. `load_encoder("runs/…/best")` loads the base model and applies the adapter automatically.
- `final/` is **merged** when `merge_lora_on_save: true`, so it loads as a plain checkpoint without `peft`. Merging writes a full copy of the weights, so check disk space.

### Hardware sizing (estimates)

| model | inference (bf16 weights) | LoRA fine-tune, seq 512, GradCache chunk 8 + grad checkpointing | full fine-tune |
|---|---|---|---|
| Qwen3-Embedding-0.6B | ~1.2 GB | ~4–6 GB | ~10–12 GB (AdamW fp32 states) |
| Qwen3-Embedding-4B | ~8 GB | ~12–16 GB | multi-GPU |
| Qwen3-Embedding-8B | ~16 GB | ~20–28 GB (24 GB is tight; 40–80 GB comfortable) | multi-GPU (not yet supported, see gap analysis) |

These are back-of-envelope numbers (weights + activations + LoRA optimizer state), not measurements. clmkit's own validation ran LoRA + GradCache on Qwen3-Embedding-0.6B on a 4-core CPU with 8 GB RAM.

## 5. Evaluate

```bash
clmkit eval --model runs/x/final --data eval.jsonl --ks 1,5,10
clmkit eval --model runs/x/final --beir path/to/beir/scifact --instruction "Given a scientific claim, retrieve documents that support or refute the claim"
```

Metrics follow trec_eval/BEIR definitions: nDCG (linear gain, log2 discount), MRR, Recall, MAP (normalised by all relevant documents).

## 6. Programmatic API

```python
from clmkit import load_encoder
from clmkit.data import load_examples, mine_hard_negatives
from clmkit.eval import RetrievalEvaluator
from clmkit.training import ContrastiveTrainer, TrainConfig, LoraSettings

encoder = load_encoder("Qwen/Qwen3-Embedding-8B", dtype="bfloat16", max_length=512)
train = mine_hard_negatives(encoder, load_examples("train.jsonl"), corpus, num_negatives=7)
trainer = ContrastiveTrainer(
    encoder,
    TrainConfig(output_dir="runs/8b", batch_size=64, mini_batch_size=8, max_negatives=7,
                lora=LoraSettings(r=16), matryoshka_dims=[4096, 1024, 256], gradient_checkpointing=True,
                loss_kwargs={"temperature": 0.02, "false_negative_margin": 0.1},
                eval_every=200, metric_for_best="ndcg@10"),
    train,
    evaluator=RetrievalEvaluator.from_examples(load_examples("eval.jsonl")),
    callbacks=[lambda rec: print(rec)],   # hook W&B / MLflow here
)
result = trainer.train()
```

Custom losses need only the call signature `(query, positive, negatives=None, scores=None) -> Tensor`. Pass one with `loss_fn=` or register it in `clmkit.LOSSES`.

## 7. Rebuild retrieval indexes after training

Fine-tuning changes the vector space. Re-encode every gallery document with the
adapted encoder into a new `Retriever` and a separate snapshot path before using
that encoder for retrieval. An index created with the earlier weights cannot be
made compatible by changing its model name or identity metadata.

For local or fine-tuned weights, supply an immutable `encoder_identity` at both
construction and reload. In this example, `checkpoint_manifest_sha256` is a
caller-computed digest of the checkpoint artifact manifest, covering the weights
and any adapters; `adapted_encoder`, `corpus`, `doc_ids` and `doc_metadata` are
the intended encoder and complete gallery:

```python
from clmkit import Retriever

identity = "sha256:" + checkpoint_manifest_sha256
rebuilt = Retriever(adapted_encoder, encoder_identity=identity)
rebuilt.add(corpus, ids=doc_ids, metadata=doc_metadata)
snapshot = rebuilt.save("indexes/intent-adapted-v2")
checked = Retriever.load(
    snapshot, adapted_encoder, encoder_identity=identity, strict=True
)
checked.validate_encoder()
```

The caller must generate and update the identity; the trainer does not produce
this manifest automatically, and the retriever does not verify arbitrary weight
changes by hashing tensors. The configuration fingerprint is a separate check.
Call `validate_encoder()` after changing a live encoder's settings or declared
identity; `add()` and `save()` also validate. Search does not recompute the
fingerprint for every request. Prefer a separate encoder instance for adaptation
so an active retriever keeps its original weights until replacement.

Validate the rebuilt snapshot and its retrieval behavior, then switch consumers
to it while retaining the previous complete snapshot. Saving these multi-file
snapshots is not an atomic replacement or a transactional deployment mechanism.
Python `Retriever.load` defaults to a warned permissive load (`strict=False`);
use `strict=True` as above to reject mismatches. See [Design](DESIGN.md) for the
configuration and legacy-snapshot contract.

## Training contracts and limits

Without `max_steps`, an epoch consumes every duplicate-aware batch. The scheduler
counts the same deterministic batch plans; this adds an initial batching pass.
Explicit `max_steps` is a step budget and may intentionally stop partway through
an epoch. Repeated duplicate groups can still make the sampler quadratic.
Singleton InfoNCE batches without hard negatives have no contrastive learning signal.

Non-finite loss or gradients raise before optimizer/scheduler updates and clear
gradient buffers. This does not roll back arbitrary custom encoder side effects.
Weighted Matryoshka dimensions retain their supplied weight pairing; dimensions
must be unique positive integers and weights finite, nonnegative with positive sum.

HF checkpoint loading requires safetensors. Saved weights/adapters are inference
checkpoints, not resumable optimizer/scheduler state. 4B/8B capacity, CUDA mixed
precision and multi-GPU training remain unverified. Training mechanics do not
establish a quality gain. See the [validation status](VALIDATION_STATUS.md) and
[recorded relevance evidence](validation-evidence/2026-10-08-relevance/README.md)
for the scope and limitations of each study. Recorded experiment source identities
remain distinct from the release version used to install the library.
