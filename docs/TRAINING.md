# Training guide

`clmkit` fine-tunes any `HFEncoder`, including Qwen3-Embedding 0.6B/4B/8B, E5, BGE and GTE, with contrastive objectives.

```bash
pip install "clmkit[train,yaml]"
clmkit train --config configs/qwen3-embedding-8b-lora.yaml
clmkit train --config configs/qwen3-embedding-0.6b-full.yaml --set train.max_steps=20 --set encoder.device=cpu
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

Tips:

- **Write task-specific instructions** for Qwen3. The Qwen authors report 1–5% gains over no instruction. Write them in English even for multilingual data.
- **Mine hard negatives with a good model**, and filter false negatives. `clmkit mine` skips the positive, optionally skips the top-N, and drops candidates scoring above `0.95 × sim(query, positive)` (positive-aware mining).
- **Deduplicate.** Batches are built so the same query or positive never appears twice in a batch. A duplicate would become a false in-batch negative.
- Hold out an eval split (`data.eval` or `data.eval_fraction`) and pick checkpoints with `metric_for_best: ndcg@10`.

## 2. Objectives

| `loss` | use when | key options (`loss_kwargs`) |
|---|---|---|
| `infonce` (default) | query/positive pairs, optional hard negatives | `temperature` (0.01–0.05), `false_negative_margin` (e.g. 0.1), `symmetric`, `in_batch_negatives` |
| `cosent` | pairs with graded similarity `score` (STS-style) | `scale` (20) |
| `triplet` | exactly one hard negative matters | `margin` |

`matryoshka_dims: [4096, 1024, 256]` wraps the loss so truncated embeddings stay strong, which is useful for cheaper indexes. Qwen3-Embedding already supports MRL (32 up to the full dim), and fine-tuning with the same dims preserves that.

**Temperature.** Lower values sharpen the softmax. Values of 0.01–0.02 suit strong LLM-based embedders; 0.05 is a safe default for smaller models.

## 3. Batch size, GradCache and memory

In-batch negatives make **larger batches learn better** (more negatives per step), but activation memory grows with batch size. `mini_batch_size` enables **GradCache**:

1. embed the whole batch in chunks of `mini_batch_size` without keeping graphs;
2. compute the full-batch loss and its gradient w.r.t. every embedding;
3. re-embed each chunk with a graph and back-propagate the cached gradient.

The result is the same gradients as the full batch (asserted in the test suite), at the memory cost of one chunk, and roughly 1.3–1.5× the compute of a plain step. Enable `gradient_checkpointing: true` on top for long sequences.

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
precision, multi-GPU training and held-out quality improvement remain unverified.
