"""Fine-tune a contrastive encoder on your own (query, positive, negatives) data.

    pip install "clmkit[train]"
    python examples/05_finetune.py                              # Qwen3-Embedding-0.6B, CPU ok (slow)
    python examples/05_finetune.py Qwen/Qwen3-Embedding-8B lora  # LoRA on a GPU

For production runs prefer the config-driven CLI: `clmkit train --config configs/...yaml`.
"""

import sys
from pathlib import Path

from clmkit import load_encoder
from clmkit.data import load_examples, mine_hard_negatives, split_examples
from clmkit.eval import RetrievalEvaluator
from clmkit.training import ContrastiveTrainer, LoraSettings, TrainConfig

model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-Embedding-0.6B"
use_lora = len(sys.argv) > 2 and sys.argv[2] == "lora"

examples = load_examples(Path(__file__).parent / "data" / "sample_train.jsonl")
train, held_out = split_examples(examples, eval_fraction=0.2, seed=0)

encoder = load_encoder(model, max_length=128)

# Optional: mine extra hard negatives with the *current* model (positive-aware filtering).
corpus = [e.positive for e in examples] + [n for e in examples for n in e.negatives]
train = mine_hard_negatives(encoder, train, corpus, num_negatives=2, max_relative_score=0.95)

evaluator = RetrievalEvaluator.from_examples(held_out, ks=[1, 5])
print("before:", evaluator(encoder))

config = TrainConfig(
    output_dir="runs/example-finetune",
    epochs=2,
    batch_size=8,
    mini_batch_size=4,  # GradCache: same result as batch 8, half the activation memory
    learning_rate=1e-4 if use_lora else 1e-5,
    max_negatives=2,
    loss_kwargs={"temperature": 0.05, "false_negative_margin": 0.1},
    lora=LoraSettings(r=8, alpha=16) if use_lora else None,
    log_every=2,
)
result = ContrastiveTrainer(encoder, config, train, evaluator=evaluator).train()
print("after:", result.eval)
print("saved to", Path(result.output_dir) / "final", "- load it with load_encoder(<that path>)")
