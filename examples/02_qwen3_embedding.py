"""Use Qwen3-Embedding (0.6B / 4B / 8B) as a contrastive encoder, raw (no fine-tuning).

    pip install "clmkit[hf]"
    python examples/02_qwen3_embedding.py                       # 0.6B, runs on CPU
    python examples/02_qwen3_embedding.py Qwen/Qwen3-Embedding-8B  # needs a ~20 GB GPU

clmkit applies the model's conventions automatically: last-token pooling, left padding,
`<|endoftext|>` handling, and the `Instruct: ...\\nQuery:` prompt for queries only.
"""

import sys

from clmkit import load_encoder

model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-Embedding-0.6B"
encoder = load_encoder(model)  # device/dtype "auto": bf16 on CUDA, fp32 on CPU
print(encoder, "native dim:", encoder.native_dim)

queries = ["What is the capital of China?", "Explain gravity"]
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical "
    "objects and is responsible for the movement of planets around the sun.",
]

q = encoder.encode(queries, kind="query")  # instruction added automatically
d = encoder.encode(documents, kind="document")
print("full-dim scores:\n", (q @ d.T).round(4))

# Task-specific instruction (Qwen3 is instruction-aware; ~1-5% better with a good one)
q_code = encoder.encode("sort a list in python", kind="query", instruction="Given a question, retrieve code snippets")
print("custom-instruction query vector:", q_code.shape)

# Matryoshka: truncate to 256 dims (16x smaller index for 8B) and still rank correctly
q256 = encoder.encode(queries, kind="query", dim=256)
d256 = encoder.encode(documents, dim=256)
print("256-dim scores:\n", (q256 @ d256.T).round(4))
