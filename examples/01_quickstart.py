"""Quickstart: lexical retrieval with a hashing baseline, no model download.

    python -m pip install "clmkit @ git+https://github.com/ZouhairMudakka/clmkit@v0.1.0a1"
    python examples/01_quickstart.py

Swap `HashingEncoder(...)` for `load_encoder("Qwen/Qwen3-Embedding-0.6B")` to use a real
contrastive model - nothing else changes.
"""

from clmkit import HashingEncoder, Retriever

retriever = Retriever(HashingEncoder(dim=512))
retriever.add(
    [
        "Contrastive learning pulls matching pairs together and pushes others apart.",
        "LoRA fine-tunes large models with small low-rank adapters.",
        "Qwen3-Embedding models produce vectors for semantic retrieval.",
        "Bananas are an excellent source of potassium.",
    ],
    metadata=[{"topic": "ml"}, {"topic": "ml"}, {"topic": "ml"}, {"topic": "food"}],
)

for hit in retriever.search("how does contrastive training work?", k=2):
    print(f"{hit.score:.3f}  {hit.text}")

print("\nfiltered to topic=food:")
for hit in retriever.search("healthy snack", k=1, filter={"topic": "food"}):
    print(f"{hit.score:.3f}  {hit.text}")
