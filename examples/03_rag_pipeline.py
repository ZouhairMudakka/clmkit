"""RAG building block: chunk -> embed -> index -> retrieve (-> rerank) -> prompt for any LLM.

    python examples/03_rag_pipeline.py                               # hashing encoder, no downloads
    python examples/03_rag_pipeline.py Qwen/Qwen3-Embedding-0.6B     # real model
    python examples/03_rag_pipeline.py Qwen/Qwen3-Embedding-0.6B Qwen/Qwen3-Reranker-0.6B

The retrieved context is printed as a ready-to-send prompt; plug in whichever LLM you use.
"""

import json
import sys
from pathlib import Path

from clmkit import Retriever, chunk_text, load_encoder
from clmkit.rerank import load_reranker

DOCS = Path(__file__).parent / "data" / "docs.jsonl"
model = sys.argv[1] if len(sys.argv) > 1 else "hashing"
reranker = load_reranker(sys.argv[2]) if len(sys.argv) > 2 else None

encoder = load_encoder(model, **({"dim": 1024} if model == "hashing" else {}))
retriever = Retriever(encoder, reranker=reranker)

texts, ids, metas = [], [], []
for line in DOCS.read_text(encoding="utf-8").splitlines():
    doc = json.loads(line)
    for i, chunk in enumerate(chunk_text(doc["text"], max_chars=400, overlap=50)):
        texts.append(f"{doc['title']}\n{chunk}")
        ids.append(f"{doc['id']}#{i}")
        metas.append({"source": doc["source"], "title": doc["title"]})
retriever.add(texts, ids=ids, metadata=metas)
print(f"indexed {len(retriever)} chunks\n")

question = "My password reset email never arrived, what should I do?"
hits = retriever.search(question, k=3)

context = "\n\n".join(f"[{h.id}] {h.text}" for h in hits)
prompt = (
    "Answer the question using only the context. Cite chunk ids in brackets.\n\n"
    f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)
print(prompt)

# Persist and reload (JSON + .npy only - no pickle)
retriever.save("examples/index")
print("\nreloaded:", len(Retriever.load("examples/index", encoder)), "chunks")
