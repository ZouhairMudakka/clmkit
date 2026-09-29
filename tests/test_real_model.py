"""Opt-in checks against real checkpoints (downloads weights).

    CLMKIT_TEST_MODEL=Qwen/Qwen3-Embedding-0.6B pytest -m slow tests/test_real_model.py
    CLMKIT_TEST_RERANKER=Qwen/Qwen3-Reranker-0.6B pytest -m slow tests/test_real_model.py

Reference numbers come from the official Qwen3-Embedding-0.6B model card, so this
verifies that clmkit's prompt format, EOS handling and pooling reproduce the
reference implementation exactly.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

MODEL = os.environ.get("CLMKIT_TEST_MODEL")

pytestmark = pytest.mark.slow

QUERIES = ["What is the capital of China?", "Explain gravity"]
DOCUMENTS = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects "
    "and is responsible for the movement of planets around the sun.",
]
# From https://huggingface.co/Qwen/Qwen3-Embedding-0.6B (transformers usage example).
QWEN3_06B_REFERENCE = np.array([[0.7645568251609802, 0.14142508804798126], [0.13549736142158508, 0.5999549627304077]])


@pytest.mark.skipif(not MODEL, reason="set CLMKIT_TEST_MODEL (e.g. Qwen/Qwen3-Embedding-0.6B)")
def test_reproduces_model_card_scores() -> None:
    pytest.importorskip("transformers")
    from clmkit import load_encoder

    enc = load_encoder(MODEL, device="cpu", dtype="float32")  # type: ignore[arg-type]
    scores = enc.encode(QUERIES, kind="query") @ enc.encode(DOCUMENTS).T
    if MODEL and "qwen3-embedding-0.6b" in MODEL.lower():
        np.testing.assert_allclose(scores, QWEN3_06B_REFERENCE, atol=2e-3)
    assert scores[0, 0] > scores[0, 1] and scores[1, 1] > scores[1, 0]
    # Matryoshka: a truncated embedding still ranks correctly
    small = enc.encode(QUERIES, kind="query", dim=128) @ enc.encode(DOCUMENTS, dim=128).T
    assert small[0, 0] > small[0, 1] and small[1, 1] > small[1, 0]


RERANKER = os.environ.get("CLMKIT_TEST_RERANKER")


@pytest.mark.skipif(not RERANKER, reason="set CLMKIT_TEST_RERANKER (e.g. Qwen/Qwen3-Reranker-0.6B)")
def test_llm_reranker_matches_reference_implementation() -> None:
    """clmkit's LLMYesNoReranker vs the Qwen3-Reranker model-card code, in fp32."""
    torch = pytest.importorskip("torch")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from clmkit.rerank import QWEN3_RERANK_PREFIX, QWEN3_RERANK_SUFFIX, LLMYesNoReranker

    tok = AutoTokenizer.from_pretrained(RERANKER, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(RERANKER, dtype=torch.float32).eval()
    task = "Given a web search query, retrieve relevant passages that answer the query"
    pre = tok.encode(QWEN3_RERANK_PREFIX, add_special_tokens=False)
    suf = tok.encode(QWEN3_RERANK_SUFFIX, add_special_tokens=False)
    pairs = [f"<Instruct>: {task}\n<Query>: {q}\n<Document>: {d}" for q in QUERIES for d in DOCUMENTS]
    enc = tok(
        pairs,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=8192 - len(pre) - len(suf),
    )
    batch = tok.pad({"input_ids": [pre + x + suf for x in enc["input_ids"]]}, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**batch).logits[:, -1, :]
    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    reference = torch.log_softmax(torch.stack([logits[:, no], logits[:, yes]], 1), 1)[:, 1].exp().numpy()

    rr = LLMYesNoReranker(model=model, tokenizer=tok, device="cpu")
    ours = np.concatenate([rr.score(q, DOCUMENTS, instruction=task) for q in QUERIES])
    np.testing.assert_allclose(ours, reference, atol=1e-6)
    assert ours[0] > 0.99 and ours[1] < 0.01  # matched vs mismatched pair
