"""Contrastive embeddings as features in a classic ML pipeline (scikit-learn).

    pip install "clmkit[sklearn]"
    python examples/06_sklearn_pipeline.py
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from clmkit.integrations.sklearn import EmbeddingTransformer

texts = [
    "my card was charged twice",
    "I need an invoice for last month",
    "refund my order please",
    "why is my bill higher this month",
    "I can't log in to my account",
    "reset my password",
    "my account is locked after failed logins",
    "how do I enable two factor authentication",
    "the API returns a 429 error",
    "what is the rate limit for the REST API",
    "webhook deliveries are failing",
    "which SDK languages do you support",
]
labels = ["billing"] * 4 + ["security"] * 4 + ["developer"] * 4

# Swap "hashing" for "Qwen/Qwen3-Embedding-0.6B" (and kind="query" + an instruction) for real semantics.
pipe = make_pipeline(
    EmbeddingTransformer("hashing", encoder_kwargs={"dim": 1024}),
    LogisticRegression(max_iter=2000),
)
print("cv accuracy:", cross_val_score(pipe, texts, labels, cv=2).mean().round(3))
pipe.fit(texts, labels)
print(pipe.predict(["please send me a receipt", "two factor code not arriving", "API rate limit exceeded"]))
