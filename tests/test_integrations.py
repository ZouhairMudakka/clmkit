from __future__ import annotations

import numpy as np
import pytest

from clmkit import HashingEncoder


def test_sklearn_pipeline_classifies_topics(toy_examples) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("sklearn")
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    from clmkit.integrations.sklearn import EmbeddingTransformer

    texts = [e.positive for e in toy_examples]
    labels = [i // 4 for i in range(len(texts))]  # 5 topics x 4 docs
    pipe = make_pipeline(
        EmbeddingTransformer("hashing", encoder_kwargs={"dim": 1024}), LogisticRegression(max_iter=2000)
    )
    pipe.fit(texts, labels)
    assert pipe.score(texts, labels) == 1.0
    cloned = clone(pipe)  # get_params/set_params round-trip works
    assert cloned.get_params()["embeddingtransformer__encoder_kwargs"] == {"dim": 1024}
    out = EmbeddingTransformer(HashingEncoder(dim=64), dim=16).fit_transform(["a", "b"])
    assert out.shape == (2, 16)


def test_langchain_adapter_duck_typed() -> None:
    from clmkit.integrations.langchain import ClmkitEmbeddings

    emb = ClmkitEmbeddings("hashing", dim=32)
    docs = emb.embed_documents(["cats eat fish", "the sun"])
    q = emb.embed_query("what do cats eat")
    assert len(docs) == 2 and len(docs[0]) == 32 and len(q) == 32
    assert np.dot(docs[0], q) > np.dot(docs[1], q)
