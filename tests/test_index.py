from __future__ import annotations

import json

import numpy as np
import pytest

from clmkit import NumpyIndex, load_index
from clmkit.utils import l2_normalize

rng = np.random.default_rng(0)


def _data(n: int = 200, d: int = 16) -> tuple[list[str], np.ndarray]:
    return [f"id{i}" for i in range(n)], l2_normalize(rng.normal(size=(n, d)).astype(np.float32))


@pytest.fixture(params=["numpy", "faiss-flat"])
def index_factory(request):  # type: ignore[no-untyped-def]
    if request.param == "numpy":
        return lambda d, metric="cosine": NumpyIndex(d, metric)
    pytest.importorskip("faiss")
    from clmkit.index.faiss_index import FaissIndex

    return lambda d, metric="cosine": FaissIndex(d, metric)


def test_topk_matches_bruteforce(index_factory) -> None:  # type: ignore[no-untyped-def]
    ids, vecs = _data()
    index = index_factory(16)
    index.add(ids, vecs)
    queries = l2_normalize(rng.normal(size=(5, 16)).astype(np.float32))
    results = index.search(queries, k=7)
    truth = np.argsort(-(queries @ vecs.T), axis=1)[:, :7]
    for row, expected in zip(results, truth):
        assert [i for i, _ in row] == [ids[j] for j in expected]
        scores = [s for _, s in row]
        assert scores == sorted(scores, reverse=True)


def test_add_validation(index_factory) -> None:  # type: ignore[no-untyped-def]
    index = index_factory(4)
    index.add(["a"], np.ones((1, 4)))
    with pytest.raises(ValueError, match="already in index"):
        index.add(["a"], np.ones((1, 4)))
    with pytest.raises(ValueError, match="duplicate"):
        index.add(["b", "b"], np.ones((2, 4)))
    with pytest.raises(ValueError, match="shape"):
        index.add(["c"], np.ones((1, 5)))
    with pytest.raises(ValueError, match="NaN"):
        index.add(["c"], np.full((1, 4), np.nan))
    with pytest.raises(ValueError, match="ids for"):
        index.add(["c", "d"], np.ones((1, 4)))
    with pytest.raises(TypeError):
        index.add([""], np.ones((1, 4)))
    assert len(index) == 1


def test_remove_filter_and_small_k(index_factory) -> None:  # type: ignore[no-untyped-def]
    ids, vecs = _data(20, 8)
    index = index_factory(8)
    index.add(ids, vecs)
    assert index.remove(["id0", "id1", "missing"]) == 2
    assert len(index) == 18 and "id0" not in index and "id2" in index
    hits = index.search(vecs[5], k=100)[0]
    assert len(hits) == 18 and hits[0][0] == "id5"
    allowed = {"id10", "id11", "id12"}
    filtered = index.search(vecs[5], k=5, allowed_ids=allowed)[0]
    assert {i for i, _ in filtered} == allowed
    assert index.search(vecs[5], k=3, allowed_ids=set())[0] == []
    with pytest.raises(ValueError):
        index.search(vecs[5], k=0)


def test_empty_index_and_dot_metric() -> None:
    index = NumpyIndex(3, metric="dot")
    assert index.search(np.ones(3), k=2) == [[]]
    index.add(["big", "small"], np.array([[10.0, 0, 0], [1.0, 0, 0]]))
    assert index.search(np.array([1.0, 0, 0]), k=1)[0][0] == ("big", 10.0)
    with pytest.raises(ValueError, match="metric"):
        NumpyIndex(3, metric="l2")  # type: ignore[arg-type]


def test_save_load_roundtrip_without_pickle(tmp_path, index_factory) -> None:  # type: ignore[no-untyped-def]
    ids, vecs = _data(30, 8)
    index = index_factory(8)
    index.add(ids, vecs)
    index.remove(["id3"])
    index.save(tmp_path / "idx")
    loaded = load_index(tmp_path / "idx")
    assert type(loaded) is type(index) and len(loaded) == 29
    q = vecs[:3]
    assert loaded.search(q, k=4) == index.search(q, k=4)
    loaded.add(["new"], vecs[3])  # ids continue correctly after reload
    assert loaded.search(vecs[3], k=1)[0][0][0] == "new"


def test_corrupt_numpy_index_is_rejected(tmp_path) -> None:
    index = NumpyIndex(4)
    index.add(["a", "b"], np.eye(2, 4))
    index.save(tmp_path)
    (tmp_path / "ids.json").write_text(json.dumps(["a"]))
    with pytest.raises(ValueError, match="corrupt"):
        NumpyIndex.load(tmp_path)
    # a pickled object array must not load
    np.save(tmp_path / "vectors.npy", np.array([object()], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError):
        NumpyIndex.load(tmp_path)


def test_hnsw_faiss_remove_rebuild() -> None:
    pytest.importorskip("faiss")
    from clmkit.index.faiss_index import FaissIndex

    ids, vecs = _data(50, 8)
    index = FaissIndex(8, factory="hnsw")
    index.add(ids, vecs)
    assert index.remove(["id0"]) == 1
    assert len(index) == 49
    assert index.search(vecs[7], k=1)[0][0][0] == "id7"
