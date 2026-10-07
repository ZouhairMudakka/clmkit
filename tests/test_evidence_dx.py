from __future__ import annotations

import numpy as np
import pytest
from validation.evidence_dx import deterministic_sample, exact_rank, load_reference, parity, run, save_reference


def test_reference_rank_ties_constraints_and_cosine() -> None:
    documents = np.asarray([[2, 0], [0, 4], [1, 0]], dtype=np.float32)
    query = np.asarray([[5, 0]], dtype=np.float32)
    assert exact_rank(documents, query, ["z", "b", "a"], 2) == [[("a", 1.0), ("z", 1.0)]]
    assert exact_rank(documents, query, ["z", "b", "a"], 2, {"b"}) == [[("b", 0.0)]]
    assert exact_rank(documents, query, ["z", "b", "a"], 2, set()) == [[]]
    with pytest.raises(ValueError, match="unique"):
        exact_rank(documents, query, ["a", "a", "b"], 1)


def test_reference_reload_requires_matching_configuration(tmp_path) -> None:
    vectors = np.eye(2, dtype=np.float32)
    documents = [{"id": "a", "metadata": {"owner": "alice"}}, {"id": "b", "metadata": {}}]
    config = {"revision": "pinned", "max_length": 128}
    save_reference(tmp_path, vectors, documents, config)
    restored_vectors, restored_documents = load_reference(tmp_path, config)
    np.testing.assert_array_equal(restored_vectors, vectors)
    assert restored_documents == documents
    with pytest.raises(ValueError, match="mismatch"):
        load_reference(tmp_path, {**config, "max_length": 256})


def test_remote_guard_precedes_model_import_or_data_access(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("CODESPACES", raising=False)
    with pytest.raises(RuntimeError, match="no local downloads"):
        run(tmp_path / "does-not-exist", revision="a" * 40)


def test_sample_and_vector_parity_are_deterministic() -> None:
    records = [{"id": str(i)} for i in range(10)]
    assert deterministic_sample(records, 3) == deterministic_sample(records[::-1], 3)
    assert len(deterministic_sample(records, 3)) == 3
    assert parity(np.eye(2), np.eye(2))["allclose"]
    assert not parity(np.eye(2), np.ones((2, 2)))["allclose"]
    assert not parity(np.eye(2), np.ones((3, 2)))["allclose"]
