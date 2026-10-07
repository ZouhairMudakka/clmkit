"""Numerical contracts for finite vectors across the float32 range."""

import numpy as np
import pytest

from clmkit.encoders.base import Encoder
from clmkit.index.numpy_index import NumpyIndex
from clmkit.utils import l2_normalize


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("magnitude", [1e20, np.finfo(np.float32).max])
def test_large_finite_vector_retains_unit_direction(single, magnitude):  # type: ignore[no-untyped-def]
    vector = np.array([magnitude, -magnitude], dtype=np.float32)
    actual = l2_normalize(vector if single else vector[None])
    expected = np.array([1, -1], dtype=np.float64) / np.sqrt(2)
    np.testing.assert_allclose(actual if single else actual[0], expected, rtol=1e-6)
    assert actual.dtype == np.float32


def test_mixed_scales_preserve_zero_and_epsilon_semantics() -> None:
    vectors = np.array([[1e20, 1e20], [3, 4], [0, 0], [1e-14, 0]], dtype=np.float32)
    expected = np.array([[2**-0.5, 2**-0.5], [0.6, 0.8], [0, 0], [0.01, 0]])
    np.testing.assert_allclose(l2_normalize(vectors), expected, rtol=1e-6)
    np.testing.assert_allclose(l2_normalize(vectors[-1], eps=1e-13), [0.1, 0], rtol=1e-6)
    assert l2_normalize(np.empty((0, 2))).shape == (0, 2)


def test_cosine_retrieval_is_invariant_to_large_finite_scale() -> None:
    index = NumpyIndex(2)
    index.add(["wrong", "right"], np.array([[0, 1e20], [1e20, 0]], dtype=np.float32))
    assert index.search(np.array([1e20, 0], dtype=np.float32), k=1)[0][0] == ("right", 1.0)
    np.testing.assert_allclose(Encoder.similarity(np.array([1e20, 0]), np.array([1e20, 0])), [[1.0]])
