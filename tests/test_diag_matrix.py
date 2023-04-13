"""
Tests for `DiagMatrix` class and interoperability with dense Numpy arrays.
"""

from itertools import product

import numpy as np
import pytest

from aspire.operators import DiagMatrix

MATRIX_SIZE = [
    32,
]

DTYPES = [
    np.float64,
    np.float32,
]

STACKS = [
    (),
    (1,),
    (2,),
    (3, 4),
]


@pytest.fixture(params=DTYPES, ids=lambda x: f"dtype={x}")
def dtype(request):
    return request.param


@pytest.fixture(params=MATRIX_SIZE, ids=lambda x: f"count={x}")
def matrix_size(request):
    return request.param


@pytest.fixture(params=STACKS, ids=lambda x: f"stack={x}")
def stack(request):
    return request.param


@pytest.fixture
def diag_matrix_fixture(stack, matrix_size, dtype):
    """
    Generate some random diagonal matrix instance with stack, matrix_size and dtype.
    """
    shape = (2,) + stack + (matrix_size,)
    d_np = np.random.random(shape)
    # Internally convert dtype.  Passthrough will be checked explicitly in `test_dtype_passthrough`
    d1 = DiagMatrix(d_np[0], dtype)
    d2 = DiagMatrix(d_np[1], dtype)

    return d1, d2, d_np


# Explicit Tests (non parameterized).
def test_dtype_passthrough():
    for dtype in (int, np.float32, np.float64, np.complex64, np.complex128):
        d_np = np.empty(42, dtype=dtype)
        d = DiagMatrix(d_np)
        assert d.dtype == dtype


def test_dense_conversion():
    # Zero Dimension stack
    d_np = np.arange(42)
    A_np = np.diag(d_np)
    # hrmm singleton
    np.testing.assert_allclose(DiagMatrix(d_np).dense, A_np)

    # One Dimension singleton
    d_np = np.arange(42).reshape(1, 42)
    diag = DiagMatrix(d_np).dense
    for i, d in enumerate(d_np):
        np.testing.assert_allclose(diag[i], np.diag(d_np[i]))

    # One Dimension stack
    d_np = np.arange(2 * 42).reshape(2, 42)
    diag = DiagMatrix(d_np).dense
    for i, d in enumerate(d_np):
        np.testing.assert_allclose(diag[i], np.diag(d_np[i]))

    # Two Dimension stack
    stack_shape = (2, 3)
    d_np = np.arange(np.prod(stack_shape) * 42).reshape(*stack_shape, 42)
    stackA = DiagMatrix(d_np).dense
    for i, j in product(range(stack_shape[0]), range(stack_shape[1])):
        np.testing.assert_allclose(stackA[i][j], np.diag(d_np[i][j]))


def test_diag_diag_add(diag_matrix_fixture):
    d1, d2, d_np = diag_matrix_fixture

    np.allclose(d1 + d2, np.sum(d_np, axis=0))
