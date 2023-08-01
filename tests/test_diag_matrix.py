"""
Tests for `DiagMatrix` class and interoperability with dense Numpy arrays.
"""

from itertools import product

import numpy as np
import pytest

from aspire.operators import DiagMatrix

MATRIX_SIZE = [
    32,
    33,
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
    # Internally convert dtype.  Passthrough will be checked explicitly in `test_dtype_passthrough` and `test_dtype_cast`
    d_np = np.random.random(shape).astype(dtype, copy=False)
    d1 = DiagMatrix(d_np[0])
    d2 = DiagMatrix(d_np[1])

    return d1, d2, d_np


# Explicit Tests (non parameterized).
def test_dtype_passthrough():
    """
    Test that the datatype is inferred correctly.
    """
    for dtype in (int, np.float32, np.float64, np.complex64, np.complex128):
        d_np = np.empty(42, dtype=dtype)
        d = DiagMatrix(d_np)
        assert d.dtype == dtype


def test_dtype_cast():
    """
    Test that a datatype is cast when overridden.
    """
    for dtype in (int, np.float32, np.float64, np.complex64, np.complex128):
        d_np = np.empty(42, dtype=np.float16)
        d = DiagMatrix(d_np, dtype)
        assert d.dtype == dtype


def test_conversion():
    """
    Test conversion between iterating over DiagMatrix.dense vs `np.diag`.
    """
    # Zero Dimension stack
    d_np = np.arange(42)  # defaults double
    A_np = np.diag(d_np)  # make dense with numpy
    # hrmm singleton
    np.testing.assert_allclose(DiagMatrix(d_np).dense, A_np)

    # One Dimension singleton
    d_np = np.arange(42).reshape(1, 42)
    diag = DiagMatrix(d_np).dense
    for i, d in enumerate(d_np):
        np.testing.assert_allclose(diag[i], np.diag(d))

    # One Dimension stack
    d_np = np.arange(2 * 42).reshape(2, 42)
    diag = DiagMatrix(d_np).dense
    for i, d in enumerate(d_np):
        np.testing.assert_allclose(diag[i], np.diag(d))

    # Two Dimension stack
    stack_shape = (2, 3)
    d_np = np.arange(np.prod(stack_shape) * 42).reshape(*stack_shape, 42)
    stackA = DiagMatrix(d_np).dense
    for i, j in product(range(stack_shape[0]), range(stack_shape[1])):
        np.testing.assert_allclose(stackA[i][j], np.diag(d_np[i][j]))


def test_stack_reshape_tuple(diag_matrix_fixture, stack):
    """
    Test stack reshape with tuple.
    """
    d1, _, _ = diag_matrix_fixture

    # This should be an no-op, but will excercise the code.
    x = d1.stack_reshape(stack)

    np.testing.assert_allclose(x, d1)


def test_diag_diag_add(diag_matrix_fixture):
    """
    Test addition.
    """
    d1, d2, d_np = diag_matrix_fixture

    np.testing.assert_allclose(d1 + d2, np.sum(d_np, axis=0))


def test_diag_diag_sub(diag_matrix_fixture):
    """
    Test subtraction.
    """
    d1, d2, d_np = diag_matrix_fixture

    np.testing.assert_allclose(d1 - d2, np.subtract(*d_np))


def test_diag_diag_mul(diag_matrix_fixture):
    """
    Test element-wise multiply of two `DiagMatrix` instances.
    """
    d1, d2, d_np = diag_matrix_fixture

    np.testing.assert_allclose(d1 * d2, np.multiply(*d_np))


def test_diag_diag_matmul(diag_matrix_fixture):
    """
    Test matrix multiply of two `DiagMatrix` instances.

    Note, this should be the equivalent of expanding to full dense
    matrices and computing the matrix multiply.  This is tested in a
    loop over the stack axes.
    """
    d1, d2, d_np = diag_matrix_fixture

    # compute the matmuls
    d = d1 @ d2

    _d1 = d1.stack_reshape(-1).asnumpy()
    _d2 = d2.stack_reshape(-1).asnumpy()

    for i, _d in enumerate(d.stack_reshape(-1).asnumpy()):
        np.testing.assert_allclose(_d, np.diag(np.diag(_d1[i]) @ np.diag(_d2[i])))


def test_neg(diag_matrix_fixture):
    """
    Test negation.
    """
    d1, _, d_np = diag_matrix_fixture

    np.testing.assert_allclose(-d1, -d_np[0])


def test_abs(diag_matrix_fixture):
    """
    Test absolute value method.
    """
    d1, _, d_np = diag_matrix_fixture

    # Compute reference via Numpy
    ref = np.abs(d_np[0])

    np.testing.assert_allclose(d1.abs(), ref)
    np.testing.assert_allclose(abs(d1), ref)


def test_pow(diag_matrix_fixture):
    """
    Test element-wise exponentiation.
    """
    d1, _, d_np = diag_matrix_fixture

    ref = d_np[0] ** 2
    np.testing.assert_allclose(d1**2, ref)
    np.testing.assert_allclose(d1.pow(2), ref)

    _d1 = d1
    d1 = d1.pow(2, inplace=True)
    np.testing.assert_allclose(d1, ref)
    assert d1 is _d1, "Object refs should be identical for inplace ops"


def test_norm(diag_matrix_fixture):
    """
    Test the `norm` compared to Numpy.
    """
    d1, _, d_np = diag_matrix_fixture

    # Expand to dense matrix and compute norm
    A = np.diag(d_np.reshape(-1, d_np.shape[-1])[0])
    ref = np.linalg.norm(A, 2)  # 2-norm

    np.testing.assert_allclose(d1.stack_reshape(-1).norm[0], ref)
    np.testing.assert_allclose(d1.norm.flatten()[0], ref)


def test_transpose(diag_matrix_fixture):
    """
    Test the transpose operations.

    Note the transpose operation is implemented for only for
    interoperability with other operators (ie `BlkDiagMatrix`).
    """
    d1, _, d_np = diag_matrix_fixture

    # Expand to dense matrix and compute transpose
    A = np.diag(d_np.reshape(-1, d_np.shape[-1])[0])
    ref = A.T.copy()

    np.testing.assert_allclose(d1.stack_reshape(-1).T.dense[0], ref)
    np.testing.assert_allclose(d1.stack_reshape(-1).transpose().dense[0], ref)


def test_ones(diag_matrix_fixture):
    """
    Test constructing a `DiagMatrix` initialized with ones.
    """
    _, _, d_np = diag_matrix_fixture
    d = DiagMatrix.ones(d_np.shape)

    np.testing.assert_allclose(d, 1)
    np.testing.assert_equal(d.shape, d_np.shape)


def test_zeros(diag_matrix_fixture):
    """
    Test constructing a `DiagMatrix` initialized with zeros.
    """
    _, _, d_np = diag_matrix_fixture
    d = DiagMatrix.zeros(d_np.shape)

    np.testing.assert_allclose(d, 0)
    np.testing.assert_equal(d.shape, d_np.shape)


def test_empty(diag_matrix_fixture):
    """
    Test constructing an uninitialized (empty) `DiagMatrix`.
    """
    _, _, d_np = diag_matrix_fixture
    d = DiagMatrix.empty(d_np.shape)

    np.testing.assert_equal(d.shape, d_np.shape)
