"""
Tests for `DiagMatrix` class and interoperability with dense Numpy arrays.
"""

from itertools import product

import numpy as np
import pytest

from aspire.operators import BlkDiagMatrix, DiagMatrix
from aspire.utils import utest_tolerance

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
def blk_diag(matrix_size, dtype):
    """
    Construct a BlkDiagMatrix compatible in size with diag_matrix_fixture.
    """
    # block sizes are not important for testing, so long as we have a few.
    partition = [
        (2, 2),
    ] * (matrix_size // 2) + [
        (1, 1),
    ] * (matrix_size % 2)

    return BlkDiagMatrix.ones(partition, dtype=dtype)


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


def test_repr():
    """
    Test accessing the `repr` does not crash.
    """

    d = DiagMatrix(np.empty((10, 8)))
    assert repr(d).startswith("DiagMatrix(")


def test_str():
    """
    Test accessing the `str` does not crash.
    """

    d = DiagMatrix(np.empty((10, 8)))
    assert str(d).startswith("DiagMatrix(")


def test_get(diag_matrix_fixture):
    """
    Test accessing using the getter syntax.
    """
    d, _, d_np = diag_matrix_fixture

    ref = d_np[0][0]
    np.testing.assert_allclose(d[0], ref)

    assert str(d).startswith("DiagMatrix(")


def test_len():
    """
    Test the `len`.
    """
    d = DiagMatrix(np.empty((10, 8)))

    assert d.size == 10
    assert d.count == 8
    assert len(d) == 8


def test_size_mismatch():
    d1 = DiagMatrix(np.empty((10, 8)))
    d2 = DiagMatrix(np.empty((10, 7)))

    with pytest.raises(RuntimeError, match=r".*not same dimension.*"):
        _ = d1 + d2


def test_dtype_mismatch():
    d1 = DiagMatrix(np.empty((10, 8)), dtype=np.float32)
    d2 = DiagMatrix(np.empty((10, 8)), dtype=np.float64)

    with pytest.raises(RuntimeError, match=r".*received different types.*"):
        _ = d1 + d2


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
    np.testing.assert_allclose(DiagMatrix(d_np).dense(), A_np)

    # One Dimension singleton
    d_np = np.arange(42).reshape(1, 42)
    diag = DiagMatrix(d_np).dense()
    for i, d in enumerate(d_np):
        np.testing.assert_allclose(diag[i], np.diag(d))

    # One Dimension stack
    d_np = np.arange(2 * 42).reshape(2, 42)
    diag = DiagMatrix(d_np).dense()
    for i, d in enumerate(d_np):
        np.testing.assert_allclose(diag[i], np.diag(d))

    # Two Dimension stack
    stack_shape = (2, 3)
    d_np = np.arange(np.prod(stack_shape) * 42).reshape(*stack_shape, 42)
    stackA = DiagMatrix(d_np).dense()
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


def test_stack_reshape_bad_size(diag_matrix_fixture, stack):
    """
    Test stack reshape with tuple.
    """
    d1, _, _ = diag_matrix_fixture

    with pytest.raises(ValueError, match=r".*cannot be reshaped.*"):
        # attempt reshaping to a large prime
        _ = d1.stack_reshape(8675309)


def test_diag_diag_add(diag_matrix_fixture):
    """
    Test addition.
    """
    d1, d2, d_np = diag_matrix_fixture

    np.testing.assert_allclose(d1 + d2, np.sum(d_np, axis=0))


def test_diag_diag_scalar_add(diag_matrix_fixture):
    """
    Test addition.
    """
    d1, _, d_np = diag_matrix_fixture

    np.testing.assert_allclose(d1 + 123, d_np[0] + 123)


def test_diag_diag_scalar_radd(diag_matrix_fixture):
    """
    Test addition.
    """
    d1, _, d_np = diag_matrix_fixture

    np.testing.assert_allclose(123 + d1, d_np[0] + 123)


def test_diag_diag_sub(diag_matrix_fixture):
    """
    Test subtraction.
    """
    d1, d2, d_np = diag_matrix_fixture

    np.testing.assert_allclose(d1 - d2, np.subtract(*d_np))


def test_diag_diag_scalar_sub(diag_matrix_fixture):
    """
    Test subtraction.
    """
    d1, _, d_np = diag_matrix_fixture

    d1 = d1 - 123

    np.testing.assert_allclose(d1, d_np[0] - 123)


def test_diag_diag_scalar_rsub(diag_matrix_fixture):
    """
    Test subtraction.
    """
    d1, _, d_np = diag_matrix_fixture

    d1 = 123 - d1

    np.testing.assert_allclose(d1, 123 - d_np[0])


def test_diag_diag_mul(diag_matrix_fixture):
    """
    Test element-wise multiply of two `DiagMatrix` instances.
    """
    d1, d2, d_np = diag_matrix_fixture

    np.testing.assert_allclose(d1 * d2, np.multiply(*d_np))


def test_diag_scalar_mul(diag_matrix_fixture):
    """
    Test element-wise multiply of two `DiagMatrix` instances.
    """
    d1, _, d_np = diag_matrix_fixture

    # left mul
    np.testing.assert_allclose(d1 * 123, d_np[0] * 123)
    # right mul
    np.testing.assert_allclose(123 * d1, 123 * d_np[0])


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


def test_diag_blk_matmul(diag_matrix_fixture, blk_diag):
    """
    Test matrix multiply of `DiagMatrix` with `BlkDiagMatrix` instances.

    Note, this should be the equivalent of expanding to full dense
    matrices and computing the matrix multiply.
    """
    d, _, d_np = diag_matrix_fixture

    # Test we raise combining stacks with BlkDiagMatrix.
    if d.stack_shape != ():
        with pytest.raises(RuntimeError, match=r".*only implemented for singletons.*"):
            _ = d @ blk_diag

    else:
        # compute the matmuls
        A = d.dense()
        B = blk_diag.dense()
        AB = A @ B  # left mul reference
        BA = B @ A  # right mul reference

        np.testing.assert_allclose((d @ blk_diag).dense(), AB)
        np.testing.assert_allclose((blk_diag @ d).dense(), BA)


def test_diag_np_matmul(diag_matrix_fixture):
    """
    Test matrix multiply of `DiagMatrix` with `BlkDiagMatrix` instances.

    Note, this should be the equivalent of expanding to full dense
    matrices and computing the matrix multiply.
    """
    d1, d2, _ = diag_matrix_fixture

    if d1.stack_shape != ():
        with pytest.raises(ValueError, match=r".*only supports 2D.*"):
            _ = d1 @ d2.dense()

    else:
        # compute the reference matmuls
        A = d2.dense()
        DA = d1.dense() @ A
        AD = A @ d1.dense()

        np.testing.assert_allclose(d1 @ A, DA)
        np.testing.assert_allclose(A @ d1, AD)


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

    np.testing.assert_allclose(d1.stack_reshape(-1).T.dense()[0], ref)
    np.testing.assert_allclose(d1.stack_reshape(-1).transpose().dense()[0], ref)


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


def test_lr_scale(diag_matrix_fixture, blk_diag):
    """
    Test `lr_scale` method.
    """

    d1 = diag_matrix_fixture[0]

    # Check we raise combined stacks with BlkDiagMatrix.

    if d1.stack_shape != ():
        with pytest.raises(RuntimeError, match=r".*only implemented for singletons.*"):
            _ = d1.lr_scale(blk_diag.partition)
    else:
        # expand A @ A.T
        AA = d1.asnumpy()[:, None] @ d1.asnumpy()[None, :]
        ref = AA * blk_diag.dense()  # zero out off block elem

        # Test default unit weight scaling
        x = d1.lr_scale(blk_diag.partition)
        np.testing.assert_allclose(x.dense(), ref, atol=utest_tolerance(d1.dtype))

        # Test with different weights
        w = np.arange(d1.count, dtype=d1.dtype)
        x = d1.lr_scale(blk_diag.partition, weights=w)
        ref = (w * AA) * blk_diag.dense()  # zero out off block elem
        np.testing.assert_allclose(x.dense(), ref, atol=utest_tolerance(d1.dtype))


def test_bad_partition():
    """
    Test incorrect partitions raise an error.
    """
    d = DiagMatrix(np.empty(8))

    partition = [(9, 9)]
    with pytest.raises(RuntimeError, match=r".*count.*does not match.*"):
        _ = d.lr_scale(partition)

    partition = [(4, 4), (3, 4)]
    with pytest.raises(RuntimeError, match=r".*was not square.*"):
        _ = d.lr_scale(partition)


def test_as_blk_diag(matrix_size, blk_diag):
    """
    Test conversion from `DiagMatrix` to `BlkDiagMatrix`.

    Note this relies on `BlkDiagMatrix.dense`.
    """

    # Construct via Numpy.
    d_np = np.random.randn(matrix_size).astype(blk_diag.dtype)
    A = np.diag(d_np)

    # Create DiagMatrix then convert to BlkDiagMatrix
    d = DiagMatrix(d_np)
    B = d.as_blk_diag(blk_diag.partition)

    # Compare the dense B with the dense A.
    np.testing.assert_equal(B.dense(), A)


def test_bad_as_blk_diag(matrix_size, blk_diag):
    with pytest.raises(RuntimeError, match=r".*only implemented for singletons.*"):
        # Construct via Numpy.
        d_np = np.empty((2, matrix_size), dtype=blk_diag.dtype)

        # Create DiagMatrix then convert to BlkDiagMatrix
        d = DiagMatrix(d_np)
        _ = d.as_blk_diag(blk_diag.partition)


def test_eigs(diag_matrix_fixture):
    """
    Test the `eigvals` util method.
    """
    d, _, d_np = diag_matrix_fixture

    np.testing.assert_equal(d.eigvals(), d_np[0])


def test_eye(matrix_size, dtype):
    d = DiagMatrix.eye(matrix_size, dtype=dtype)

    np.testing.assert_equal(d, np.ones(matrix_size, dtype=dtype))


def test_apply(diag_matrix_fixture):
    d1, _, d_np = diag_matrix_fixture

    x = d1.apply(d_np)

    np.testing.assert_allclose(x, d_np[0][None, :] * d_np)


def test_rapply(diag_matrix_fixture):
    d1, _, d_np = diag_matrix_fixture

    x = d1.rapply(d_np)

    np.testing.assert_allclose(x, (d_np * d_np[0]))


def test_solve(diag_matrix_fixture):
    """
    Test `solve`.
    """
    a, _, d_np = diag_matrix_fixture

    b = np.arange(a.count, dtype=a.dtype)

    # Test we raise combining stacks with BlkDiagMatrix.
    if a.stack_shape != ():
        with pytest.raises(RuntimeError, match=r".*only implemented for singletons.*"):
            _ = a.solve(b)

    else:
        x = a.solve(b)

        np.testing.assert_allclose(a.dense() @ x, b, atol=utest_tolerance(a.dtype))


def test_diag_blk_mul(diag_matrix_fixture, blk_diag):
    """
    Test mixing `BlkDiagMatrix` with `DiagMatrix` element-wise multiplication.
    """
    d1, _, d_np = diag_matrix_fixture

    # Test we raise combining stacks with BlkDiagMatrix.
    if d1.stack_shape != ():
        with pytest.raises(RuntimeError, match=r".*only implemented for singletons.*"):
            _ = d1 * blk_diag

    else:
        # Compare commutation (tests rmul with blk)
        db = d1 * blk_diag
        bd = blk_diag * d1
        np.testing.assert_allclose(db.dense(), bd.dense())

        # Compare with numpy
        ref = d1.dense() * blk_diag.dense()
        np.testing.assert_allclose((d1 * blk_diag).dense(), ref)
