import numpy as np
import pytest

from aspire.numeric import numeric_object, sparse_object

# If cupy is not available, skip this entire test module
pytest.importorskip("cupy")

NUMERICS = ["numpy", "cupy"]


@pytest.fixture(params=NUMERICS, ids=lambda x: f"{x}", scope="module")
def backends(request):
    xp = numeric_object(request.param)
    sparse = sparse_object(request.param)
    return xp, sparse


def test_csr_matrix(backends):
    """
    Create csr_matrix and multiply with an `xp` array.
    """
    xp, sparse = backends

    m, n = 10, 10
    jdx = xp.arange(m)
    idx = xp.arange(n)
    vals = xp.random.random(10)

    # Compute dense matmul
    _A = np.diag(xp.asnumpy(vals))
    _B = np.random.random((n, 20))
    _C = _A @ _B

    # Compute matmul using sparse csr
    A = sparse.csr_matrix((vals, (jdx, idx)), shape=(m, n), dtype=np.float64)
    B = xp.array(_B)
    C = A @ B

    # Compare
    np.testing.assert_allclose(_C, xp.asnumpy(C))


def test_eigsh(backends):
    """
    Invoke sparse eigsh call with `xp` arrays.
    """
    xp, sparse = backends

    A = xp.eye(1234)

    lamb, _ = sparse.linalg.eigsh(A)
    np.testing.assert_allclose(xp.asnumpy(lamb), 1.0)
    print(lamb)
