import numpy as np
import pytest

from aspire.abinitio.commonline_sync3n import CLSync3N
from aspire.source import Simulation

# If cupy is not available, skip this entire module
pytest.importorskip("cupy")


N = 32  # Number of images
n_pairs = N * (N - 1) // 2
DTYPES = [np.float32, np.float64]


@pytest.fixture(scope="module", params=DTYPES, ids=lambda x: f"dtype={x}")
def dtype(request):
    return request.param


@pytest.fixture(scope="module")
def src_fixture(dtype):
    src = Simulation(n=N, L=32, C=1, dtype=dtype)
    src = src.cache()
    return src


@pytest.fixture(scope="module")
def cl3n_fixture(src_fixture):
    cl = CLSync3N(src_fixture)
    return cl


@pytest.fixture(scope="module")
def rijs_fixture(dtype):
    Rijs = np.arange(n_pairs * 3 * 3, dtype=dtype).reshape(n_pairs, 3, 3)
    return Rijs


def test_pairs_prob_host_vs_cupy(cl3n_fixture, rijs_fixture):
    """
    Compares pairs_probabilities  between host and cupy implementations.
    """

    P2, A, a, B, b, x0 = 1, 2, 3, 4, 5, 6

    # DTYPE is critical here (manually calling private method
    params = np.array([P2, A, a, B, b, x0], dtype=np.float64)

    # Execute CUPY
    indscp, arbcp = cl3n_fixture._pairs_probabilities_cupy(rijs_fixture, *params)

    # Execute host
    indsh, arbh = cl3n_fixture._pairs_probabilities_host(rijs_fixture, *params)

    # Compare host to cupy calls
    rtol = 1e-07  # np testing default
    if rijs_fixture.dtype != np.float64:
        rtol = 2e-5
    np.testing.assert_allclose(indsh, indscp, rtol=rtol)
    np.testing.assert_allclose(arbh, arbcp, rtol=rtol)


def test_triangle_scores_host_vs_cupy(cl3n_fixture, rijs_fixture):
    """
    Compares triangle_scores between host and cupy implementations.
    """

    # Execute CUPY
    hist_cp = cl3n_fixture._triangle_scores_inner_cupy(rijs_fixture)

    # Execute host
    hist_h = cl3n_fixture._triangle_scores_inner_host(rijs_fixture)

    # Compare host to cupy calls
    np.testing.assert_allclose(hist_cp, hist_h)


def test_stv_host_vs_cupy(cl3n_fixture, rijs_fixture):
    """
    Compares signs_times_v between host and cupy implementations.

    Default J_weighting=False
    """
    # dummy data vector
    vec = np.ones(n_pairs, dtype=rijs_fixture.dtype)

    # J_weighting=False
    assert cl3n_fixture.J_weighting is False

    # Execute CUPY
    new_vec_cp = cl3n_fixture._signs_times_v_cupy(rijs_fixture, vec)

    # Execute host
    new_vec_h = cl3n_fixture._signs_times_v_host(rijs_fixture, vec)

    # Compare host to cupy calls
    np.testing.assert_allclose(new_vec_cp, new_vec_h)


def test_stvJwt_host_vs_cupy(cl3n_fixture, rijs_fixture):
    """
    Compares signs_times_v between host and cupy implementations.

    Force J_weighting=True
    """
    # dummy data vector
    vec = np.ones(n_pairs, dtype=rijs_fixture.dtype)

    # J_weighting=True
    cl3n_fixture.J_weighting = True

    # Execute CUPY
    new_vec_cp = cl3n_fixture._signs_times_v_cupy(rijs_fixture, vec)

    # Execute host
    new_vec_h = cl3n_fixture._signs_times_v_host(rijs_fixture, vec)

    # Compare host to cupy calls
    rtol = 1e-7  # np testing default
    if vec.dtype != np.float64:
        rtol = 3e-07
    np.testing.assert_allclose(new_vec_cp, new_vec_h, rtol=rtol)


# The following fixture and tests compare against the legacy MATLAB implementation


@pytest.fixture
def matlab_ref_fixture():
    """
    Setup ASPIRE-Python objects using dummy data that is easily
    constructed in MATLAB.
    """
    DTYPE = np.float64  # MATLAB code is doubles only
    n = 5
    n_pairs = n * (n - 1) // 2

    # Dummy input vector.
    Rijs = np.transpose(
        np.arange(1, n_pairs * 3 * 3 + 1, dtype=DTYPE).reshape(n_pairs, 3, 3), (0, 2, 1)
    )
    # Equivalent MATLAB
    # n=5; np=n*(n-1)/2; rijs= reshape([1:np*3*3],[3,3,np])

    # Create CL object for testing function calls
    src = Simulation(L=8, n=n, C=1, dtype=DTYPE)
    cl3n = CLSync3N(src, seed=314, S_weighting=False, J_weighting=False)

    return Rijs, cl3n


def test_triangles_scores(matlab_ref_fixture):
    """
    Compares output of identical dummy data between this
    implementation and legacy MATLAB triangles_scores_mex.
    """
    Rijs, cl3n = matlab_ref_fixture

    hist = cl3n._triangle_scores_inner(Rijs)

    # Default is 100 histogram intervals,
    # so the histogram reference is compressed.
    ref_hist = np.zeros(cl3n.hist_intervals)
    # Nonzeros, [[indices, ...], [values, ...]]
    ref_compressed = np.array(
        [[0, 10, 11, 12, 70, 71, 72, 76, 81, 89], [14, 2, 2, 2, 1, 1, 2, 1, 2, 3]]
    )
    # Pack the reference histogram
    np.put(ref_hist, *ref_compressed)

    np.testing.assert_allclose(hist, ref_hist)


def test_pairs_prob_mex(matlab_ref_fixture):
    """
    Compares output of identical dummy data between this
    implementation and legacy MATLAB pairs_probabilities_mex.
    """
    Rijs, cl3n = matlab_ref_fixture

    params = np.arange(1, 7)

    ln_f_ind, ln_f_arb = cl3n._pairs_probabilities_host(Rijs, *params)

    ref_ln_f_ind = [
        -24.1817,
        -5.6554,
        4.9117,
        12.7047,
        -12.9374,
        -5.5158,
        1.5289,
        -9.0406,
        -2.2067,
        -7.3968,
    ]

    ref_ln_f_arb = [
        -17.1264,
        -6.7218,
        -0.8876,
        3.3437,
        -10.7251,
        -6.7051,
        -2.9029,
        -8.5061,
        -4.8288,
        -7.5608,
    ]

    np.testing.assert_allclose(ln_f_arb, ref_ln_f_arb, atol=5e-5)

    np.testing.assert_allclose(ln_f_ind, ref_ln_f_ind, atol=5e-5)


def test_signs_times_v_mex(matlab_ref_fixture):
    """
    Compares output of identical dummy data between this
    implementation and legacy MATLAB signs_times_v.
    """
    Rijs, cl3n = matlab_ref_fixture

    # Dummy input vector
    vec = np.ones(len(Rijs), dtype=Rijs.dtype)
    # Equivalent matlab
    # vec=ones([1,np]);

    new_vec = cl3n._signs_times_v(Rijs, vec)

    ref_vec = [0, -2, -2, 0, -6, -4, -2, -2, -2, 0]

    np.testing.assert_allclose(new_vec, ref_vec)
