import numpy as np
import pytest

from aspire.abinitio.commonline_sync3n import CLSync3N
from aspire.source import Simulation

DTYPE = np.float64  # TODO, consider single precision.
N = 64  # Number of images
n_pairs = N * (N - 1) // 2


# XXX TODO, conditionally run these only if GPU present.


@pytest.fixture
def src_fixture():
    src = Simulation(n=N, L=32, C=1, dtype=DTYPE)
    src = src.cache()
    return src


@pytest.fixture
def cl3n_fixture(src_fixture):
    cl = CLSync3N(src_fixture)
    return cl


@pytest.fixture
def rijs_fixture():
    Rijs = np.arange(n_pairs * 3 * 3).reshape(n_pairs, 3, 3)
    Rijs = Rijs.astype(dtype=DTYPE, copy=False)
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
    np.testing.assert_allclose(indsh, indscp)
    np.testing.assert_allclose(arbh, arbcp)


def test_triangle_scores_host_vs_cupy(cl3n_fixture, rijs_fixture):
    """
    Compares triangle_scores between host and cupy implementations.
    """
    # DTYPE is critical here (manually calling private method

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
    vec = np.random.random(n_pairs).astype(dtype=DTYPE, copy=False)

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
    vec = np.random.random(n_pairs).astype(dtype=DTYPE, copy=False)

    # J_weighting=True
    cl3n_fixture.J_weighting = True

    # Execute CUPY
    new_vec_cp = cl3n_fixture._signs_times_v_cupy(rijs_fixture, vec)

    # Execute host
    new_vec_h = cl3n_fixture._signs_times_v_host(rijs_fixture, vec)

    # Compare host to cupy calls
    np.testing.assert_allclose(new_vec_cp, new_vec_h)
