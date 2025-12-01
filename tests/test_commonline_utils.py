import numpy as np
import pytest

from aspire.abinitio import JSync
from aspire.abinitio.commonline_utils import (
    _complete_third_row_to_rot,
    _estimate_third_rows,
    build_outer_products,
)
from aspire.utils import J_conjugate, Rotation, randn, utest_tolerance

DTYPES = [np.float32, np.float64]


@pytest.fixture(params=DTYPES, ids=lambda x: f"dtype={x}", scope="module")
def dtype(request):
    return request.param


def test_estimate_third_rows(dtype):
    """
    Test we accurately estimate a set of 3rd rows of rotation matrices
    given the 3rd row outer products vijs =  vi @ vj.T and viis = vi @ vi.T.
    """
    n_img = 20

    # `build_outer_products` generates a set of ground truth 3rd rows
    # of rotation matrices, then forms the outer products vijs =  vi @ vj.T
    # and viis = vi @ vi.T.
    vijs, viis, gt_vis = build_outer_products(n_img, dtype)

    # Estimate third rows from outer products.
    # Due to factorization of V, these might be negated third rows.
    vis = _estimate_third_rows(vijs, viis)

    # Check if all-close up to difference of sign
    ground_truth = np.sign(gt_vis[0, 0]) * gt_vis
    estimate = np.sign(vis[0, 0]) * vis
    np.testing.assert_allclose(ground_truth, estimate, rtol=1e-05, atol=1e-08)

    # Check dtype passthrough
    assert vis.dtype == dtype


def test_complete_third_row(dtype):
    """
    Test that `complete_third_row_to_rot` produces a proper rotations
    given a set of 3rd rows.
    """
    # Build random third rows.
    r3 = randn(10, 3, seed=123).astype(dtype)
    r3 /= np.linalg.norm(r3, axis=1)[..., np.newaxis]

    # Set first row to be identical with z-axis.
    r3[0] = np.array([0, 0, 1], dtype=dtype)

    # Generate rotations.
    R = _complete_third_row_to_rot(r3)

    # Check dtype passthrough
    assert R.dtype == dtype

    # Assert that first rotation is the identity matrix.
    np.testing.assert_allclose(R[0], np.eye(3, dtype=dtype))

    # Assert that each rotation is orthogonal with determinant 1.
    assert np.allclose(
        R @ R.transpose((0, 2, 1)), np.eye(3, dtype=dtype), atol=utest_tolerance(dtype)
    )
    assert np.allclose(np.linalg.det(R), 1)


def test_J_sync(dtype):
    """
    Test that the J_sync `power_method` returns a set of signs indicating
    the set of relative rotations that need to be J-conjugated to attain
    global handedness consistency, and that `global_J_sync` returns the
    ground truth rotations up to a spurious J-conjugation.
    """
    n = 25
    rots = Rotation.generate_random_rotations(n, dtype=dtype).matrices

    # Generate ground truth and randomly J-conjugate relative rotations,
    # keeping track of the signs associated with J-conjugated rotations.
    n_choose_2 = (n * (n - 1)) // 2
    signs = np.random.randint(0, 2, n_choose_2) * 2 - 1
    Rijs_gt = np.zeros((n_choose_2, 3, 3), dtype=dtype)
    Rijs_conjugated = np.zeros((n_choose_2, 3, 3), dtype=dtype)
    ij = 0
    for i in range(n - 1):
        Ri = rots[i]
        for j in range(i + 1, n):
            Rj = rots[j]
            Rijs_gt[ij] = Rij = Ri.T @ Rj
            if signs[ij] == -1:
                Rij = J_conjugate(Rij)
            Rijs_conjugated[ij] = Rij
            ij += 1

    # Initialize JSync instance with default params.
    J_sync = JSync(n)

    # Perform power method and check that signs are correct up to
    # multilication by -1. Also check dtype pass-through.
    signs_est = J_sync.power_method(Rijs_conjugated)
    np.testing.assert_allclose(signs[0] * signs, signs_est[0] * signs_est)
    assert signs_est.dtype == dtype

    # Perform global J sync and check that rotations are correct up to
    # a spurious J conjugation. Also check dtype pass-through.
    Rijs_sync = J_sync.global_J_sync(Rijs_conjugated)

    # If the first is off by a J, J-conjugate the whole set.
    if np.allclose(Rijs_gt[0], J_conjugate(Rijs_sync[0])):
        Rijs_sync = J_conjugate(Rijs_sync)

    np.testing.assert_allclose(Rijs_sync, Rijs_gt)
    assert Rijs_sync.dtype == dtype
