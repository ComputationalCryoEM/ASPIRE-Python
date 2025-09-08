import numpy as np
import pytest

from aspire.abinitio import (
    JSync,
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
    n_img = 20

    # Build outer products vijs, viis, and get ground truth third rows.
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


def test_J_sync_power_method(dtype):
    n = 25
    rots = Rotation.generate_random_rotations(n, dtype=dtype).matrices

    # Generate random signs [-1, 1].
    n_choose_2 = (n * (n - 1)) // 2
    signs = np.random.randint(0, 2, n_choose_2) * 2 - 1
    Rijs = np.zeros((n_choose_2, 3, 3), dtype=dtype)
    ij = 0
    for i in range(n - 1):
        Ri = rots[i]
        for j in range(i + 1, n):
            Rj = rots[j]
            Rij = Ri @ Rj.T
            if signs[ij] == -1:
                Rij = J_conjugate(Rij)
            Rijs[ij] = Rij
            ij += 1

    # Initialize JSync instance and perform power method.
    J_sync = JSync(n)
    signs_est = J_sync.power_method(Rijs)

    # signs_est should be correct up to multiplication by -1.
    # So we force signs/signs_est to start with +1.
    np.testing.assert_allclose(signs[0] * signs, signs_est[0] * signs_est)

    # Test dtype passthrough.
    assert signs_est.dtype == dtype
