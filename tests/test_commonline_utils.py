import numpy as np
import pytest

from aspire.abinitio import (
    _complete_third_row_to_rot,
    _estimate_third_rows,
    build_outer_products,
)
from aspire.utils import randn, utest_tolerance

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
