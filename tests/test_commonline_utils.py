import numpy as np
import pytest

from aspire.abinitio import JSync, g_sync
from aspire.abinitio.commonline_utils import (
    _complete_third_row_to_rot,
    _estimate_third_rows,
    build_outer_products,
)
from aspire.utils import (
    J_conjugate,
    Rotation,
    mean_aligned_angular_distance,
    randn,
    utest_tolerance,
)
from aspire.volume import SymmetryGroup

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


@pytest.mark.parametrize("symmetry", ["C3", "C4", "D3", "D4", "T", "O"])
def test_g_sync(symmetry):
    n = 100
    dtype = np.float64

    # Get symmetry group matrices
    gs = SymmetryGroup.parse(symmetry).matrices

    # Build set of ground truth rotations
    gt_rots = Rotation.generate_random_rotations(n, dtype=dtype)

    # Build set of estimates which are close to ground truth
    # by generating set of small perturbation rotations to apply
    # to ground truth rotations.
    target_mean_deg = 2.0
    axes = np.random.normal(size=(n, 3)).astype(dtype)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    angles = np.random.uniform(0, 2 * np.deg2rad(target_mean_deg), n).astype(dtype)
    delta_rots = Rotation.from_rotvec(axes * angles[:, None], dtype=dtype)
    noisy_rots = Rotation(delta_rots.matrices @ gt_rots.matrices)

    # Get mean ang dist for aligned estimates
    # and check we're close to target.
    og_maad = mean_aligned_angular_distance(noisy_rots, gt_rots)
    np.testing.assert_array_less(abs(og_maad - target_mean_deg), 0.2)

    # Simulate symmetry desynchronization for clean and noisy case.
    g_idx = np.random.randint(len(gs), size=n)
    desynced_noisy_rots = Rotation(gs[g_idx] @ noisy_rots)
    desynced_clean_rots = Rotation(gs[g_idx] @ gt_rots.matrices)

    # Apply a global rotation to the noisy rotations to
    # to simulate a set of estimated rotations
    desynced_noisy_rots = (
        Rotation.generate_random_rotations(1, dtype=dtype).matrices
        @ desynced_noisy_rots
    )

    # Mean aligned angular distance of unsynced rots should be bad
    np.testing.assert_array_less(
        10 * og_maad, mean_aligned_angular_distance(desynced_noisy_rots, gt_rots)
    )

    # Perform g_sync and check that mean aligned angular distance
    # matches ground truth MAAD to within .1 degrees.
    rots_gt_synced_to_noisy = g_sync(desynced_noisy_rots, gt_rots, symmetry)
    est_maad = mean_aligned_angular_distance(
        desynced_noisy_rots, rots_gt_synced_to_noisy
    )
    np.testing.assert_array_less(abs(og_maad - est_maad), 0.1)

    # For the clean case the synced rotations should match allclose up to
    # a global multiplication by one of the symmetry group elements.
    gt_rots_synced_to_clean = g_sync(desynced_clean_rots, gt_rots, symmetry)
    errs = np.linalg.norm(
        gs @ gt_rots_synced_to_clean[0] - desynced_clean_rots[0], axis=(-2, -1)
    )
    best_g = np.argmin(errs)
    np.testing.assert_allclose(
        gs[best_g] @ gt_rots_synced_to_clean, desynced_clean_rots
    )
