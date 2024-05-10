import numpy as np
import pytest

from aspire.abinitio import CLSymmetryD2
from aspire.source import Simulation
from aspire.utils import J_conjugate, all_pairs, mean_aligned_angular_distance
from aspire.volume import DnSymmetricVolume, DnSymmetryGroup

##############
# Parameters #
##############

DTYPE = [np.float64, np.float32]
RESOLUTION = [48, 49]
N_IMG = [10]
OFFSETS = [0]
SEED = 42


@pytest.fixture(params=DTYPE, ids=lambda x: f"dtype={x}")
def dtype(request):
    return request.param


@pytest.fixture(params=RESOLUTION, ids=lambda x: f"resolution={x}")
def resolution(request):
    return request.param


@pytest.fixture(params=N_IMG, ids=lambda x: f"n images={x}")
def n_img(request):
    return request.param


@pytest.fixture(params=OFFSETS, ids=lambda x: f"offsets={x}")
def offsets(request):
    return request.param


############
# Fixtures #
############


@pytest.fixture
def source(n_img, resolution, dtype, offsets):
    vol = DnSymmetricVolume(
        L=resolution, order=2, C=1, K=100, dtype=dtype, seed=SEED
    ).generate()

    src = Simulation(
        n=n_img,
        L=resolution,
        vols=vol,
        offsets=offsets,
        amplitudes=1,
        seed=SEED,
    )

    return src


@pytest.fixture
def orient_est(source):
    orient_est = CLSymmetryD2(
        source,
        max_shift=0,
        shift_step=1,
        n_theta=360,
        n_rad=source.L,
        grid_res=350,  # Tuned for speed
        inplane_res=15,  # Tuned for speed
        eq_min_dist=10,  # Tuned for speed
        epsilon=0.01,
        seed=SEED,
    )

    return orient_est


#########
# Tests #
#########


def test_estimate_rotations(orient_est):
    # Estimate rotations.
    orient_est.estimate_rotations()
    rots_est = orient_est.rotations

    # Ground truth rotations.
    rots_gt = orient_est.src.rotations

    # g-sync ground truth rotations.
    rots_gt_sync = g_sync_d2(rots_est, rots_gt)

    # Register estimates to ground truth rotations and check that the
    # mean angular distance between them is less than 5 degrees.
    mean_aligned_angular_distance(rots_est, rots_gt_sync, degree_tol=5)


####################
# Helper Functions #
####################


def g_sync_d2(rots, rots_gt):
    """
    Every estimated rotation might be a version of the ground truth rotation
    rotated by g^{s_i}, where s_i = 0, 1, ..., order. This method synchronizes the
    ground truth rotations so that only a single global rotation need be applied
    to all estimates for error analysis.

    :param rots: Estimated rotation matrices
    :param rots_gt: Ground truth rotation matrices.

    :return: g-synchronized ground truth rotations.
    """
    assert len(rots) == len(
        rots_gt
    ), "Number of estimates not equal to number of references."
    n_img = len(rots)
    dtype = rots.dtype

    rots_symm = DnSymmetryGroup(2, dtype).matrices
    order = len(rots_symm)

    A_g = np.zeros((n_img, n_img), dtype=complex)

    pairs = all_pairs(n_img)

    for i, j in pairs:
        Ri = rots[i]
        Rj = rots[j]
        Rij = Ri.T @ Rj

        Ri_gt = rots_gt[i]
        Rj_gt = rots_gt[j]

        diffs = np.zeros(order)
        for s, g_s in enumerate(rots_symm):
            Rij_gt = Ri_gt.T @ g_s @ Rj_gt
            diffs[s] = min(
                [
                    np.linalg.norm(Rij - Rij_gt),
                    np.linalg.norm(Rij - J_conjugate(Rij_gt)),
                ]
            )

        idx = np.argmin(diffs)

        A_g[i, j] = np.exp(-1j * 2 * np.pi / order * idx)

    # A_g(k,l) is exp(-j(-theta_k+theta_l))
    # Diagonal elements correspond to exp(-i*0) so put 1.
    # This is important only for verification purposes that spectrum is (K,0,0,0...,0).
    A_g += np.conj(A_g).T + np.eye(n_img)

    _, eig_vecs = np.linalg.eigh(A_g)
    leading_eig_vec = eig_vecs[:, -1]

    angles = np.exp(1j * 2 * np.pi / order * np.arange(order))
    rots_gt_sync = np.zeros((n_img, 3, 3), dtype=dtype)

    for i, rot_gt in enumerate(rots_gt):
        # Since the closest ccw or cw rotation are just as good,
        # we take the absolute value of the angle differences.
        angle_dists = np.abs(np.angle(leading_eig_vec[i] / angles))
        power_g_Ri = np.argmin(angle_dists)
        rots_gt_sync[i] = rots_symm[power_g_Ri] @ rot_gt

    return rots_gt_sync
