import numpy as np
import pytest

from aspire.abinitio import CommonlineSDP
from aspire.nufft import backend_available
from aspire.source import Simulation
from aspire.utils import Rotation, mean_aligned_angular_distance, rots_to_clmatrix
from aspire.volume import AsymmetricVolume

RESOLUTION = [
    32,
    33,
]

OFFSETS = [
    None,  # Defaults to random offsets.
    0,
]

DTYPES = [
    np.float32,
    pytest.param(np.float64, marks=pytest.mark.expensive),
]


@pytest.fixture(params=RESOLUTION, ids=lambda x: f"resolution={x}")
def resolution(request):
    return request.param


@pytest.fixture(params=OFFSETS, ids=lambda x: f"offsets={x}")
def offsets(request):
    return request.param


@pytest.fixture(params=DTYPES, ids=lambda x: f"dtype={x}")
def dtype(request):
    return request.param


@pytest.fixture
def src_orient_est_fixture(resolution, offsets, dtype):
    """Fixture for simulation source and orientation estimation object."""
    src = Simulation(
        n=50,
        L=resolution,
        vols=AsymmetricVolume(L=resolution, C=1, K=100, seed=0).generate(),
        offsets=offsets,
        amplitudes=1,
        seed=0,
    )

    # Increase max_shift and set shift_step to be sub-pixel when using
    # random offsets in the Simulation. This improves common-line detection.
    max_shift = 0.20
    shift_step = 0.25

    # Set max_shift 1 pixel and shift_step to 1 pixel when using 0 offsets.
    if np.all(src.offsets == 0.0):
        max_shift = 1 / src.L
        shift_step = 1

    orient_est = CommonlineSDP(
        src, max_shift=max_shift, shift_step=shift_step, mask=False
    )

    return src, orient_est


def test_estimate_rotations(src_orient_est_fixture):
    src, orient_est = src_orient_est_fixture

    if backend_available("cufinufft") and src.dtype == np.float32:
        pytest.skip("CI on GPU fails for singles.")

    orient_est.estimate_rotations()

    # Register estimates to ground truth rotations and compute the
    # angular distance between them (in degrees).
    # Assert that mean aligned angular distance is less than 1 degrees.
    mean_aligned_angular_distance(orient_est.rotations, src.rotations, degree_tol=1)


def test_construct_S(src_orient_est_fixture):
    """Test properties of the common-line quadratic form matrix S."""
    src, orient_est = src_orient_est_fixture

    # Since we are using the ground truth cl_matrix there is no need to test with offsets.
    if src.offsets.all() != 0:
        pytest.skip("No need to test with offsets.")

    # Construct the matrix S using ground truth common-lines.
    gt_cl_matrix = rots_to_clmatrix(src.rotations, orient_est.n_theta)
    S = orient_est._construct_S(gt_cl_matrix)

    # Check that S is symmetric.
    np.testing.assert_allclose(S, S.T)

    # For uniformly distributed rotations the top eigenvalue should have multiplicity 3.
    # As such, we can expect that the top 3 eigenvalues will all be close in value to their mean.
    eigs = np.linalg.eigvalsh(S)
    eigs_mean = np.mean(eigs[:3])

    # Check that the top 3 eigenvalues are all within 10% of the their mean.
    np.testing.assert_array_less(abs((eigs[:3] - eigs_mean) / eigs_mean), 0.10)

    # Check that the next eigenvalue is not close to the top 3, ie. multiplicity is not greater than 3.
    np.testing.assert_array_less(0.25, abs((eigs[4] - eigs_mean) / eigs_mean))


def test_gram_matrix(src_orient_est_fixture):
    """Test properties of the common-line Gram matrix."""
    src, orient_est = src_orient_est_fixture

    # Since we are using the ground truth cl_matrix there is no need to test with offsets.
    if src.offsets.all() != 0:
        pytest.skip("No need to test with offsets.")

    # Construct a ground truth S to pass into Gram computation.
    gt_cl_matrix = rots_to_clmatrix(src.rotations, orient_est.n_theta)
    S = orient_est._construct_S(gt_cl_matrix)

    # Estimate the Gram matrix
    A, b = orient_est._sdp_prep()
    gram = orient_est._compute_gram_SDP(S, A, b)

    # Construct the ground truth Gram matrix, G = R @ R.T, where R = [R1, R2]
    # with R1 and R2 being the concatenation of the first and second columns
    # of all ground truth rotation matrices, respectively.
    rots = src.rotations
    R1 = rots[:, :, 0]
    R2 = rots[:, :, 1]
    R = np.concatenate((R1, R2))
    gt_gram = R @ R.T

    # We'll check that the RMSE is within 10% of the mean value of gt_gram
    rmse = np.sqrt(np.mean((gram - R @ R.T) ** 2))
    np.testing.assert_array_less(rmse / np.mean(gt_gram), 0.10)


def test_ATA_solver():
    # Generate some rotations.
    seed = 42
    n_rots = 73
    dtype = np.float32
    rots = Rotation.generate_random_rotations(n=n_rots, seed=seed, dtype=dtype).matrices

    # Create a simple reference linear transformation A that is rank-3.
    A_ref = np.diag([1, 2, 3]).astype(dtype, copy=False)

    # Create v1 and v2 such that A_ref*v1=R1 and A_ref*v2=R2, R1 and R2 are the first
    # and second columns of all rotations.
    R1 = rots[:, :, 0].T
    R2 = rots[:, :, 1].T
    v1 = np.linalg.inv(A_ref) @ R1
    v2 = np.linalg.inv(A_ref) @ R2

    # Use ATA_solver to solve for A, given v1 and v2.
    A = CommonlineSDP._ATA_solver(v1, v2)

    # Check that A is close to A_ref.
    np.testing.assert_allclose(A, A_ref, atol=1e-7)


def test_deterministic_rounding(src_orient_est_fixture):
    """Test deterministic rounding, which recovers rotations from a Gram matrix."""
    src, orient_est = src_orient_est_fixture

    # Since we are using the ground truth cl_matrix there is no need to test with offsets.
    if src.offsets.all() != 0:
        pytest.skip("No need to test with offsets.")

    # Construct the ground truth Gram matrix, G = R @ R.T, where R = [R1, R2]
    # with R1 and R2 being the concatenation of the first and second columns
    # of all ground truth rotation matrices, respectively.
    gt_rots = src.rotations
    R1 = gt_rots[:, :, 0]
    R2 = gt_rots[:, :, 1]
    R = np.concatenate((R1, R2))
    gt_gram = R @ R.T

    # Pass the Gram matrix into the deterministic rounding procedure to recover rotations.
    est_rots = orient_est._deterministic_rounding(gt_gram)

    # Check that the estimated rotations are close to ground truth after global alignment.
    mean_aligned_angular_distance(est_rots, gt_rots, degree_tol=1e-5)
