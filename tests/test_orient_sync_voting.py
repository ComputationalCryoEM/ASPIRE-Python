import os
import os.path
import tempfile

import numpy as np
import pytest
from click.testing import CliRunner

from aspire.abinitio import CLOrient3D, CLSyncVoting
from aspire.commands.orient3d import orient3d
from aspire.noise import WhiteNoiseAdder
from aspire.source import Simulation
from aspire.utils import mean_aligned_angular_distance, rots_to_clmatrix
from aspire.volume import AsymmetricVolume

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")

RESOLUTION = [
    40,
    41,
]

# `None` defaults to random offsets.
OFFSETS = [
    0,
    pytest.param(None, marks=pytest.mark.expensive),
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
def source_orientation_objs(resolution, offsets, dtype):
    src = Simulation(
        n=50,
        L=resolution,
        vols=AsymmetricVolume(L=resolution, C=1, K=100, seed=0).generate(),
        offsets=offsets,
        amplitudes=1,
        seed=0,
    )

    # Search for common lines over less shifts for 0 offsets.
    max_shift = 1 / resolution
    shift_step = 1
    if src.offsets.all() != 0:
        max_shift = 0.20
        shift_step = 0.25  # Reduce shift steps for non-integer offsets of Simulation.
    orient_est = CLSyncVoting(
        src, max_shift=max_shift, shift_step=shift_step, mask=False
    )

    return src, orient_est


def test_build_clmatrix(source_orientation_objs):
    src, orient_est = source_orientation_objs

    # Build clmatrix estimate.
    orient_est.build_clmatrix()

    gt_clmatrix = rots_to_clmatrix(src.rotations, orient_est.n_theta)

    angle_diffs = abs(orient_est.clmatrix - gt_clmatrix) * 360 / orient_est.n_theta

    # Count number of estimates within 5 degrees of ground truth.
    within_5 = np.count_nonzero(angle_diffs < 5)
    within_5 += np.count_nonzero(angle_diffs > 355)

    # Check that at least 98% of estimates are within 5 degrees.
    tol = 0.98
    assert within_5 / angle_diffs.size > tol


def test_estimate_rotations(source_orientation_objs):
    src, orient_est = source_orientation_objs

    orient_est.estimate_rotations()

    # Register estimates to ground truth rotations and compute the
    # mean angular distance between them (in degrees).
    # Assert that mean angular distance is less than 1 degree.
    mean_aligned_angular_distance(orient_est.rotations, src.rotations, degree_tol=1)


def test_estimate_shifts(source_orientation_objs):
    src, orient_est = source_orientation_objs
    if src.offsets.all() != 0:
        pytest.xfail("Currently failing under non-zero offsets.")

    est_shifts = orient_est.estimate_shifts()

    # Assert that estimated shifts are close to src.offsets
    assert np.allclose(est_shifts, src.offsets)


def test_estimate_rotations_fuzzy_mask():
    noisy_src = Simulation(
        n=35,
        vols=AsymmetricVolume(L=128, C=1, K=400, seed=0).generate(),
        offsets=0,
        amplitudes=1,
        noise_adder=WhiteNoiseAdder.from_snr(snr=2),
        seed=0,
    )

    # Orientation estimation without fuzzy_mask.
    max_shift = 1 / noisy_src.L
    shift_step = 1
    orient_est = CLSyncVoting(
        noisy_src, max_shift=max_shift, shift_step=shift_step, mask=False
    )
    orient_est.estimate_rotations()

    # Orientation estimation with fuzzy mask.
    orient_est_fuzzy = CLSyncVoting(
        noisy_src, max_shift=max_shift, shift_step=shift_step
    )
    orient_est_fuzzy.estimate_rotations()

    # Check that fuzzy_mask improves orientation estimation.
    mean_angle_dist = mean_aligned_angular_distance(
        orient_est.rotations, noisy_src.rotations
    )
    mean_angle_dist_fuzzy = mean_aligned_angular_distance(
        orient_est_fuzzy.rotations, noisy_src.rotations
    )

    assert mean_angle_dist_fuzzy < mean_angle_dist < 10


def test_theta_error():
    """
    Test that CLSyncVoting when instantiated with odd value for `n_theta`
    gives appropriate error.
    """
    sim = Simulation()

    # Test we raise with expected error.
    with pytest.raises(NotImplementedError, match=r"n_theta must be even*"):
        _ = CLSyncVoting(sim, 16, 35)


def test_n_check_error():
    """Test we get expected error when n_check is out of range."""
    sim = Simulation()

    with pytest.raises(NotImplementedError, match=r"n_check must be in*"):
        _ = CLOrient3D(sim, n_check=-2)
    with pytest.raises(NotImplementedError, match=r"n_check must be in*"):
        _ = CLOrient3D(sim, n_check=sim.n + 1)


def test_command_line():
    # Ensure that the command line tool works as expected
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save the simulation object into STAR and MRCS files
        starfile_out = os.path.join(tmpdir, "save_test.star")
        starfile_in = os.path.join(DATA_DIR, "sample_particles_relion31.star")
        result = runner.invoke(
            orient3d,
            [
                f"--starfile_in={starfile_in}",
                "--n_rad=10",
                "--n_theta=60",
                "--max_shift=0.15",
                "--shift_step=1",
                f"--starfile_out={starfile_out}",
            ],
        )
        # check that the command completed successfully
        assert result.exit_code == 0
