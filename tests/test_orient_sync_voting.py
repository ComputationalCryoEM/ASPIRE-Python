import os
import os.path
import tempfile

import numpy as np
import pytest
from click.testing import CliRunner

from aspire.abinitio import CLOrient3D, CLSyncVoting
from aspire.commands.orient3d import orient3d
from aspire.source import Simulation
from aspire.utils import (
    Rotation,
    get_aligned_rotations,
    register_rotations,
    rots_to_clmatrix,
)
from aspire.volume import AsymmetricVolume

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")

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
    np.float64,
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
        vols=AsymmetricVolume(L=resolution, C=1, K=100).generate(),
        offsets=offsets,
        amplitudes=1,
        seed=0,
    )

    # Search for common lines over less shifts for 0 offsets.
    max_shift = 1 / resolution
    if src.offsets.all() != 0:
        max_shift = 0.25

    orient_est = CLSyncVoting(src, max_shift=max_shift)

    return src, orient_est


def test_build_clmatrix(source_orientation_objs):
    src, orient_est = source_orientation_objs

    # Build clmatrix estimate.
    orient_est.build_clmatrix()

    gt_clmatrix = rots_to_clmatrix(src.rotations, orient_est.n_theta)

    angle_diffs = abs(orient_est.clmatrix - gt_clmatrix) * 360 / orient_est.n_theta

    # Count number of estimates within 5 degrees of ground truth.
    within_5 = np.count_nonzero(angle_diffs < 10)
    within_5 += np.count_nonzero(angle_diffs > 350)

    # Check that at least 99% (70% with shifts) of estimates are within 5 degrees.
    tol = 0.99
    if src.offsets.all() != 0:
        tol = 0.70
    assert within_5 / angle_diffs.size > tol


def test_estimate_rotations(source_orientation_objs):
    src, orient_est = source_orientation_objs

    orient_est.estimate_rotations()

    # Register estimates to ground truth rotations and compute the
    # angular distance between them (in degrees).
    Q_mat, flag = register_rotations(orient_est.rotations, src.rotations)
    regrot = get_aligned_rotations(orient_est.rotations, Q_mat, flag)
    ang_dist = np.zeros(src.n, dtype=src.dtype)
    for i in range(src.n):
        ang_dist[i] = (
            Rotation.angle_dist(
                regrot[i],
                src.rotations[i],
                dtype=src.dtype,
            )
            * 180
            / np.pi
        )

    # Assert that mean angular distance is less than 1 degree (8 degrees with shifts).
    degree_tol = 1
    if src.offsets.all() != 0:
        degree_tol = 8
    assert np.mean(ang_dist) < degree_tol


def test_estimate_shifts(source_orientation_objs):
    src, orient_est = source_orientation_objs
    if src.offsets.all() != 0:
        pytest.xfail("Currently failing under non-zero offsets.")

    est_shifts = orient_est.estimate_shifts().T

    # Assert that estimated shifts are close to src.offsets
    assert np.allclose(est_shifts, src.offsets)


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
