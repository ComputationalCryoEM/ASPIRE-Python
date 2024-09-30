import copy
import os

import numpy as np
import pytest

from aspire.abinitio import CLSync3N
from aspire.source import Simulation
from aspire.utils import mean_aligned_angular_distance, rots_to_clmatrix
from aspire.volume import AsymmetricVolume

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")

RESOLUTION = [
    40,
    pytest.param(41, marks=pytest.mark.expensive),
]

OFFSETS = [
    0,
    pytest.param(None, marks=pytest.mark.expensive),
]

DTYPES = [
    np.float32,
    pytest.param(np.float64, marks=pytest.mark.expensive),
]


@pytest.fixture(params=RESOLUTION, ids=lambda x: f"resolution={x}", scope="module")
def resolution(request):
    return request.param


@pytest.fixture(params=OFFSETS, ids=lambda x: f"offsets={x}", scope="module")
def offsets(request):
    return request.param


@pytest.fixture(params=DTYPES, ids=lambda x: f"dtype={x}", scope="module")
def dtype(request):
    return request.param


@pytest.fixture(scope="module")
def source_orientation_objs(resolution, offsets, dtype):
    src = Simulation(
        n=100,
        L=resolution,
        vols=AsymmetricVolume(
            L=resolution, C=1, K=100, seed=123, dtype=dtype
        ).generate(),
        offsets=offsets,
        amplitudes=1,
        seed=456,
    ).cache()

    # Search for common lines over less shifts for 0 offsets.
    max_shift = 1 / resolution
    shift_step = 1
    if src.offsets.all() != 0:
        max_shift = 0.20
        shift_step = 0.25  # Reduce shift steps for non-integer offsets of Simulation.

    orient_est = CLSync3N(src, max_shift=max_shift, shift_step=shift_step, seed=789)

    # Estimate rotations once for all tests.
    orient_est.estimate_rotations()

    return src, orient_est


def test_build_clmatrix(source_orientation_objs):
    src, orient_est = source_orientation_objs

    gt_clmatrix = rots_to_clmatrix(src.rotations, orient_est.n_theta)

    angle_diffs = abs(orient_est.clmatrix - gt_clmatrix) * 360 / orient_est.n_theta

    # Count number of estimates within 5 degrees of ground truth.
    within_5 = np.sum((angle_diffs - 360) % 360 < 5)

    # Check that at least 98% of estimates are within 5 degrees.
    tol = 0.98
    if src.offsets.all() != 0:
        # Set tolerance to 75% when using nonzero offsets.
        tol = 0.75
    assert within_5 / angle_diffs.size > tol


def test_estimate_shifts_with_gt_rots(source_orientation_objs):
    src, orient_est = source_orientation_objs

    # Assign ground truth rotations.
    # Deep copy to prevent altering for other tests.
    orient_est = copy.deepcopy(orient_est)
    orient_est.rotations = src.rotations

    # Estimate shifts using ground truth rotations.
    est_shifts = orient_est.estimate_shifts()

    # Calculate the mean 2D distance between estimates and ground truth.
    error = src.offsets - est_shifts
    mean_dist = np.hypot(error[:, 0], error[:, 1]).mean()

    # Assert that on average estimated shifts are close (within 0.8 pix) to src.offsets
    if src.offsets.all() != 0:
        np.testing.assert_array_less(mean_dist, 0.8)
    else:
        np.testing.assert_allclose(mean_dist, 0)


def test_estimate_shifts_with_est_rots(source_orientation_objs):
    src, orient_est = source_orientation_objs
    # Estimate shifts using estimated rotations.
    est_shifts = orient_est.estimate_shifts()

    # Calculate the mean 2D distance between estimates and ground truth.
    error = src.offsets - est_shifts
    mean_dist = np.hypot(error[:, 0], error[:, 1]).mean()

    # Assert that on average estimated shifts are close (within 0.8 pix) to src.offsets
    if src.offsets.all() != 0:
        np.testing.assert_array_less(mean_dist, 0.8)
    else:
        np.testing.assert_allclose(mean_dist, 0)


def test_estimate_rotations(source_orientation_objs):
    src, orient_est = source_orientation_objs

    # Register estimates to ground truth rotations and compute the
    # mean angular distance between them (in degrees).
    # Assert that mean angular distance is less than 1 degree (4 with offsets).
    tol = 1
    if src.offsets.all() != 0:
        tol = 4
    mean_aligned_angular_distance(orient_est.rotations, src.rotations, degree_tol=tol)
