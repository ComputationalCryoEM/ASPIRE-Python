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
    41,
]

# `None` defaults to random offsets.
OFFSETS = [
    0,
]

DTYPES = [
    np.float32,
    #    pytest.param(np.float64, marks=pytest.mark.expensive),
    np.float64,
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
        vols=AsymmetricVolume(L=resolution, C=1, K=100, seed=123).generate(),
        offsets=offsets,
        amplitudes=1,
        seed=456,
    ).cache()

    orient_est = CLSync3N(src, S_weighting=True, seed=789)

    return src, orient_est


def test_build_clmatrix(source_orientation_objs):
    src, orient_est = source_orientation_objs

    # Build clmatrix estimate.
    orient_est.build_clmatrix()

    gt_clmatrix = rots_to_clmatrix(src.rotations, orient_est.n_theta)

    angle_diffs = abs(orient_est.clmatrix - gt_clmatrix) * 360 / orient_est.n_theta

    # Count number of estimates within 5 degrees of ground truth.
    within_5 = np.sum((angle_diffs - 360) % 360 < 5)

    # Check that at least 98% of estimates are within 5 degrees.
    tol = 0.98
    if src.offsets.all() != 0:
        # Set tolerance to 95% when using nonzero offsets.
        tol = 0.95
    assert within_5 / angle_diffs.size > tol


def test_estimate_rotations(source_orientation_objs):
    src, orient_est = source_orientation_objs

    orient_est.estimate_rotations()

    # Register estimates to ground truth rotations and compute the
    # mean angular distance between them (in degrees).
    # Assert that mean angular distance is less than 1 degree.
    mean_aligned_angular_distance(orient_est.rotations, src.rotations, degree_tol=1)