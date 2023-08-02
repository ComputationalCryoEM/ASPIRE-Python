import numpy as np
import pytest

from aspire.abinitio import CLOrient3D, CLSyncVoting
from aspire.source import Simulation
from aspire.utils import rots_to_clmatrix
from aspire.volume import AsymmetricVolume

# Parametrize over (n_img, L, dtype)
PARAMS = [
    (50, 32, np.float32),
    (51, 32, np.float64),
    (50, 33, np.float32),
    (51, 33, np.float64),
]


def source_orientation_objs(n_img, L, dtype):
    src = Simulation(
        n=n_img,
        L=L,
        vols=AsymmetricVolume(L=L, C=1, K=100).generate(),
        offsets=0,
        amplitudes=1,
        seed=123,
    )

    orient_est = CLSyncVoting(src)
    return src, orient_est


@pytest.mark.parametrize("n_img, L, dtype", PARAMS)
def test_build_clmatrix(n_img, L, dtype):
    src, orient_est = source_orientation_objs(n_img, L, dtype)

    # Build clmatrix estimate.
    orient_est.build_clmatrix()

    gt_clmatrix = rots_to_clmatrix(src.rotations, orient_est.n_theta)

    angle_diffs = abs(orient_est.clmatrix - gt_clmatrix) * 360 / orient_est.n_theta

    # Count number of estimates within 5 degrees of ground truth.
    within_5 = np.count_nonzero(angle_diffs < 5)
    within_5 += np.count_nonzero(angle_diffs > 355)

    # Check that at least 99% of estimates are within 5 degrees.
    assert within_5 / angle_diffs.size > 0.99
