import logging
import os

import numpy as np
import pytest
from numpy.linalg import norm
from numpy.random import normal

from aspire.utils.bot_align import align_BO
from aspire.utils.rotation import Rotation
from aspire.volume import Volume


def generate_data(snr, dtype):
    """
    Generates test data from MRC file.

    :param snr: Signal to noise ratio (0, float('inf') ).
        float('inf') would represent a clean image.
    :return: Reference Volume, Test Volume, size (pixels), True rotation
    """

    # Original author tested the code with `emd-3683.mrc`,
    #     and those should be within 1 degree up to 0.1 SNR.
    # v = Volume(mrcfile.open('emd-3683.mrc').data, dtype=dtype)
    # We will use a smaller volume already shipped with ASPIRE instead and looser constraints.
    vol_path = os.path.join(
        os.path.dirname(__file__), "saved_test_data", "clean70SRibosome_vol.npy"
    )
    v = Volume(np.load(vol_path), dtype=dtype)
    L = v.resolution
    shape = (L, L, L)
    ns_std = np.sqrt(norm(v) ** 2 / (L**3 * snr)).astype(v.dtype)
    reference_vol = v + normal(0, ns_std, shape)
    r = Rotation.generate_random_rotations(1, dtype=v.dtype)
    R_true = r.matrices[0]
    test_vol = v.rotate(r) + normal(0, ns_std, shape)

    return reference_vol, test_vol, L, R_true


def angular_dist_degrees(R1, R2):
    return Rotation.angle_dist(R1, R2) * (180) / np.pi


# Test the following parameters:
# loss type ('wemd' or 'eu')
# downsampling level (32 or 64 recommended)
# total number of iterations (150 or 200 recommended)

ALGO_PARAMS = [
    ["wemd", 32, 200],
    ["wemd", 64, 150],
    ["eu", 32, 200],
    ["eu", 64, 150],
]

SNRS = [float("inf"), 0.5]
DTYPES = [np.float32, np.float64]


@pytest.fixture(params=DTYPES, ids=lambda x: f"dtype={x}")
def dtype(request):
    return request.param


@pytest.fixture(params=SNRS, ids=lambda x: f"SNR={x}")
def snr(request):
    return request.param


@pytest.fixture
def vol_data_fixture(snr, dtype):
    return generate_data(snr, dtype)


def algo_params_id(params):
    return (
        f"loss_type={params[0]}, downsampling_level={params[1]}, max_iters={params[2]}"
    )


@pytest.mark.parametrize("algo_params", ALGO_PARAMS, ids=algo_params_id)
def test_bot_align(algo_params, vol_data_fixture):
    vol0, vol_given, L, R_true = vol_data_fixture

    R_init, R_rec = align_BO(
        vol0,
        vol_given,
        loss_type=algo_params[0],
        downsampling_level=algo_params[1],
        max_iters=algo_params[2],
        dtype=np.float32,
    )
    # Recovery without refinement (degrees)
    angle_init = angular_dist_degrees(R_init, R_true.T)
    # Recovery with refinement (degrees)
    angle_rec = angular_dist_degrees(R_rec, R_true.T)
    logging.debug(f"angle_init: {angle_init}, angle_rec: {angle_rec}")
    # Check we're within a degree.
    assert angle_rec < 2.0
