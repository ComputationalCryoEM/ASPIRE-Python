import logging
import os

import numpy as np
import pytest
from numpy.linalg import norm
from numpy.random import normal

from aspire.utils import Rotation, align_BO
from aspire.volume import Volume


def _angular_dist_degrees(R1, R2):
    """
    Util wrapper computing angular distance as degrees.

    :param R1: 3x3 rotation matrix
    :param R2: 3x3 rotation matrix
    :return: Angular distance between R1 and R2 in degrees.
    """
    return Rotation.angle_dist(R1, R2) * 180 / np.pi


# Original author suggests testing with the following parameters:
#     loss type ('wemd' or 'euclidean')
#     downsampling level (32 or 64 recommended)
#     total number of iterations (150 or 200 recommended)

ALGO_PARAMS = [
    pytest.param(["wemd", 32, 200], marks=pytest.mark.expensive),
    ["wemd", 64, 150],
    ["euclidean", 32, 200],
    pytest.param(["euclidean", 64, 150], marks=pytest.mark.expensive),
    pytest.param(["wemd", 31, 400], marks=pytest.mark.expensive),
]

SNRS = [pytest.param(float("inf"), marks=pytest.mark.expensive), 0.5]
DTYPES = [np.float32, pytest.param(np.float64, marks=pytest.mark.expensive)]


def algo_params_id(params):
    return f"loss_type={params[0]}, downsampled_size={params[1]}, max_iters={params[2]}"


@pytest.fixture(params=DTYPES, ids=lambda x: f"dtype={x}")
def dtype(request):
    return request.param


@pytest.fixture(params=SNRS, ids=lambda x: f"SNR={x}")
def snr(request):
    return request.param


@pytest.fixture
def vol_data_fixture(snr, dtype):
    """
    Generates test data from MRC file.

    :param snr: Signal to noise ratio (0, float('inf') ).
        float('inf') would represent a clean image.
    :param dtype: Datatype of generated data.
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
    r = Rotation.generate_random_rotations(1, dtype=v.dtype, seed=1234)
    R_true = r.matrices[0]
    test_vol = v.rotate(r) + normal(0, ns_std, shape)

    return reference_vol, test_vol, R_true


@pytest.mark.parametrize("algo_params", ALGO_PARAMS, ids=algo_params_id)
def test_bot_align(algo_params, vol_data_fixture):
    """
    Tests the BOT alignment algorithm for a given set of
    parameters and volume data.
    """

    vol0, vol_given, R_true = vol_data_fixture

    R_init, R_rec = align_BO(
        vol0,
        vol_given,
        loss_type=algo_params[0],
        downsampled_size=algo_params[1],
        max_iters=algo_params[2],
    )

    # Recovery without refinement (degrees)
    angle_init = _angular_dist_degrees(R_init, R_true.T)
    # Recovery with refinement (degrees)
    angle_rec = _angular_dist_degrees(R_rec, R_true.T)
    logging.debug(f"angle_init: {angle_init}, angle_rec: {angle_rec}")

    # Check we're close
    assert angle_rec < 2.0
