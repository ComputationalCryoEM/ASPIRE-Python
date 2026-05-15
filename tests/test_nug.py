import numpy as np
import pytest

from aspire.abinitio import CommonlineNUG
from aspire.source import Simulation
from aspire.utils import (
    J_conjugate,
    Random,
    Rotation,
    all_pairs,
    mean_aligned_angular_distance,
    utest_tolerance,
)
from aspire.volume import CnSymmetricVolume, CnSymmetryGroup

DTYPE = [np.float32]
RESOLUTION = [48, 49]
N_IMG = [5]
OFFSETS = [0]
ORDER = [3, 4]
PR = [False]
SEED = 1980


@pytest.fixture(params=DTYPE, ids=lambda x: f"dtype={x}", scope="module")
def dtype(request):
    return request.param


@pytest.fixture(params=RESOLUTION, ids=lambda x: f"resolution={x}", scope="module")
def resolution(request):
    return request.param


@pytest.fixture(params=N_IMG, ids=lambda x: f"n images={x}", scope="module")
def n_img(request):
    return request.param


@pytest.fixture(params=OFFSETS, ids=lambda x: f"offsets={x}", scope="module")
def offsets(request):
    return request.param


@pytest.fixture(params=ORDER, ids=lambda x: f"order={x}", scope="module")
def order(request):
    return request.param


@pytest.fixture(params=PR, ids=lambda x: f"proximal_refine={x}", scope="module")
def proximal_refine(request):
    return request.param


############
# Fixtures #
############

@pytest.fixture(scope="module")
def source(n_img, resolution, dtype, offsets, order):
    vol = CnSymmetricVolume(
        L=resolution, order=order, C=1, K=100, dtype=dtype, seed=SEED
    ).generate()

    src = Simulation(
        n=n_img,
        L=resolution,
        vols=vol,
        offsets=offsets,
        amplitudes=1,
        seed=SEED,
    )
    src = src.cache()  # Precompute image stack

    return src

@pytest.fixture(scope="module")
def orient_est(src, proximal_refine):
    orient_est = CommonlineNUG(
        src,
        perform_pr=proximal_refine,
    )

    return orient_est


#########
# Tests #
#########


def test_dtypes(orient_est):
    """
    Check dtypes for each major step of the algorithm.
    """
    pass
        
