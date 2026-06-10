import os
import socket

import numpy as np
import pytest

from aspire.basis import FFBBasis2D, FLEBasis2D
from aspire.covariance import BatchedRotCov2D
from aspire.source import RelionSource

DTYPES = [
    np.float32,
    np.float64,
]


@pytest.fixture(params=DTYPES, ids=lambda x: f"dtype={x}", scope="module")
def dtype(request):
    return request.param


IMG_SIZES = [
    128,
    179,
]


@pytest.fixture(params=IMG_SIZES, ids=lambda x: f"img_size={x}", scope="module")
def img_size(request):
    return request.param


BASI = [
    FFBBasis2D,
    FLEBasis2D,
]


@pytest.fixture(params=BASI, ids=lambda x: f"basis={x}", scope="module")
def basis(request, img_size, dtype):
    return request.param(img_size, dtype=dtype)


RADIAL = [
    False,
    True,
]


@pytest.fixture(params=RADIAL, ids=lambda x: f"force_radial={x}", scope="module")
def force_radial(request):
    return request.param


MOLECULES = {
    10028: "10028/data/shiny_2sets_fixed9.star",
    11618: "11618/data/particles/J43_particles.star",
}


@pytest.fixture(params=MOLECULES.keys(), ids=lambda x: f"molecule={x}", scope="module")
def molecule(request):
    return request.param


def _raw_data_path():
    """
    Attempt getting a working path to raw EMPIAR data location
    """
    # Try to populate a path to raw data
    raw_data_path = None

    # Check if we're on a known testing platform.
    known_hosts = [
        "caf.math.princeton.edu.private",
        "decaf.math.princeton.edu.private",
    ]
    # Default to their expected location
    if socket.gethostname() in known_hosts:
        raw_data_path = "/scratch/ExperimentalData/raw"

    # Check if a user has provided or overides the location
    raw_data_path = os.environ.get("ASPIRE_RAW_DATA_PATH", raw_data_path)

    if raw_data_path is None:
        raise RuntimeError("Must provide path to raw data")
    if not os.path.exists(raw_data_path):
        raise RuntimeError(f"Provided path {raw_data_path} does not exist.")

    return raw_data_path


@pytest.fixture(scope="module")
def preprocessed_src(img_size, molecule, force_radial, dtype):
    starfile_path = os.path.join(_raw_data_path(), MOLECULES[molecule])
    if not os.path.exists(starfile_path):
        raise RuntimeError(f"Expected starfile path {starfile_path} does not exist.")

    src = RelionSource(starfile_path, dtype=dtype)

    # preprocess
    src = src.downsample(img_size).cache()
    src = src.phase_flip().cache()
    src = src.normalize_background().cache()
    src = src.whiten().cache()
    src = src.invert_contrast()

    # To run radially optimized code we need
    #  i) radial filters
    #  ii) set radial expand mode in cov2d
    if force_radial:
        src.filter_stack = src.filter_stack.to_radial()

    return src


@pytest.mark.covar
def test_covar(preprocessed_src, basis, force_radial):

    # To run radially optimized code we need
    #  i) radial filters
    #  ii) set radial expand mode in cov2d
    expand_method = None  # default for cov2d
    if force_radial:
        assert (
            preprocessed_src.filter_stack.radial
        ), "Expected radial filters under `force_radial=True`"
        expand_method = "radial"

    cov2d = BatchedRotCov2D(preprocessed_src, basis, expand_method=expand_method)
    # smoke test
    _ = cov2d.get_covar()
