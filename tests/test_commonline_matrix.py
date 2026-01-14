import numpy as np
import pytest

from aspire.abinitio import (
    CLMatrixOrient3D,
    CLSymmetryC2,
    CLSymmetryC3C4,
    CLSync3N,
    CLSyncVoting,
    CommonlineIRLS,
    CommonlineLUD,
    CommonlineSDP,
)
from aspire.downloader import emdb_2660
from aspire.source import Simulation

SUBCLASSES = [
    CLSymmetryC2,
    CLSymmetryC3C4,
    CLSync3N,
    CLSyncVoting,
    CommonlineIRLS,
    CommonlineLUD,
    CommonlineSDP,
]


DTYPES = [
    np.float32,
    pytest.param(np.float64, marks=pytest.mark.expensive),
]


@pytest.fixture(params=SUBCLASSES, ids=lambda x: f"subclass={x}", scope="module")
def subclass(request):
    return request.param


@pytest.fixture(params=DTYPES, ids=lambda x: f"dtype={x}", scope="module")
def dtype(request):
    return request.param


@pytest.fixture(scope="module")
def src(dtype):
    src = Simulation(
        n=10,
        vols=emdb_2660().astype(dtype).downsample(32),
        offsets=0,
        amplitudes=1,
        seed=0,
    ).cache()

    return src


def test_class_structure(subclass):
    assert issubclass(subclass, CLMatrixOrient3D)


def test_clmatrix_lazy_eval(subclass, src, caplog):
    """
    Test lazy evaluation of commonlines matrix and associated log message.
    """
    cl_kwargs = dict(src=src)
    if subclass == CLSymmetryC3C4:
        cl_kwargs["symmetry"] = "C3"

    caplog.clear()
    msg = "Using existing estimated `clmatrix`."

    # Initialize commonlines class
    clmat_algo = subclass(**cl_kwargs)

    # clmatrix should be none at this point
    assert clmat_algo._clmatrix is None
    assert msg not in caplog.text

    # Request clmatrix
    _ = clmat_algo.clmatrix

    # clmatrix should be populated
    assert clmat_algo._clmatrix is not None
    assert msg not in caplog.text

    # 2nd request should access cached matrix and log message
    # that we are using the stored matrix
    _ = clmat_algo.clmatrix
    assert msg in caplog.text
