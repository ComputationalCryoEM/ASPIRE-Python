import logging

from aspire import config

from .complex_pca.complex_pca import ComplexPCA

logger = logging.getLogger(__name__)


def numeric_object(which):
    if which == "cupy":
        from .cupy import Cupy as NumericClass
    elif which == "numpy":
        from .numpy import Numpy as NumericClass
    else:
        raise RuntimeError(f"Invalid selection for numeric module: {which}")
    return NumericClass()


xp = numeric_object(config["common"]["numeric"].as_str())


def fft_object(which):
    if which == "pyfftw":
        from .pyfftw_fft import PyfftwFFT as FFTClass
    elif which == "cupy":
        from .cupy_fft import CupyFFT as FFTClass
    elif which == "scipy":
        from .scipy_fft import ScipyFFT as FFTClass
    elif which == "mkl":
        from .mkl_fft import MKLFFT as FFTClass
    else:
        raise RuntimeError(f"Invalid selection for fft class: {which}")
    return FFTClass()


fft = fft_object(config["common"]["fft"].as_str())

# Sanity check.
if (config["common"]["numeric"].as_str() == "cupy") and (
    config["common"]["fft"].as_str() != "cupy"
):
    raise RuntimeError(
        "Using `cupy` numeric backend without `cupy` fft is unsupported."
    )

if (config["common"]["fft"].as_str() == "cupy") and (
    config["common"]["numeric"].as_str() != "cupy"
):
    raise RuntimeError(
        "Using `cupy` fft without `cupy` numeric backend is unsupported."
    )


# Configure `sparse` in tandem with `numeric` as the arrays generally will need to interoperate.
def sparse_object(which):
    if which == "cupy":
        from cupyx.scipy import sparse as SparseClass

        # CuPy imports don't work the same as scipy
        from cupyx.scipy.sparse.linalg import eigsh

        SparseClass.linalg.eigsh = eigsh
    elif which == "numpy":
        from scipy import sparse as SparseClass
    else:
        raise RuntimeError(f"Invalid selection for sparse module: {which}")
    return SparseClass


sparse = sparse_object(config["common"]["numeric"].as_str())
