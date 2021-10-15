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


xp = numeric_object(config.common.numeric)


def fft_object(which):
    if which == "pyfftw":
        from .pyfftw_fft import PyfftwFFT as FFTClass
    elif which == "cupy":
        from .cupy_fft import CupyFFT as FFTClass
    elif which == "scipy":
        from .scipy_fft import ScipyFFT as FFTClass
    else:
        raise RuntimeError(f"Invalid selection for fft class: {which}")
    return FFTClass()


fft = fft_object(config.common.fft)
