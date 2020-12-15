import logging

from aspire import config

logger = logging.getLogger(__name__)

if config.common.cupy:
    from .cupy import Cupy as NumericClass
else:
    from .numpy import Numpy as NumericClass

xp = NumericClass()


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
