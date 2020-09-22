from aspire import config

if config.common.cupy:
    from .cupy import Cupy as NumericClass
else:
    from .numpy import Numpy as NumericClass

xp = NumericClass()

if config.common.normal_fft == 0:
    from .pyfftw_fft import PyfftwFFT as FFTClass
elif config.common.normal_fft == 1:
    from .cupy_fft import CupyFFT as FFTClass
else:
    from .scipy_fft import ScipyFFT as FFTClass

fft = FFTClass()
