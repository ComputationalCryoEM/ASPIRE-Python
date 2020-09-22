import logging
from aspire import config

logger = logging.getLogger(__name__)

if config.common.cupy:
    from .cupy import Cupy as NumericClass
else:
    from .numpy import Numpy as NumericClass

xp = NumericClass()

if config.common.fft == 'pyfftw':
    from .pyfftw_fft import PyfftwFFT as FFTClass
elif config.common.fft == 'cupy':
    from .cupy_fft import CupyFFT as FFTClass
elif config.common.fft == 'scipy':
    from .scipy_fft import ScipyFFT as FFTClass
else:
    logger.error(f'FFT selection, {config.common.normal_fft}, not implemented.')

fft = FFTClass()
