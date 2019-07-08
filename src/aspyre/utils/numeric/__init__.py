from aspyre import config


if config.common.cupy:
    from .cupy import Cupy as NumericClass
else:
    from .numpy import Numpy as NumericClass

xp = NumericClass()
