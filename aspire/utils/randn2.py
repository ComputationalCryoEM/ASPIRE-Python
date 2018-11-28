# Copyright (c) 2015 Jonas Rauber
# License: The MIT License (MIT)
# See: https://github.com/jonasrauber/randn-matlab-python

from numpy import sqrt
from numpy.random import rand
from scipy.special import erfinv

def randn2(*args,**kwargs):
    '''
    Calls rand and applies inverse transform sampling to the output.
    '''
    uniform = rand(*args, **kwargs)
    return sqrt(2) * erfinv(2 * uniform - 1)