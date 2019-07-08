"""
Functions for compatibility with MATLAB behavior.
At some point when the package is full validated against MatLab, the 'order' arguments in the functions here
can be changed to 'C', and subsequently, this package deprecated altogether (i.e. the reshape/flatten methods used
directly by the caller).
"""

import numpy as np
from scipy.special import erfinv

SQRT2 = np.sqrt(2)

# A list of random states, used as a stack
random_states = []


def m_reshape(x, new_shape):
    # This is a somewhat round-about way of saying:
    #   return x.reshape(new_shape, order='F')
    # We follow this approach since numba/cupy don't support the 'order'
    # argument, and we may want to use those decorators in the future
    # Note that flattening is required before reshaping, because
    if isinstance(new_shape, tuple):
        return m_flatten(x).reshape(new_shape[::-1]).T
    else:
        return x


def m_flatten(x):
    # This is a somewhat round-about way of saying:
    #   return x.flatten(order='F')
    # We follow this approach since numba/cupy don't support the 'order'
    # argument, and we may want to use those decorators in the future
    return x.T.flatten()


def randi(i_max, size, seed=None):
    """
    A MATLAB compatible randi implementation that returns numbers from a discrete uniform distribution.
    While a direct use of np.random.choice would be convenient, this doesn't seem to return results
    identical to MATLAB.

    :param iMax: TODO
    :param size: size of the resulting np array
    :param seed: Random seed to use (None to apply no seed)
    :return: A np array
    """
    with Random(seed):
        return np.ceil(i_max * np.random.random(size=size)).astype('int')


def randn(*args, **kwargs):
    """
    Calls rand and applies inverse transform sampling to the output.
    """
    seed = None
    if 'seed' in kwargs:
        seed = kwargs.pop('seed')

    with Random(seed):
        uniform = np.random.rand(*args, **kwargs)
        result = SQRT2 * erfinv(2 * uniform - 1)
        # TODO: Rearranging elements to get consistent behavior with MATLAB 'randn2'
        result = m_reshape(result.flatten(), args)
        return result


def rand(size, seed=None):
    with Random(seed):
        return m_reshape(np.random.random(np.prod(size)), size)


class Random:
    """
    A context manager that pushes a random seed to the stack for reproducible results,
    and pops it on exit.
    """
    def __init__(self, seed=None):
        self.seed = seed

    def __enter__(self):
        if self.seed is not None:
            # Push current state on stack
            random_states.append(np.random.get_state())

            seed = self.seed
            # 5489 is the default seed used by MATLAB for seed 0 !
            if seed == 0:
                seed = 5489

            new_state = np.random.RandomState(seed)
            np.random.set_state(new_state.get_state())

    def __exit__(self, *args):
        if self.seed is not None:
            np.random.set_state(random_states.pop())
