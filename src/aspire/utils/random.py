"""
Utilities for controlling and generating random numbers.
"""

import numpy as np
from scipy.special import erfinv

from aspire.utils.matlab_compat import m_reshape

# A list of random states, used as a stack
random_states = []


def choice(*args, **kwargs):
    """
    Wraps numpy random.choice call in ASPIRE Random context.
    """
    seed = None
    if "seed" in kwargs:
        seed = kwargs.pop("seed")

    with Random(seed):
        return np.random.choice(*args, **kwargs)


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
        return np.ceil(i_max * np.random.random(size=size)).astype("int")


def randn(*args, **kwargs):
    """
    Calls rand and applies inverse transform sampling to the output.
    """
    seed = None
    if "seed" in kwargs:
        seed = kwargs.pop("seed")

    with Random(seed):
        uniform = np.random.rand(*args, **kwargs)
        result = np.sqrt(2) * erfinv(2 * uniform - 1)
        # TODO: Rearranging elements to get consistent behavior with MATLAB 'randn2'
        result = m_reshape(result.flatten(), args)
        return result


def rand(size, seed=None):
    """
    Note this is for MATLAB repro (see m_reshape).

    Other uses prefer use of `random`.
    """
    with Random(seed):
        return m_reshape(np.random.random(np.prod(size)), size)


def random(*args, **kwargs):
    """
    Wraps numpy.random.random with ASPIRE Random context manager.
    """
    seed = None
    if "seed" in kwargs:
        seed = kwargs.pop("seed")

    with Random(seed):
        return np.random.random(*args, **kwargs)


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
