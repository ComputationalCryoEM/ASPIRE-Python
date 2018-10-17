import functools
import os
import sys

import numpy as np

from aspire.common.config import AspireConfig
from aspire.common.exceptions import DimensionsIncompatible
from aspire.common.logger import logger


def cart2rad(n):
    """ Compute the radii corresponding to the points of a cartesian grid of size NxN points
        XXX This is a name for this function. """

    n = np.floor(n)
    x, y = image_grid(n)
    r = np.sqrt(np.square(x) + np.square(y))
    return r


def get_file_type(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    return os.path.splitext(file_path)[1]


def image_grid(n):
    # Return the coordinates of Cartesian points in an NxN grid centered around the origin.
    # The origin of the grid is always in the center, for both odd and even N.
    p = (n - 1.0) / 2.0
    x, y = np.meshgrid(np.linspace(-p, p, n), np.linspace(-p, p, n))
    return x, y


class TupleCompare(object):
    """ Helper class to compare between all members of tuples. """

    @classmethod
    def validate_same_length(cls, a, b):
        if len(a) != len(b):
            raise DimensionsIncompatible("Can't compare tuples of different length")

    @classmethod
    def gt(cls, a, b, eq=False):
        cls.validate_same_length(a, b)
        if eq is True:
            return all([i >= j for i, j in zip(a, b)])
        else:
            return all([i > j for i, j in zip(a, b)])

    @classmethod
    def lt(cls, a, b, eq=False):
        cls.validate_same_length(a, b)
        if eq is True:
            return all([i <= j for i, j in zip(a, b)])
        else:
            return all([i < j for i, j in zip(a, b)])

    @classmethod
    def eq(cls, a, b):
        cls.validate_same_length(a, b)
        return all([i == j for i, j in zip(a, b)])


def f_flatten(mat):
    """ Accept a matrix and return a flattened vector.

        This func mimics MATLAB's behavior:
        mat = mat(:)

    """

    return mat.flatten('F')


def yellow(s):
    yellow_color = '\033[1;33m'
    no_color = '\033[0m'
    return f"{yellow_color}{s}{no_color}"


def requires_binaries(*filenames):

    def decorator(func):

        missing_binaries = set()
        for fn in filenames:
            if not os.path.exists(f'./binaries/{fn}'):
                missing_binaries.add(fn)

        if missing_binaries:
            for fn in missing_binaries:
                logger.error(f"Binary file {yellow(fn)} is required!")

            logger.error(f"Please run \'{yellow('make data')}\' and try again.")
            sys.exit(1)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper
    return decorator


def requires_finufftpy(func):

    try:
        import finufftpy

    except ImportError:
        logger.error("Python package finufftpy is not installed! "
                     f"Please run \'{yellow('make finufftpy')}\'.")
        sys.exit(1)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper
