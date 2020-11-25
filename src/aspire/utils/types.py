"""
Miscellaneous utilities for common data type operations.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def real_type(complextype):
    """
    Get Numpy real type from corresponding complex type

    :param complextype: Numpy complex type
    :return realtype: Numpy real type
    """
    complextype = np.dtype(complextype)
    realtype = None
    if complextype == np.complex64:
        realtype = np.float32
    elif complextype == np.complex128:
        realtype = np.float64
    elif complextype in (np.float32, np.float64):
        logger.debug(f"Corresponding type is already real {complextype}.")
        realtype = complextype
    else:
        msg = f"Corresponding real type is not defined for {complextype}."
        logger.error(msg)
        raise TypeError(msg)

    return realtype


def complex_type(realtype):
    """
    Get Numpy complex type from corresponding real type

    :param realtype: Numpy real type
    :return complextype: Numpy complex type
    """
    realtype = np.dtype(realtype)
    complextype = None
    if realtype == np.float32:
        complextype = np.complex64
    elif realtype == np.float64:
        complextype = np.complex128
    elif realtype in (np.complex64, np.complex128):
        logger.debug(f"Corresponding type is already complex {realtype}.")
        complextype = realtype
    else:
        msg = f"Corresponding complex type is not defined for {realtype}."
        logger.error(msg)
        raise TypeError(msg)

    return complextype


def utest_tolerance(dtype):
    """
    Return ASPIRE tolerance for unit tests based on `dtype`.
    """

    if dtype == np.float64:
        # Use default np.allclose atol
        tol = 1e-8
    elif dtype == np.float32:
        tol = 1e-5
    else:
        msg = f"utest_tolerance not implemented for dtype: {dtype}."
        logger.error(msg)
        raise TypeError(msg)

    return tol
