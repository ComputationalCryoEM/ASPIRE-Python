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
        logger.debug(f'Corresponding type is already real {complextype}.')
        realtype = complextype
    else:
        msg = f'Corresponding real type is not defined for {complextype}.'
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
        logger.debug(f'Corresponding type is already complex {realtype}.')
        complextype = realtype
    else:
        msg = f'Corresponding complex type is not defined for {realtype}.'
        logger.error(msg)
        raise TypeError(msg)

    return complextype

def dtype_legacy_string(dtype):
    """
    Get legacy dtype string from corresponding numpy dtype.

    :param dtype: Numpy dtype
    :return: string
    """

    dtype_to_string_map = {'float32': 'single',
                           'float64': 'double',
                           'complex128': 'complex',
                           }

    dtype_str = dtype_to_string_map.get(str(dtype))

    if not dtype_str:
        msg = f'Corresponding dtype {str(dtype)} is not defined.'
        logger.error(msg)
        raise TypeError(msg)

    return dtype_str
