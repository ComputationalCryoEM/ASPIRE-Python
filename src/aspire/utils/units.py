"""
Miscellaneous utilities for common unit conversions.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def ratio_to_decibel(p):
    """
    Convert a ratio of powers to decibel (log) scale.

    Follows numpy broadcasting rules.

    :param p: Power ratio.
    :returns: Power ratio in log scale
    """

    return 10 * np.log10(p)
