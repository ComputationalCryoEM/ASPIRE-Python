"""
This module contains utility functions that we often use in various modules.
Unlike helper functions, these functions have to do solely with algorithmic computations.

TODO move cfft, icfft, lgwt, pca_y here
"""

import numpy as np

from aspire.helpers import cart2rad


def estimate_snr(images):
    """
    Estimate signal-noise-ratio for a stack of projections.

    TODO test error size, we might have a bug here. it might be too large.
    """

    if len(images.shape) == 2:
        images = images[:, :, None]

    n = images.shape[2]

    p = images.shape[1]
    radius_of_mask = np.floor(p / 2.0) - 1.0

    r = cart2rad(p)
    points_inside_circle = r < radius_of_mask
    num_signal_points = np.count_nonzero(points_inside_circle)
    num_noise_points = p * p - num_signal_points

    var_n = np.sum(np.var(images[~points_inside_circle], axis=0)) * num_noise_points / (
                num_noise_points * n - 1)
    var_s = np.sum(np.var(images[points_inside_circle], axis=0)) * num_signal_points / (
                num_signal_points * n - 1)

    var_s -= var_n
    snr = var_s / var_n

    return snr, var_s, var_n
