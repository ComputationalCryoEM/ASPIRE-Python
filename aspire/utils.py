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

    :arg images: stack of projections (between 1 and N projections)

    """

    if len(images.shape) == 2:  # in case of a single projection
        images = images[:, :, None]

    p = images.shape[1]
    n = images.shape[2]  # TODO test for single projection. This would most-prob fail

    radius_of_mask = np.floor(p / 2.0) - 1.0

    r = cart2rad(p)
    points_inside_circle = r < radius_of_mask
    num_signal_points = np.count_nonzero(points_inside_circle)
    num_noise_points = p * p - num_signal_points

    noise = np.sum(np.var(images[~points_inside_circle], axis=0)) * num_noise_points / (
                num_noise_points * n - 1)

    signal = np.sum(np.var(images[points_inside_circle], axis=0)) * num_signal_points / (
                num_signal_points * n - 1)

    signal -= noise

    snr = signal / noise

    return snr, signal, noise
