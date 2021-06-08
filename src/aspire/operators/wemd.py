"""
Wavelet-based approximate Earthmover's distance (EMD) for n-dimensional signals.

This code is based on the following paper:
    Sameer Shirdhonkar and David W. Jacobs.
    "Approximate earth mover’s distance in linear time."
    2008 IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

More details are available in their technical report:
    CAR-TR-1025 CS-TR-4908 UMIACS-TR-2008-06.
"""

import warnings

import numpy as np
import pywt


def wemd_embed(arr, wavelet="coif3", level=None):
    """
    This function computes an embedding of Numpy arrays such that
    for non-negative arrays that sum to one, the L1 distance between the
    resulting embeddings is strongly equivalent to the Earthmover distance
    of the arrays.

    :param arr: Numpy array
    :param level: Decomposition level of the wavelets.
    Larger levels yield more coefficients and more accurate results.
    If no level is given, we take the the log2 of the side-length of the domain.
    :param wavelet: Either the name of a wavelet supported by PyWavelets
    (e.g. 'coif3', 'sym3', 'sym5', etc.) or a pywt.Wavelet object
    See https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html#built-in-wavelets-wavelist
    The default is 'coif3', because it seems to work well empirically.
    :returns: One-dimensional numpy array containing weighted details coefficients.
    """
    dimension = arr.ndim

    if level is None:
        level = int(np.ceil(np.log2(max(arr.shape)))) + 1

    # Using wavedecn with the default level creates this boundary effects warning.
    # However, this doesn't seem to be a cause for concern.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Level value of .* is too high:"
            " all coefficients will experience boundary effects.",
        )
        arrdwt = pywt.wavedecn(arr, wavelet, mode="zero", level=level)

    detail_coefs = arrdwt[1:]
    assert len(detail_coefs) == level

    weighted_coefs = []
    for (j, details_level_j) in enumerate(detail_coefs):
        multiplier = 2 ** ((level - 1 - j) * (1 + (dimension / 2.0)))
        for coefs in details_level_j.values():
            weighted_coefs.append(multiplier * coefs.flatten())

    return np.concatenate(weighted_coefs)


def wemd_norm(arr, wavelet="coif3", level=None):
    """
    Wavelet-based norm used to approximate the Earthmover's distance between
    mass distributions specified as Numpy arrays (typically images or volumes).

    :param arr: Numpy array of the difference between the two mass distributions.
    :param level: Decomposition level of the wavelets.
    Larger levels yield more coefficients and more accurate results.
    If no level is given, we take the the log2 of the side-length of the domain.
    Larger levels yield more coefficients and more accurate results
    :param wavelet: Either the name of a wavelet supported by PyWavelets
    (e.g. 'coif3', 'sym3', 'sym5', etc.) or a pywt.Wavelet object
    See https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html#built-in-wavelets-wavelist
    The default is 'coif3', because it seems to work well empirically.
    :return: Approximated Earthmover's Distance
    """

    coefs = wemd_embed(arr, wavelet, level)
    return np.linalg.norm(coefs, ord=1)
