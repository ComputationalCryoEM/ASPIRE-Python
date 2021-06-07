"""
Wavelet-based approximate Earthmover's distance (EMD) for n-dimensional signals.

This code is based on the following paper:
    Sameer Shirdhonkar and David W. Jacobs.
    "Approximate earth mover’s distance in linear time."
    2008 IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

More details are available in their technical report:
    CAR-TR-1025 CS-TR-4908 UMIACS-TR-2008-06.
"""

import numpy as np
import pywt


def wemd_embed(arr, wavelet, level):
    """
    This function computes an embedding of Numpy arrays such that
    for non-negative arrays that sum to one, the L1 distance between the
    resulting embeddings is strongly equivalent to the Earthmover distance
    of the arrays.

    :param arr: Numpy array
    :param level: Decomposition level of the wavelets.
    Larger levels yield more coefficients and more accurate results.
    As a rule of thumb, you can take level to be the log2 of the side-length of the domain.
    :param wavelet: Either the name of a wavelet supported by PyWavelets
    (e.g. 'coif3', 'sym3', 'sym5', etc.) or a pywt.Wavelet object
    See https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html#built-in-wavelets-wavelist
    :returns: One-dimensional numpy array containing weighted details coefficients.
    """
    dimension = arr.ndim

    arrdwt = pywt.wavedecn(arr, wavelet, mode="zero", level=level)
    detail_coefs = arrdwt[1:]
    assert len(detail_coefs) == level

    weighted_coefs = []
    for (j, details_level_j) in enumerate(detail_coefs):
        for coefs in details_level_j.values():
            multiplier = 2 ** ((level-1-j) * (1+(dimension/2.0)))
            weighted_coefs.append(coefs.flatten() * multiplier)

    return np.concatenate(weighted_coefs)


def wemd_norm(arr, wavelet, level):
    """
    Wavelet-based norm used to approximate the Earthmover's distance between
    mass distributions specified as Numpy arrays (typically images or volumes).

    :param arr: Numpy array of the difference between the two mass distributions.
    :param level: Decomposition level of the wavelets
    Larger levels yield more coefficients and more accurate results
    :param wavelet: Either the name of a wavelet supported by PyWavelets
    (e.g. 'coif3', 'sym3') or a pywt.Wavelet object
    See https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html#built-in-wavelets-wavelist
    :return: Approximated Earthmover's Distance
    """

    coefs = wemd_embed(arr, wavelet, level)
    return np.linalg.norm(coefs, ord=1)
