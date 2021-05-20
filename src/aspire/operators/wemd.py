"""
Wavelet-based approximate Earthmover's distance (EMD) for 2D/3D signals.

This code is based on the following paper:
    Sameer Shirdhonkar and David W. Jacobs. "Approximate earth mover’s distance in linear time." 2008 IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

More details are available in their technical report: CAR-TR-1025 CS-TR-4908 UMIACS-TR-2008-06.
"""

import numpy as np
import pywt


def wembed(arr, wavelet, level):
    """
    This function computes an embedding of Numpy arrays such that
    the L1 distance between the resulting embeddings is approximately
    equal to the Earthmover distance of the arrays.

    :param arr: Numpy array
    :param level: Decomposition level of the wavelets
    Larger levels yield more coefficients and more accurate results
    :param wavelet: Either the name of a wavelet supported by PyWavelets
    (e.g. 'coif3', 'sym3') or a pywt.Wavelet object
    See https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html#built-in-wavelets-wavelist
    :returns: One-dimensional numpy array containing weighted details coefficients.
    """

    arrdwt = pywt.wavedecn(arr / arr.sum(), wavelet, mode="zero", level=level)

    dimension = len(arr.shape)
    assert dimension in [2, 3], f"Dimension {dimension} should be 2 or 3"

    n_levels = len(arrdwt[1:])

    weighted_coefs = []
    for (j, details_level_j) in enumerate(arrdwt[1:]):
        for coefs in details_level_j.values():
            multiplier = 2 ** ((n_levels - 1 - j) * (1 + (dimension / 2.0)))
            weighted_coefs.append(coefs.flatten() * multiplier)

    return np.concatenate(weighted_coefs)


def wemd(arr1, arr2, wavelet, level):
    """
    Approximate Earthmover's distance between  using `embed`.

    :param arr1: Numpy array
    :param arr2: Numpy array
    :param level: Decomposition level of the wavelets
    Larger levels yield more coefficients and more accurate results
    :param wavelet: Either the name of a wavelet supported by PyWavelets
    (e.g. 'coif3', 'sym3') or a pywt.Wavelet object
    See https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html#built-in-wavelets-wavelist
    :return: Approximated Earthmover Distance
    """

    coefs1 = wembed(arr1, wavelet, level)
    coefs2 = wembed(arr2, wavelet, level)

    return np.linalg.norm(coefs1 - coefs2, ord=1)
