import logging
import math

import numpy as np
from scipy.special import erf

from aspire.numeric import fft, xp
from aspire.utils import ensure

logger = logging.getLogger(__name__)


def crop_pad(mat, n, fill_value=None):
    """
    Reduce the size of a vector, square or cube 'mat' by cropping (or
    increase the size by padding with fill_value, by default zero) to a final
    size of n, (n x n), or (n x n x n) respectively. This is the analogue of downsample, but
    it doesn't change magnification.

    If mat is 2-dimensional and n is a vector, m is cropped to n=[mat_x mat_y].

    The function handles odd and even-sized arrays correctly. The center of
    an odd array is taken to be at (n+1)/2, and an even array is n/2+1.

    If flag is_stack is set to True, then a 3D array 'mat' is treated as a stack of 2D
    images, and each image is cropped to (n x n).

    For 2D images, the input image doesn't have to be square.

    * The original MATLAB function supported cropping to non-square matrices.
      As real-world uses will always crop to square (n, n), we don't support it with Python.


    :param mat: Vector, 2D array, stack of 2D arrays or a 3D array
    :param n: Size of desired cropped vector, side of 2D array or side of 3D array
    :param fill_value: Padding value. Defaults to 0.

    :return: Cropped or padded mat to size of n, (n x n) or (n x n x n)

    """

    num_dimensions = len(mat.shape)

    if num_dimensions not in [1, 2, 3]:
        raise RuntimeError(
            "cropping/padding failed! number of dimensions is too big!"
            f" ({num_dimensions} while max is 3)."
        )

    if num_dimensions == 2 and 1 in mat.shape:
        num_dimensions = 1

    if fill_value is None:
        fill_value = 0.0

    if num_dimensions == 1:  # mat is a vector for 1D object
        mat = np.reshape(mat, [mat.size, 1])  # force a column vector
        ns = math.floor(mat.size / 2) - math.floor(n / 2)  # shift term for scaling down
        if ns >= 0:  # cropping
            return mat[ns : ns + n].astype(np.float32)

        else:  # padding
            result_mat = fill_value * np.ones([n, 1], dtype=complex)
            result_mat[-ns : mat.size - ns] = mat
            return result_mat.astype(np.float32)

    elif num_dimensions == 2:  # mat is 2D image
        mat_x, mat_y = mat.shape
        # start_x = math.floor(mat_x / 2) - math.floor(n / 2)
        start_x = mat_x / 2 - n / 2  # shift term for scaling down
        # start_y = math.floor(mat_y / 2) - math.floor(n / 2)
        start_y = mat_y / 2 - n / 2  # shift term for scaling down

        if start_x >= 0 and start_y >= 0:  # cropping
            start_x, start_y = math.floor(start_x), math.floor(start_y)
            return mat[start_x : start_x + int(n), start_y : start_y + int(n)].astype(
                np.float32
            )

        elif start_x < 0 and start_y < 0:  # padding
            start_x, start_y = math.floor(start_x), math.floor(start_y)
            result_mat = fill_value * np.ones([n, n], dtype=complex)
            result_mat[-start_x : mat_x - start_x, -start_y : mat_y - start_y] = mat
            return result_mat.astype(np.float32)

        else:
            raise RuntimeError("Can't crop and pad simultaneously!")

    else:  # mat is 3D object

        from_shape = np.array(mat.shape)
        to_shape = np.array((n, n, n))

        ns = np.floor(from_shape / 2) - np.floor(to_shape / 2)
        ns, to_shape = ns.astype(int), to_shape.astype(
            int
        )  # can't slice later with float

        if np.all(ns >= 0):  # crop
            return mat[
                ns[0] : ns[0] + to_shape[0],
                ns[1] : ns[1] + to_shape[1],
                ns[2] : ns[2] + to_shape[2],
            ]

        elif np.all(ns <= 0):  # pad
            result_mat = fill_value * np.ones([n, n, n], dtype=complex)
            result_mat[
                -ns[0] : from_shape[0] - ns[0],
                -ns[1] : from_shape[2] - ns[1],
                -ns[2] : from_shape[2] - ns[2],
            ] = mat

            return result_mat.astype(np.float32)

        else:
            raise RuntimeError("Can't crop and pad simultaneously!")


def downsample(insamples, szout, mask=None):
    """
    Blur and downsample 1D to 3D objects such as, curves, images or volumes

    The function handles odd and even-sized arrays correctly. The center of
    an odd array is taken to be at (n+1)/2, and an even array is n/2+1.
    :param insamples: Set of objects to be downsampled in the form of an array.\
    the first dimension is the number of objects.
    :param szout: The desired resolution of for output objects.
    :return: An array consists of the blurred and downsampled objects.
    """

    ensure(
        insamples.ndim - 1 == np.size(szout),
        "The number of downsampling dimensions is not the same as that of objects.",
    )

    L_in = insamples.shape[1]
    L_out = szout[0]
    ndata = insamples.shape[0]
    outdims = np.r_[ndata, szout]

    outsamples = np.zeros(outdims, dtype=insamples.dtype)

    if mask is None:
        mask = 1.0

    if insamples.ndim == 2:
        # stack of one dimension objects

        for idata in range(ndata):
            insamples_shifted = fft.fftshift(fft.fft(xp.asarray(insamples[idata])))
            insamples_fft = crop_pad(insamples_shifted, L_out) * mask

            outsamples_shifted = fft.ifft(fft.ifftshift(xp.asarray(insamples_fft)))
            outsamples[idata] = np.real(xp.asnumpy(outsamples_shifted) * (L_out / L_in))

    elif insamples.ndim == 3:
        # stack of two dimension objects
        for idata in range(ndata):
            insamples_shifted = fft.fftshift(fft.fft2(xp.asarray(insamples[idata])))
            insamples_fft = crop_pad(insamples_shifted, L_out) * mask

            outsamples_shifted = fft.ifft2(fft.ifftshift(xp.asarray(insamples_fft)))
            outsamples[idata] = np.real(
                xp.asnumpy(outsamples_shifted) * (L_out ** 2 / L_in ** 2)
            )

    elif insamples.ndim == 4:
        # stack of three dimension objects
        for idata in range(ndata):
            insamples_shifted = fft.fftshift(
                fft.fftn(xp.asarray(insamples[idata]), axes=(0, 1, 2))
            )
            insamples_fft = crop_pad(insamples_shifted, L_out) * mask

            outsamples_shifted = fft.ifftn(
                fft.ifftshift(xp.asarray(insamples_fft)), axes=(0, 1, 2)
            )
            outsamples[idata] = np.real(
                xp.asnumpy(outsamples_shifted) * (L_out ** 3 / L_in ** 3)
            )

    else:
        raise RuntimeError("Number of dimensions > 3 for input objects.")

    return outsamples


def fuzzy_mask(L, r0, risetime, origin=None):
    """
    Create a centered 1D to 3D fuzzy mask of radius r0

    Made with an error function with effective rise time.
    :param L: The sizes of image in tuple structure
    :param r0: The specified radius
    :param risetime: The rise time for `erf` function
    :param origin: The coordinates of origin
    :return: The desired fuzzy mask
    """

    center = [sz // 2 + 1 for sz in L]
    if origin is None:
        origin = center

    grids = [np.arange(1 - org, ell - org + 1) for ell, org in zip(L, origin)]
    XYZ = np.meshgrid(*grids, indexing="ij")
    XYZ_sq = [X ** 2 for X in XYZ]
    R = np.sqrt(np.sum(XYZ_sq, axis=0))
    k = 1.782 / risetime
    m = 0.5 * (1 - erf(k * (R - r0)))

    return m
