import logging
import math

import numpy as np
from scipy.fftpack import (fft, fft2, fftn, fftshift, ifft, ifft2, ifftn,
                           ifftshift)
from scipy.interpolate import RegularGridInterpolator

from aspire.nfft import Plan
from aspire.utils import ensure
from aspire.utils.coor_trans import grid_1d, grid_2d, grid_3d
from aspire.utils.fft import (centered_fft1, centered_fft2_C, centered_fft3_C,
                              centered_ifft1, centered_ifft2_C, centered_ifft3_C)
from aspire.utils.matlab_compat import m_reshape

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
        raise RuntimeError("cropping/padding failed! number of dimensions is too big!"
                           f" ({num_dimensions} while max is 3).")

    if num_dimensions == 2 and 1 in mat.shape:
        num_dimensions = 1

    if fill_value is None:
            fill_value = 0.0

    if num_dimensions == 1:  # mat is a vector for 1D object
        mat = np.reshape(mat, [mat.size, 1])  # force a column vector
        ns = math.floor(mat.size / 2) - math.floor(n / 2)  # shift term for scaling down
        if ns >= 0:  # cropping
            return mat[ns: ns + n].astype('float32')

        else:  # padding
            result_mat = fill_value * np.ones([n, 1], dtype=complex)
            result_mat[-ns: mat.size - ns] = mat
            return result_mat.astype('float32')

    elif num_dimensions == 2:  # mat is 2D image
        mat_x, mat_y = mat.shape
        # start_x = math.floor(mat_x / 2) - math.floor(n / 2)
        start_x = mat_x / 2 - n / 2       # shift term for scaling down
        # start_y = math.floor(mat_y / 2) - math.floor(n / 2)
        start_y = mat_y / 2 - n / 2       # shift term for scaling down

        if start_x >= 0 and start_y >= 0:  # cropping
            start_x, start_y = math.floor(start_x), math.floor(start_y)
            return mat[start_x: start_x + int(n), start_y: start_y + int(n)].astype('float32')

        elif start_x < 0 and start_y < 0:  # padding
            start_x, start_y = math.floor(start_x), math.floor(start_y)
            result_mat = fill_value * np.ones([n, n], dtype=complex)
            result_mat[-start_x: mat_x - start_x, -start_y: mat_y - start_y] = mat
            return result_mat.astype('float32')

        else:
            raise RuntimeError("Can't crop and pad simultaneously!")

    else:  # mat is 3D object

        from_shape = np.array(mat.shape)
        to_shape = np.array((n, n, n))

        ns = np.floor(from_shape / 2) - np.floor(to_shape / 2)
        ns, to_shape = ns.astype(int), to_shape.astype(int)  # can't slice later with float

        if np.all(ns >= 0):   # crop
            return mat[ns[0]: ns[0]+to_shape[0],
                       ns[1]: ns[1]+to_shape[1],
                       ns[2]: ns[2]+to_shape[2]]

        elif np.all(ns <= 0): # pad
            result_mat = fill_value * np.ones([n, n, n], dtype=complex)
            result_mat[-ns[0]: from_shape[0] - ns[0],
                       -ns[1]: from_shape[2] - ns[1],
                       -ns[2]: from_shape[2] - ns[2]] = mat

            return result_mat.astype('float32')

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

    ensure(insamples.ndim-1 == np.size(szout),
           'The number of downsampling dimensions is not the same as that of objects.')

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
            insamples_fft = crop_pad(fftshift(fft(insamples[idata])), L_out)*mask
            outsamples[idata] = np.real(ifft(ifftshift(insamples_fft))*(L_out / L_in))

    elif insamples.ndim == 3:
        # stack of two dimension objects
        for idata in range(ndata):
            insamples_fft = crop_pad(fftshift(fft2(insamples[idata])), L_out)*mask
            outsamples[idata] = np.real(ifft2(ifftshift(insamples_fft)) * (L_out**2/L_in**2))

    elif insamples.ndim == 4:
        # stack of three dimension objects
        for idata in range(ndata):
            insamples_fft = crop_pad(fftshift(fftn(insamples[idata])), L_out)*mask
            outsamples[idata] = np.real(ifftn(ifftshift(insamples_fft)) * (L_out**3/L_in**3))

    else:
        raise RuntimeError('Number of dimensions > 3 for input objects.')

    return outsamples


def vol2img(volume, rots, L=None, dtype=None):
    """
    Generate 2D images from the input volume and rotation angles

    The function handles odd and even-sized arrays correctly. The center of
    an odd array is taken to be at (n+1)/2, and an even array is n/2+1.
    :param volume: A 3D volume objects.
    :param rots: A n-by-3-by-3 array of rotation angles.
    :param L: The output size of 2D images.
    :return: An array consists of 2D images.
    """

    if L is None:
        L = np.size(volume, 0)
    if dtype is None:
        dtype = volume.dtype

    lv = np.size(volume, 0)
    if L > lv+1:
        # For compatibility with gen_projections, allow one pixel aliasing.
        # More precisely, it should be N>nv, however, by using nv+1 the
        # results match those of gen_projections.
        if np.mod(L-lv, 2)==1:
            raise RuntimeError('Upsampling from odd to even sizes or vice versa is '
                               'currently not supported')
        dL = np.floor((L-lv)/2)
        fv = centered_fft3(volume)
        padded_volume = np.zeros((L, L, L), dtype=dtype)
        padded_volume[dL+1:dL+lv+1, dL+1:dL+lv+1, dL+1:dL+lv+1] = fv
        volume = centered_ifft3_C(padded_volume)
        ensure(np.norm(np.imag(volume[:]))/np.norm(volume[:]) < 1.0e-5,
               "The image part of volume is related large (>1.0e-5).")
        #  The new volume size
        lv = L

    grid2d = grid_2d(lv, shifted=True, normalized=False)

    num_pts = lv**2
    num_rots = rots.shape[0]
    pts = np.pi * np.vstack([grid2d['x'].flatten('F'), grid2d['y'].flatten('F'), np.zeros(num_pts)])

    pts_rot = np.zeros((3, num_pts, num_rots))

    for i in range(num_rots):
        pts_rot[:, :, i] = rots[i, :, :].T @ pts

    pts_rot = m_reshape(pts_rot, (3, lv**2*num_rots))

    pts_rot = -2*pts_rot/lv

    im_f = Plan(volume.shape, -pts_rot).transform(volume)

    im_f = m_reshape(im_f, (lv, lv, -1))

    if lv % 2 == 0:
        pts_rot = m_reshape(pts_rot, (3, lv, lv, num_rots))
        im_f = im_f * np.exp(1j*np.sum(pts_rot, 0)/2)
        im_f = im_f * np.expand_dims(np.exp(2*np.pi*1j*(grid2d['x'] +grid2d['y']-1)/(2*lv)), 0)

    im = centered_ifft2_C(im_f)
    if lv % 2 == 0:
        im = im * m_reshape(np.exp(2*np.pi*1j * (grid2d['x'] + grid2d['y'])/(2*lv)), (lv, lv, 1))

    return np.real(im)
