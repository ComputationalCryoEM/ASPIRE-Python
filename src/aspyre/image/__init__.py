import numpy as np
from scipy.fftpack import ifftshift, ifft2, fft2
from scipy.interpolate import RegularGridInterpolator

from aspyre.utils import ensure
from aspyre.utils.math import grid_2d
from aspyre.utils.fft import centered_fft2, centered_ifft2
from aspyre.utils.matrix import roll_dim, unroll_dim


# TODO: The following functions all need to move inside the Image class
def im_translate(im, shifts):
    """
    Translate image by shifts
    :param im: An array of size L-by-L-by-n containing images to be translated.
    :param shifts: An array of size 2-by-n specifying the shifts in pixels.
        Alternatively, it can be a column vector of length 2, in which case the same shifts is applied to each image.
    :return: The images translated by the shifts, with periodic boundaries.
    """

    n_im = im.shape[-1]
    n_shifts = shifts.shape[-1]

    ensure(shifts.shape[0] == 2, "shifts must be 2xn")
    ensure(n_shifts == 1 or n_shifts == n_im, "no. of shifts must be 1 or match the no. of images")
    ensure(im.shape[0] == im.shape[1], "images must be square")

    L = im.shape[0]
    im_f = fft2(im, axes=(0, 1))
    grid_1d = ifftshift(np.ceil(np.arange(-L/2, L/2))) * 2 * np.pi / L
    om_x, om_y = np.meshgrid(grid_1d, grid_1d, indexing='ij')

    phase_shifts_x = np.broadcast_to(-shifts[0, :], (L, L, n_shifts))
    phase_shifts_y = np.broadcast_to(-shifts[1, :], (L, L, n_shifts))
    phase_shifts = (om_x[:, :, np.newaxis] * phase_shifts_x) + (om_y[:, :, np.newaxis] * phase_shifts_y)

    mult_f = np.exp(-1j * phase_shifts)
    im_translated_f = im_f * mult_f
    im_translated = ifft2(im_translated_f, axes=(0, 1))
    im_translated = np.real(im_translated)

    return im_translated


def im_downsample(im, L_ds):
    """
    Blur and downsample image
    :param im: Set of images to be downsampled in the form of an array L-by-L-by-K, where K is the number of images.
    :param L_ds: The desired resolution of the downsampled images. Must be smaller than L.
    :return: An array of the form L_ds-by-L_ds-by-K consisting of the blurred and downsampled images.
    """
    N = im.shape[0]
    grid = grid_2d(N)
    grid_ds = grid_2d(L_ds)

    im_ds = np.zeros((L_ds, L_ds, im.shape[2])).astype(im.dtype)

    # x, y values corresponding to 'grid'. This is what scipy interpolator needs to function.
    x = y = np.ceil(np.arange(-N/2, N/2)) / (N/2)

    mask = (np.abs(grid['x']) < L_ds/N) & (np.abs(grid['y']) < L_ds/N)
    im = np.real(centered_ifft2(centered_fft2(im) * np.expand_dims(mask, 2)))

    for s in range(im_ds.shape[-1]):
        interpolator = RegularGridInterpolator(
            (x, y),
            im[:, :, s],
            bounds_error=False,
            fill_value=0
        )
        im_ds[:, :, s] = interpolator(np.dstack([grid_ds['x'], grid_ds['y']]))

    return im_ds


def im_filter(im, filt, *args, **kwargs):
    # TODO: Move inside appropriate object
    L = im.shape[0]
    im, sz_roll = unroll_dim(im, 3)
    filter_vals = filt.evaluate_grid(L, *args, **kwargs)
    im_f = centered_fft2(im)
    if im_f.ndim > filter_vals.ndim:
        im_f = np.expand_dims(filter_vals, 2) * im_f
    else:
        im_f = filter_vals * im_f
    im = centered_ifft2(im_f)
    im = np.real(im)
    im = roll_dim(im, sz_roll)

    return im
