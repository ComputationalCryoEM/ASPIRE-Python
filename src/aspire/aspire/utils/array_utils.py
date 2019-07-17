"""
This module contains functions useful for common tasks such as crop/mask/downsample etc.
It is different from the functions in PreProcessor as they allow you to crop/downsample to a
non square size. PreProcessor.crop accepts size as int and here it's accepted as (X,Y) tuple.
TODO reduced this file by refactoring it and moving crop/downsample capabilities to PreProcessor.
"""
import numpy as np
import pyfftw
from console_progressbar import ProgressBar
from numpy.fft import fftshift, ifftshift
from pyfftw.interfaces.numpy_fft import fft2, ifft2

from aspire.aspire.common.logger import logger


def crop(images, out_size, is_stack, fillval=0.0):
    if is_stack:
        in_size = images.shape[:-1]
    else:
        in_size = images.shape
    num_images = images.shape[-1]
    num_dim = len(in_size)
    out_shape = [s for s in out_size] + [num_images]
    if num_dim == 1:
        out_x_size = out_size[0]
        x_size = in_size[0]
        nx = int(np.floor(x_size * 1.0 / 2) - np.floor(out_x_size * 1.0 / 2))
        if nx >= 0:
            nc = images[nx:nx + out_x_size]
        else:
            nc = np.zeros(out_shape) + fillval
            nc[-nx:x_size - nx] = images

    elif num_dim == 2:
        out_x_size = out_size[0]
        x_size = in_size[0]
        nx = int(np.floor(x_size * 1.0 / 2) - np.floor(out_x_size * 1.0 / 2))
        out_y_size = out_size[1]
        y_size = in_size[1]
        ny = int(np.floor(y_size * 1.0 / 2) - np.floor(out_y_size * 1.0 / 2))
        if nx >= 0 and ny >= 0:
            nc = images[nx:nx + out_x_size, ny:ny + out_y_size]
        elif nx < 0 and ny < 0:
            nc = np.zeros(out_shape) + fillval
            nc[-nx:x_size - nx, -ny:y_size - ny] = images
        else:
            return 0  # raise error

    elif num_dim == np.e:
        out_x_size = out_size[0]
        x_size = in_size[0]
        nx = int(np.floor(x_size * 1.0 / 2) - np.floor(out_x_size * 1.0 / 2))
        out_y_size = out_size[1]
        y_size = in_size[1]
        ny = int(np.floor(y_size * 1.0 / 2) - np.floor(out_y_size * 1.0 / 2))
        out_z_size = out_size[2]
        z_size = in_size[2]
        nz = int(np.floor(z_size * 1.0 / 2) - np.floor(out_z_size * 1.0 / 2))
        if nx >= 0 and ny >= 0 and nz >= 0:
            nc = images[nx:nx + out_x_size, ny:ny + out_y_size, nz:nz + out_z_size]
        elif nx < 0 and ny < 0 and nz < 0:
            nc = np.zeros(out_shape) + fillval
            nc[-nx:x_size - nx, -ny:y_size - ny, -nz:y_size - nz] = images
        else:
            return 0  # raise error
    else:
        return 0  # raise error
    return nc


def mask(images, is_stack=False, r=None, rise_time=None):

    num_dims = images.ndim
    if num_dims < 2 or num_dims > 3:
        pass  # raise error

    if is_stack and num_dims == 2:
        pass  # raise error

    if is_stack:
        num_dims = 2

    shape = images.shape[:num_dims]
    if num_dims == 2:
        if shape[0] != shape[1]:
            pass  # raise error

    if num_dims == 3:
        if shape[0] != shape[1] or shape[0] != shape[2] or shape[1] != shape[2]:
            pass  # raise error

    n = shape[0]
    if r is None:
        r = int(np.floor(0.45 * n))

    if rise_time is None:
        rise_time = int(np.floor(0.05 * n))

    m = fuzzymask(n, num_dims, r, rise_time)
    out = (images.transpose((2, 0, 1)) * m).transpose((1, 2, 0))
    return out


def fuzzymask(n, dims, r0, rise_time, origin=None):
    if isinstance(n, int):
        n = np.array([n])

    if isinstance(r0, int):
        r0 = np.array([r0])

    center = (n + 1.0) / 2
    k = 1.782 / rise_time

    if dims == 1:
        if origin is None:
            origin = center
            origin = origin.astype('int')
        r = np.abs(np.arange(1 - origin[0], n - origin[0] + 1))

    elif dims == 2:
        if origin is None:
            origin = np.floor(n / 2) + 1
            origin = origin.astype('int')
        if len(n) == 1:
            x, y = np.mgrid[1 - origin[0]:n[0] - origin[0] + 1, 1 - origin[0]:n[0] - origin[0] + 1]
        else:
            x, y = np.mgrid[1 - origin[0]:n[0] - origin[0] + 1, 1 - origin[1]:n[1] - origin[1] + 1]

        if len(r0) < 2:
            r = np.sqrt(np.square(x) + np.square(y))
        else:
            r = np.sqrt(np.square(x) + np.square(y * r0[0] / r0[1]))

    elif dims == 3:
        if origin is None:
            origin = center
            origin = origin.astype('int')
        if len(n) == 1:
            x, y, z = np.mgrid[1 - origin[0]:n[0] - origin[0] + 1, 1 - origin[0]:n[0] - origin[0] + 1, 1 - origin[0]:n[0] - origin[0] + 1]
        else:
            x, y, z = np.mgrid[1 - origin[0]:n[0] - origin[0] + 1, 1 - origin[1]:n[1] - origin[1] + 1, 1 - origin[2]:n[2] - origin[2] + 1]

        if len(r0) < 3:
            r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
        else:
            r = np.sqrt(np.square(x) + np.square(y * r0[0] / r0[1]) + np.square(z * r0[0] / r0[2]))
    else:
        return 0  # raise error

    m = 0.5 * (1 - erf(k * (r - r0[0])))
    return m


def downsample(images, out_size, is_stack=True):

    if not is_stack:
        images = np.expand_dims(images, 0)

    in_size = np.array(images.shape[:-1])
    out_size = np.zeros(in_size.shape, dtype='int') + out_size
    num_dim = len(in_size)
    down = all(in_size < out_size)
    up = all(in_size > out_size)
    if not (down or up):
        if all(in_size == out_size):
            return images
        pass  # raise error

    if num_dim > 3:
        pass  # raise error

    if num_dim == 1:
        images = images.swapaxes(0, 1)
    elif num_dim == 2:
        images = images.swapaxes(0, 2)
        images = images.swapaxes(1, 2)
    else:
        images = images.swapaxes(0, 3)
        images = images.swapaxes(1, 3)
        images = images.swapaxes(2, 3)

    out = np.zeros([images.shape[0]] + out_size.tolist(), dtype='complex128')

    for i, image in enumerate(images):
        tmp = pyfftw.interfaces.numpy_fft.fftshift(pyfftw.interfaces.numpy_fft.fftn(image))
        out_tmp = pyfftw.interfaces.numpy_fft.ifftshift(crop(tmp, out_size, False))
        out[i] = pyfftw.interfaces.numpy_fft.ifftn(out_tmp)

    if num_dim == 1:
        out = out.swapaxes(0, 1)
    elif num_dim == 2:
        out = out.swapaxes(1, 2)
        out = out.swapaxes(0, 2)
    else:
        out = out.swapaxes(2, 3)
        out = out.swapaxes(1, 3)
        out = out.swapaxes(0, 3)

    out = out.squeeze() * np.prod(out_size) * 1.0 / np.prod(in_size)
    return out


def flatten(array):
    """ Accept an array and return a flattened vector using Fortran convention.

        This func mimics MATLAB's behavior:
        array = array(:)

    """

    return array.flatten('F')


def estimate_snr(images):
    """
    Estimate signal-noise-ratio for a stack of projections.

    :arg images: stack of projections (between 1 and N projections)

    TODO test error size, we might have a bug here. it might be too large.
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


def cart2rad(n):
    """ Compute the radii corresponding to the points of a cartesian grid of size NxN points
        XXX This is a name for this function. """

    n = np.floor(n)
    x, y = image_grid(n)
    r = np.sqrt(np.square(x) + np.square(y))
    return r


def image_grid(n):
    # Return the coordinates of Cartesian points in an NxN grid centered around the origin.
    # The origin of the grid is always in the center, for both odd and even N.
    p = (n - 1.0) / 2.0
    x, y = np.meshgrid(np.linspace(-p, p, n), np.linspace(-p, p, n))
    return x, y


def radius_norm(n: int, origin=None):
    """
        Create an n(1) x n(2) array where the value at (x,y) is the distance from the
        origin, normalized such that a distance equal to the width or height of
        the array = 1.  This is the appropriate function to define frequencies
        for the fft of a rectangular image.

        For a square array of size n (or [n n]) the following is true:
        RadiusNorm(n) = Radius(n)/n.
        The org argument is optional, in which case the FFT center is used.

        Theta is the angle in radians.

        (Transalted from Matlab RadiusNorm.m)
    """

    if isinstance(n, int):
        n = np.array([n, n])

    if origin is None:
        origin = np.ceil((n + 1) / 2)

    a, b = origin[0], origin[1]
    x, y = np.meshgrid(np.arange(1-a, n[0]-a+1)/n[0],
                       np.arange(1-b, n[1]-b+1)/n[1])  # zero at x,y
    radius = np.sqrt(x ** 2 + y ** 2)

    theta = np.arctan2(x, y)

    return radius, theta


def cryo_epsds(imstack, samples_idx, max_d, verbose=None):
    p = imstack.shape[0]
    if max_d >= p:
        max_d = p-1
        print('max_d too large. Setting max_d to {}'.format(max_d))

    r, x, _ = cryo_epsdr(imstack, samples_idx, max_d, verbose)

    r2 = np.zeros((2 * p - 1, 2 * p - 1))
    dsquare = np.square(x)
    for i in range(-max_d, max_d + 1):
        for j in range(-max_d, max_d + 1):
            d = i ** 2 + j ** 2
            if d <= max_d ** 2:
                idx, _ = bsearch(dsquare, d*(1-1e-13), d*(1+1e-13))
                if idx is None:
                    logger.warning('something went wrong in bsearch')
                r2[i+p-1, j+p-1] = r[idx-1]

    w = gwindow(p, max_d)
    p2 = cfft2(r2 * w)
    err = np.linalg.norm(p2.imag) / np.linalg.norm(p2)
    if err > 1e-12:
        logger.warning('Large imaginary components in P2 = {}'.format(err))

    p2 = p2.real

    e = 0
    for i in range(imstack.shape[2]):
        im = imstack[:, :, i]
        e += np.sum(np.square(im[samples_idx] - np.mean(im[samples_idx])))

    mean_e = e / (len(samples_idx[0]) * imstack.shape[2])
    p2 = (p2 / p2.sum()) * mean_e * p2.size
    neg_idx = np.where(p2 < 0)
    if len(neg_idx[0]) != 0:
        max_neg_err = np.max(np.abs(p2[neg_idx]))
        if max_neg_err > 1e-2:
            neg_norm = np.linalg.norm(p2[neg_idx])
            logger.warning('Power specrtum P2 has negative values with energy {}'.format(neg_norm))
        p2[neg_idx] = 0
    return p2, r, r2, x


def gwindow(p, max_d):
    x, y = np.meshgrid(np.arange(-(p-1), p), np.arange(-(p-1), p))
    alpha = 3.0
    w = np.exp(-alpha * (np.square(x) + np.square(y)) / (2 * max_d ** 2))
    return w


def cryo_epsdr(vol, samples_idx, max_d, verbose):
    p = vol.shape[0]
    k = vol.shape[2]
    i, j = np.meshgrid(np.arange(max_d + 1), np.arange(max_d + 1))
    dists = np.square(i) + np.square(j)
    dsquare = np.sort(np.unique(dists[np.where(dists <= max_d ** 2)]))

    corrs = np.zeros(len(dsquare))
    corr_count = np.zeros(len(dsquare))
    x = np.sqrt(dsquare)

    dist_map = np.zeros(dists.shape)
    for i in range(max_d + 1):
        for j in range(max_d + 1):
            d = i ** 2 + j ** 2
            if d <= max_d ** 2:
                idx, _ = bsearch(dsquare, d - 1e-13, d + 1e-13)
                if idx is None:
                    logger.warning('something went wrong in bsearch')
                dist_map[i, j] = idx

    dist_map = dist_map.astype('int') - 1
    valid_dists = np.where(dist_map != -1)

    mask = np.zeros((p, p))
    mask[samples_idx] = 1
    tmp = np.zeros((2 * p + 1, 2 * p + 1))
    tmp[:p, :p] = mask
    ftmp = np.fft.fft2(tmp)
    c = np.fft.ifft2(ftmp * np.conj(ftmp))
    c = c[:max_d+1, :max_d+1]
    c = np.round(c.real).astype('int')

    r = np.zeros(len(corrs))

    # optimized version
    vol = vol.transpose((2, 0, 1)).copy()
    input_fft2 = np.zeros((2 * p + 1, 2 * p + 1), dtype='complex128')
    output_fft2 = np.zeros((2 * p + 1, 2 * p + 1), dtype='complex128')
    input_ifft2 = np.zeros((2 * p + 1, 2 * p + 1), dtype='complex128')
    output_ifft2 = np.zeros((2 * p + 1, 2 * p + 1), dtype='complex128')
    flags = ('FFTW_MEASURE', 'FFTW_UNALIGNED')
    fft2 = pyfftw.FFTW(input_fft2, output_fft2, axes=(0, 1), direction='FFTW_FORWARD', flags=flags)
    ifft2 = pyfftw.FFTW(input_ifft2, output_ifft2, axes=(0, 1), direction='FFTW_BACKWARD', flags=flags)
    sum_s = np.zeros(output_ifft2.shape, output_ifft2.dtype)
    sum_c = c * vol.shape[0]
    for i in range(k):
        proj = vol[i]

        input_fft2[samples_idx] = proj[samples_idx]
        fft2()
        np.multiply(output_fft2, np.conj(output_fft2), out=input_ifft2)
        ifft2()
        sum_s += output_ifft2

    for curr_dist in zip(valid_dists[0], valid_dists[1]):
        dmidx = dist_map[curr_dist]
        corrs[dmidx] += sum_s[curr_dist].real
        corr_count[dmidx] += sum_c[curr_dist]

    idx = np.where(corr_count != 0)[0]
    r[idx] += corrs[idx] / corr_count[idx]
    cnt = corr_count[idx]

    idx = np.where(corr_count == 0)[0]
    r[idx] = 0
    x[idx] = 0
    return r, x, cnt


def bsearch(x, lower_bound, upper_bound):
    if lower_bound > x[-1] or upper_bound < x[0] or upper_bound < lower_bound:
        return None, None
    lower_idx_a = 1
    lower_idx_b = len(x)
    upper_idx_a = 1
    upper_idx_b = len(x)

    while lower_idx_a + 1 < lower_idx_b or upper_idx_a + 1 < upper_idx_b:
        lw = int(np.floor((lower_idx_a + lower_idx_b) / 2))
        if x[lw-1] >= lower_bound:
            lower_idx_b = lw
        else:
            lower_idx_a = lw
            if upper_idx_a < lw < upper_idx_b:
                upper_idx_a = lw

        up = int(np.ceil((upper_idx_a + upper_idx_b) / 2))
        if x[up-1] <= upper_bound:
            upper_idx_a = up
        else:
            upper_idx_b = up
            if lower_idx_a < up < lower_idx_b:
                lower_idx_b = up

    if x[lower_idx_a-1] >= lower_bound:
        lower_idx = lower_idx_a
    else:
        lower_idx = lower_idx_b
    if x[upper_idx_b-1] <= upper_bound:
        upper_idx = upper_idx_b
    else:
        upper_idx = upper_idx_a

    if upper_idx < lower_idx:
        return None, None

    return lower_idx, upper_idx


def cfft2(x):
    if len(x.shape) == 2:
        return np.fft.fftshift(np.transpose(np.fft.fft2(np.transpose(np.fft.ifftshift(x)))))
    elif len(x.shape) == 3:
        y = np.fft.ifftshift(x, (1, 2))
        y = np.transpose(y, (0, 2, 1))
        y = np.fft.fft2(y)
        y = np.transpose(y, (0, 2, 1))
        y = np.fft.fftshift(y, (1, 2))
        return y
    else:
        raise ValueError("x must be 2D or 3D")


def icfft2(x):
    if len(x.shape) == 2:
        return np.fft.fftshift(np.transpose(np.fft.ifft2(np.transpose(np.fft.ifftshift(x)))))
    elif len(x.shape) == 3:
        y = np.fft.ifftshift(x, (1, 2))
        y = np.transpose(y, (0, 2, 1))
        y = np.fft.ifft2(y)
        y = np.transpose(y, (0, 2, 1))
        y = np.fft.fftshift(y, (1, 2))
        return y
    else:
        raise ValueError("x must be 2D or 3D")


def fast_cfft2(x):
    if len(x.shape) == 2:
        return fftshift(np.transpose(fft2(np.transpose(ifftshift(x)))))
    elif len(x.shape) == 3:
        y = ifftshift(x, (1, 2))
        y = np.transpose(y, (0, 2, 1))
        y = fft2(y)
        y = np.transpose(y, (0, 2, 1))
        y = fftshift(y, (1, 2))
        return y
    else:
        raise ValueError("x must be 2D or 3D")


def fast_icfft2(x):
    if len(x.shape) == 2:
        return fftshift(np.transpose(ifft2(np.transpose(ifftshift(x)))))

    elif len(x.shape) == 3:
        y = ifftshift(x, (1, 2))
        y = np.transpose(y, (0, 2, 1))
        y = ifft2(y)
        y = np.transpose(y, (0, 2, 1))
        y = fftshift(y, (1, 2))
        return y

    else:
        raise ValueError("x must be 2D or 3D")
