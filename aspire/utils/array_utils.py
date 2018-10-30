"""
This module contains functions useful for common tasks such as crop/mask/downsample etc.
It is different from the functions in PreProcessor as they allow you to crop/downsample to a
non square size. PreProcessor.crop accepts size as int and here it's accepted as (X,Y) tuple.
TODO reduced this file by refactoring it and moving crop/downsample capabilities to PreProcessor.
"""
import numpy as np
import pyfftw
from scipy.special._ufuncs import erf


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

    theta = np.arctan2(y, x)

    return radius, theta