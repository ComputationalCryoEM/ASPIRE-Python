"""
General purpose math functions, mostly geometric in nature.
"""

import logging
import math
from functools import lru_cache

import numpy as np

from aspire import config
from aspire.numeric import xp
from aspire.utils.rotation import Rotation

logger = logging.getLogger(__name__)


def cart2pol(x, y):
    """
    Convert Cartesian to Polar Coordinates. All input arguments must be the same shape.

    :param x: x-coordinate in Cartesian space
    :param y: y-coordinate in Cartesian space
    :return: A 2-tuple of values:
        theta: angular coordinate/azimuth
        r: radial distance from origin
    """
    return np.arctan2(y, x), np.hypot(x, y)


def cart2sph(x, y, z):
    """
    Transform cartesian coordinates to spherical. All input arguments must be the same shape.

    :param x: X-values of input co-ordinates.
    :param y: Y-values of input co-ordinates.
    :param z: Z-values of input co-ordinates.
    :return: A 3-tuple of values, all of the same shape as the inputs.
        (<azimuth>, <elevation>, <radius>)
        azimuth and elevation are returned in radians.

    This function is equivalent to MATLAB's cart2sph function.
    """
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r


def _mgrid_slice(n, shifted, normalized):
    """
    Util to generate a `slice` representing a 1d linspace
    as expected by `np.mgrid`.

    :param shifted: shifted by half of grid or not when n is even.
    :param normalized: normalize the grid in the range of (-1, 1) or not.
    :return: `slice` to be used by `np.mgrid`.
    """

    num_points = n * 1j
    start = -n // 2 + 1
    end = n // 2

    if shifted and n % 2 == 0:
        start -= 1 / 2
        end -= 1 / 2
    elif n % 2 == 0:
        start -= 1
        end -= 1

    if normalized:
        start /= n / 2
        end /= n / 2

    return slice(start, end, num_points)


@lru_cache(maxsize=config["cache"]["grid_cache_size"].get())
def grid_1d(n, shifted=False, normalized=True, dtype=np.float32):
    """
    Generate one dimensional grid.

    :param n: the number of grid points.
    :param shifted: shifted by half of grid or not when n is even.
    :param normalized: normalize the grid in the range of (-1, 1) or not.
    :return: the rectangular and polar coordinates of all grid points.
    """

    r = x = np.mgrid[_mgrid_slice(n, shifted, normalized)].astype(dtype)

    return {"x": x, "r": r}


@lru_cache(maxsize=config["cache"]["grid_cache_size"].get())
def grid_2d(n, shifted=False, normalized=True, indexing="yx", dtype=np.float32):
    """
    Generate two dimensional grid.

    :param n: the number of grid points in each dimension.
        May be a single integer value or 2-tuple of integers.
    :param shifted: shifted by half of grid or not when n is even.
    :param normalized: normalize the grid in the range of (-1, 1) or not.
    :param indexing: 'yx' (C) or 'xy' (F), defaulting to 'yx'.
        See https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
    :return: the rectangular and polar coordinates of all grid points.
    """

    if isinstance(n, int):
        n = (n, n)
    rows, cols = n

    rows = _mgrid_slice(rows, shifted, normalized)
    cols = _mgrid_slice(cols, shifted, normalized)

    y, x = np.mgrid[rows, cols].astype(dtype)
    if indexing == "xy":
        x, y = y, x
    elif indexing != "yx":
        raise RuntimeError(
            f"grid_2d indexing {indexing} not supported." "  Try 'xy' or 'yx'"
        )

    phi, r = cart2pol(x, y)

    return {"x": x, "y": y, "phi": phi, "r": r}


@lru_cache(maxsize=config["cache"]["grid_cache_size"].get())
def grid_3d(n, shifted=False, normalized=True, indexing="zyx", dtype=np.float32):
    """
    Generate three dimensional grid.

    :param n: the number of grid points in each dimension.
    :param shifted: shifted by half of grid or not when n is even.
    :param normalized: normalize the grid in the range of (-1, 1) or not.
    :param indexing: 'zyx' (C) or 'xyz' (F), defaulting to 'zyx'.
        See https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
    :return: the rectangular and spherical coordinates of all grid points.
    """

    grid = _mgrid_slice(n, shifted, normalized)
    z, y, x = np.mgrid[grid, grid, grid].astype(dtype)

    if indexing == "xyz":
        x, y, z = z, y, x
    elif indexing != "zyx":
        raise RuntimeError(
            f"grid_3d indexing {indexing} not supported." "  Try 'xyz' or 'zyx'"
        )

    phi, theta, r = cart2sph(x, y, z)

    # TODO: Should this theta adjustment be moved inside cart2sph?
    theta = np.pi / 2 - theta

    return {"x": x, "y": y, "z": z, "phi": phi, "theta": theta, "r": r}


def mean_aligned_angular_distance(rots_est, rots_gt, degree_tol=None):
    """
    Register estimates to ground truth rotations and compute the
    mean angular distance between them (in degrees).

    :param rots_est: A set of estimated rotations. A Rotation object or
        array of size nx3x3.
    :param rots_gt: A set of ground truth rotations. A Rotation object or
        array of size nx3x3.
    :param degree_tol: Option to assert if the mean angular distance is
        less than `degree_tol` degrees. If `None`, returns the mean
        aligned angular distance.

    :return: The mean angular distance between registered estimates
        and the ground truth (in degrees).
    """
    if not isinstance(rots_est, Rotation):
        rots_est = Rotation(rots_est)
    if not isinstance(rots_gt, Rotation):
        rots_gt = Rotation(rots_gt)

    Q_mat, flag = rots_est.find_registration(rots_gt)
    logger.debug(f"Registration Q_mat: {Q_mat}\nflag: {flag}")
    regrot = rots_est.apply_registration(Q_mat, flag)
    mean_ang_dist = Rotation.mean_angular_distance(regrot, rots_gt) * 180 / np.pi

    if degree_tol is not None:
        np.testing.assert_array_less(mean_ang_dist, degree_tol)

    return mean_ang_dist


def rots_to_clmatrix(rots, n_theta):
    """
    Compute the common lines matrix induced by all pairs of rotation
    matrices, `rots`, provided.

    :param rots: n_rotsx3x3 array of rotation matrices.
    :param n_theta: Number of theta values for common lines indices.

    :return: n_rots x n_rots common lines matrix.
    """
    n_rots = len(rots)
    cl_matrix = -np.ones((n_rots, n_rots))
    for i in range(n_rots):
        Ri = rots[i]
        Ri3 = Ri[:, 2]
        for j in range(i + 1, n_rots):
            Rj = rots[j]
            Rj3 = Rj[:, 2]
            common_axis = np.cross(Ri3, Rj3) / np.linalg.norm(np.cross(Ri3, Rj3))
            xij = Ri.T @ common_axis
            xji = Rj.T @ common_axis
            theta_ij = np.rad2deg(np.arctan2(xij[1], xij[0])) % 360
            theta_ji = np.rad2deg(np.arctan2(xji[1], xji[0])) % 360

            if theta_ij > 180:
                theta_ij -= 180
                theta_ji -= 180

            cl_matrix[i, j] = round(theta_ij * n_theta / 360)
            cl_matrix[j, i] = round((theta_ji % 360) * n_theta / 360)

    return cl_matrix


def crop_pad_2d(im, size, fill_value=0):
    """
    Crop/pads `im` according to `size`.

    Padding will use `fill_value`.
    Return's host/GPU array based on `im`.

    :param im: A >=2-dimensional numpy array
    :param size: Integer size of cropped/padded output
    :return: Array of shape (..., size, size)
    """

    im_y, im_x = im.shape[-2:]
    # shift terms
    start_x = math.floor(im_x / 2) - math.floor(size / 2)
    start_y = math.floor(im_y / 2) - math.floor(size / 2)

    # cropping
    if size <= min(im_y, im_x):
        return im[..., start_y : start_y + size, start_x : start_x + size]
    # padding
    elif size >= max(im_y, im_x):
        # Determine shape
        shape = list(im.shape[:-2])
        shape.extend([size, size])

        # Ensure that we return the same dtype as the input
        _full = np.full  # Default to numpy array
        if isinstance(im, xp.ndarray):
            # Use cupy when `im` _and_ xp are cupy ndarray
            # Avoids having to handle when cupy is not installed
            _full = xp.full

        to_return = _full(shape, fill_value, dtype=im.dtype)

        # when padding, start_x and start_y are negative since size is larger
        # than im_x and im_y; the below line calculates where the original image
        # is placed in relation to the (now-larger) box size
        to_return[..., -start_y : im_y - start_y, -start_x : im_x - start_x] = im
        return to_return
    else:
        # target size is between mat_x and mat_y
        raise ValueError(
            "Cannot crop and pad Image at the same time."
            "If this is really what you intended,"
            " make two seperate calls for cropping and padding."
        )


def crop_pad_3d(vol, size, fill_value=0):
    """
    Crop/pads `vol` according to `size`.

    Padding will use `fill_value`.
    Return's host/GPU array based on `vol`.

    :param vol: A >=3-dimensional numpy array
    :param size: Integer size of cropped/padded output
    :return: Array of shape (..., size, size, size)
    """

    vol_z, vol_y, vol_x = vol.shape[-3:]
    # shift terms
    start_z = math.floor(vol_z / 2) - math.floor(size / 2)
    start_y = math.floor(vol_y / 2) - math.floor(size / 2)
    start_x = math.floor(vol_x / 2) - math.floor(size / 2)

    # cropping
    if size <= min(vol_z, vol_y, vol_x):
        return vol[
            ...,
            start_z : start_z + size,
            start_y : start_y + size,
            start_x : start_x + size,
        ]
    # padding
    elif size >= max(vol_z, vol_y, vol_x):
        # Determine shape
        shape = list(vol.shape[:-3])
        shape.extend([size, size, size])

        # Ensure that we return the same dtype as the input
        _full = np.full  # Default to numpy array
        if isinstance(vol, xp.ndarray):
            # Use cupy when `vol` _and_ xp are cupy ndarray
            # Avoids having to handle when cupy is not installed
            _full = xp.full

        to_return = _full(shape, fill_value, dtype=vol.dtype)

        to_return[
            ...,
            -start_z : vol_z - start_z,
            -start_y : vol_y - start_y,
            -start_x : vol_x - start_x,
        ] = vol
        return to_return
    else:
        # target size is between min and max of (vol_x, vol_y, vol_z)
        raise ValueError(
            "Cannot crop and pad Volume at the same time."
            "If this is really what you intended,"
            " make two seperate calls for cropping and padding."
        )
