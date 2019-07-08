"""
General purpose math functions, mostly geometric in nature.
"""

import numpy as np


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


def grid_2d(n):
    grid_1d = np.ceil(np.arange(-n/2, n/2)) / (n/2)
    x, y = np.meshgrid(grid_1d, grid_1d, indexing='ij')
    phi, r = cart2pol(x, y)

    return {
        'x': x,
        'y': y,
        'phi': phi,
        'r': r
    }


def grid_3d(n):
    grid_1d = np.ceil(np.arange(-n/2, n/2)) / (n/2)
    x, y, z = np.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')
    phi, theta, r = cart2sph(x, y, z)

    # TODO: Should this theta adjustment be moved inside cart2sph?
    theta = np.pi/2 - theta

    return {
        'x': x,
        'y': y,
        'z': z,
        'phi': phi,
        'theta': theta,
        'r': r
    }


def angles_to_rots(angles):
    n_angles = angles.shape[-1]
    rots = np.zeros(shape=(3, 3, n_angles))

    for i in range(n_angles):
        rots[:, :, i] = erot(angles[:, i])
    return rots


def erot(angles):
    return zrot(angles[0]) @ yrot(angles[1]) @ zrot(angles[2])


def zrot(theta):
    sin, cos = np.sin(theta), np.cos(theta)
    return np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])


def yrot(theta):
    sin, cos = np.sin(theta), np.cos(theta)
    return np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
