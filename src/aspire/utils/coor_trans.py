"""
General purpose math functions, mostly geometric in nature.
"""

import numpy as np
from aspire.utils.matlab_compat import randn, Random


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


def grid_1d(n):
    grid = np.ceil(np.arange(-n/2, n/2)) / (n/2)
    x = np.meshgrid(grid)
    r = x

    return {
        'x': x,
        'r': r
    }


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


def cgrid_2d(n):
    if n % 2 == 1:
        grid_1d = np.arange(-(n-1)/2, (n-1)/2+1)
    else:
        grid_1d = np.arange(-n/2+1/2, n/2+1/2)
    x, y = np.meshgrid(grid_1d, grid_1d, indexing='ij')

    return {
        'x': x,
        'y': y
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


def qrand(nrot, seed=0):

    """
    Generate a set of quaternions from random normal distribution.

    Each quaternions is a four-elements column vector. Returns a matrix of
    size 4xn.

    The 3-sphere S^3 in R^4 is a double cover of the rotation group SO(3), SO(3) = RP^3.
    We identify unit norm quaternions a^2+b^2+c^2+d^2=1 with group elements.
    The antipodal points (-a,-b,-c,-d) and (a,b,c,d) are identified as the same group elements,
    so we take a>=0.
    :param nrot: The number of quaternions for rotations.
    :param seed: The random seed.
    :return: An array consists of 4 dimensions quaternions
    """
    q = randn(4, nrot, seed=seed)
    l2_norm = np.sqrt(q[0, :]**2 + q[1, :]**2 + q[2, :]**2 + q[3, :]**2)
    for i in range(4):
        q[i, :] = q[i, :] / l2_norm

    for k in range(nrot):
        if q[0, k] < 0:
            q[:, k] = -q[:, k]

    return q


def q_to_rot(q):
    """
    Convert the quaternions into a rotation matrices.

    :param q: Array of quaternions. May be a vector of dimensions 4 x n
    :return rot_mat: n-by-3-by-3 array of 3x3 rotation matrices.
    """

    nrot = np.size(q, 1)
    rot_mat = np.zeros((nrot, 3, 3), dtype=q.dtype)

    rot_mat[:, 0, 0] = q[0, :]**2 + q[1, :]**2 - q[2, :]**2 - q[3, :]**2
    rot_mat[:, 0, 1] = 2*q[1, :]*q[2, :] - 2*q[0, :]*q[3, :]
    rot_mat[:, 0, 2] = 2*q[0, :]*q[2, :] + 2*q[1, :]*q[3, :]

    rot_mat[:, 1, 0] = 2*q[1, :]*q[2, :] + 2*q[0, :]*q[3, :]
    rot_mat[:, 1, 1] = q[0, :]**2 - q[1, :]**2 + q[2, :]**2 - q[3, :]**2
    rot_mat[:, 1, 2] = -2*q[0, :]*q[1, :] + 2*q[2, :]*q[3, :]

    rot_mat[:, 2, 0] = -2*q[0, :]*q[2, :] + 2*q[1, :]*q[3, :]
    rot_mat[:, 2, 1] = 2*q[0, :]*q[1, :] + 2*q[2, :]*q[3, :]
    rot_mat[:, 2, 2] = q[0, :]**2 - q[1, :]**2 - q[2, :]**2 + q[3, :]**2
    return rot_mat


def qrand_rots(nrot, seed=0):
    """
    Generate random rotations from quaternions

    :param nrot: The totalArray of quaternions. May be a vector of dimensions 4 x n
    :return: Array of 3x3 rotation matrices.
    """
    qs = qrand(nrot, seed)

    return q_to_rot(qs)


def uniform_random_angles(n, seed=None):
    """
    Generate random 3D rotation angles
    :param n: The number of rotation angles to generate
    :param seed: Random integer seed to use. If None, the current random state is used.
    :return: A n-by-3 ndarray of rotation angles
    """
    # Generate random rotation angles, in radians
    with Random(seed):
        angles = np.column_stack((
            np.random.random(n) * 2 * np.pi,
            np.arccos(2 * np.random.random(n) - 1),
            np.random.random(n) * 2 * np.pi
        ))
    return angles
