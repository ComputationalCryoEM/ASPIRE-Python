"""
General purpose math functions, mostly geometric in nature.
"""

import numpy as np
from numpy.linalg import norm
from scipy.linalg import svd

from aspire.utils import ensure
from aspire.utils.random import Random


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


def grid_1d(n, shifted=False, normalized=True):
    """
    Generate one dimensional grid.

    :param n: the number of grid points.
    :param shifted: shifted by half of grid or not when n is even.
    :param normalized: normalize the grid in the range of (-1, 1) or not.
    :return: the rectangular and polar coordinates of all grid points.
    """

    grid = np.ceil(np.arange(-n / 2, n / 2))

    if shifted and n % 2 == 0:
        grid = np.arange(-n / 2 + 1 / 2, n / 2 + 1 / 2)

    if normalized:
        if shifted and n % 2 == 0:
            grid = grid / (n / 2 - 1 / 2)
        else:
            grid = grid / (n / 2)

    x = np.meshgrid(grid)
    r = x

    return {"x": x, "r": r}


def grid_2d(n, shifted=False, normalized=True, dtype=np.float32):
    """
    Generate two dimensional grid.

    :param n: the number of grid points in each dimension.
    :param shifted: shifted by half of grid or not when n is even.
    :param normalized: normalize the grid in the range of (-1, 1) or not.
    :return: the rectangular and polar coordinates of all grid points.
    """
    grid = np.ceil(np.arange(-n / 2, n / 2, dtype=dtype))

    if shifted and n % 2 == 0:
        grid = np.arange(-n / 2 + 1 / 2, n / 2 + 1 / 2, dtype=dtype)

    if normalized:
        if shifted and n % 2 == 0:
            grid = grid / (n / 2 - 1 / 2)
        else:
            grid = grid / (n / 2)

    x, y = np.meshgrid(grid, grid, indexing="ij")
    phi, r = cart2pol(x, y)

    return {"x": x, "y": y, "phi": phi, "r": r}


def grid_3d(n, shifted=False, normalized=True, dtype=np.float32):
    """
    Generate three dimensional grid.

    :param n: the number of grid points in each dimension.
    :param shifted: shifted by half of grid or not when n is even.
    :param normalized: normalize the grid in the range of (-1, 1) or not.
    :return: the rectangular and spherical coordinates of all grid points.
    """
    grid = np.ceil(np.arange(-n / 2, n / 2, dtype=dtype))

    if shifted and n % 2 == 0:
        grid = np.arange(-n / 2 + 1 / 2, n / 2 + 1 / 2, dtype=dtype)

    if normalized:
        if shifted and n % 2 == 0:
            grid = grid / (n / 2 - 1 / 2)
        else:
            grid = grid / (n / 2)

    x, y, z = np.meshgrid(grid, grid, grid, indexing="ij")
    phi, theta, r = cart2sph(x, y, z)

    # TODO: Should this theta adjustment be moved inside cart2sph?
    theta = np.pi / 2 - theta

    return {"x": x, "y": y, "z": z, "phi": phi, "theta": theta, "r": r}


def uniform_random_angles(n, seed=None, dtype=np.float32):
    """
    Generate random 3D rotation angles
    :param n: The number of rotation angles to generate
    :param seed: Random integer seed to use. If None, the current random state is used.
    :return: A n-by-3 ndarray of rotation angles
    """
    # Generate random rotation angles, in radians
    with Random(seed):
        angles = np.column_stack(
            (
                np.random.random(n) * 2 * np.pi,
                np.arccos(2 * np.random.random(n) - 1),
                np.random.random(n) * 2 * np.pi,
            )
        )
    return angles.astype(dtype)


def register_rotations(rots, rots_ref):
    """
    Register estimated orientations to reference ones.

    Finds the orthogonal transformation that best aligns the estimated rotations
    to the reference rotations.

    :param rots: The rotations to be aligned in the form of a n-by-3-by-3 array.
    :param rots_ref: The reference rotations to which we would like to align in
        the form of a n-by-3-by-3 array.
    :return: o_mat, optimal orthogonal 3x3 matrix to align the two sets;
            flag, flag==1 then J conjugacy is required and 0 is not.
    """

    ensure(
        rots.shape == rots_ref.shape, "Two sets of rotations must have same dimensions."
    )
    K = rots.shape[0]

    # Reflection matrix
    J = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])

    Q1 = np.zeros((3, 3), dtype=rots.dtype)
    Q2 = np.zeros((3, 3), dtype=rots.dtype)

    for k in range(K):
        R = rots[k, :, :]
        Rref = rots_ref[k, :, :]
        Q1 = Q1 + R @ Rref.T
        Q2 = Q2 + (J @ R @ J) @ Rref.T

    # Compute the two possible orthogonal matrices which register the
    # estimated rotations to the true ones.
    Q1 = Q1 / K
    Q2 = Q2 / K

    # We are registering one set of rotations (the estimated ones) to
    # another set of rotations (the true ones). Thus, the transformation
    # matrix between the two sets of rotations should be orthogonal. This
    # matrix is either Q1 if we recover the non-reflected solution, or Q2,
    # if we got the reflected one. In any case, one of them should be
    # orthogonal.

    err1 = norm(Q1 @ Q1.T - np.eye(3, dtype=rots.dtype), ord="fro")
    err2 = norm(Q2 @ Q2.T - np.eye(3, dtype=rots.dtype), ord="fro")

    # In any case, enforce the registering matrix O to be a rotation.
    if err1 < err2:
        # Use Q1 as the registering matrix
        U, _, V = svd(Q1)
        flag = 0
    else:
        # Use Q2 as the registering matrix
        U, _, V = svd(Q2)
        flag = 1

    Q_mat = U @ V

    return Q_mat, flag


def get_aligned_rotations(rots, Q_mat, flag):
    """
    Get aligned rotation matrices to reference ones.

    Calculated aligned rotation matrices from the orthogonal transformation
    that best aligns the estimated rotations to the reference rotations.


    :param rots: The reference rotations to which we would like to align in
        the form of a n-by-3-by-3 array.
    :param Q_mat:  optimal orthogonal 3x3 transformation matrix
    :param flag:  flag==1 then J conjugacy is required and 0 is not
    :return: regrot, aligned rotation matrices
    """

    K = rots.shape[0]

    # Reflection matrix
    J = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])

    regrot = np.zeros_like(rots)
    for k in range(K):
        R = rots[k, :, :]
        if flag == 1:
            R = J @ R @ J
        regrot[k, :, :] = Q_mat.T @ R

    return regrot


def get_rots_mse(rots_reg, rots_ref):
    """
    Calculate MSE between the estimated orientations to reference ones.

    :param rots_reg: The estimated rotations after alignment in the form of
        a n-by-3-by-3 array.
    :param rots_ref: The reference rotations.
    :return: The MSE value between two sets of rotations.
    """
    ensure(
        rots_reg.shape == rots_ref.shape,
        "Two sets of rotations must have same dimensions.",
    )
    K = rots_reg.shape[0]

    diff = np.zeros(K)
    mse = 0
    for k in range(K):
        diff[k] = norm(rots_reg[k, :, :] - rots_ref[k, :, :], ord="fro")
        mse += diff[k] ** 2
    mse = mse / K
    return mse


def common_line_from_rots(r1, r2, ell):
    """
    Compute the common line induced by rotation matrices r1 and r2.

    :param r1: The first rotation matrix of 3-by-3 array.
    :param r2: The second rotation matrix of 3-by-3 array.
    :param ell: The total number of common lines.
    :return: The common line indices for both first and second rotations.
    """

    assert r1.dtype == r2.dtype, "Ambiguous dtypes"

    ut = np.dot(r2, r1.T)
    alpha_ij = np.arctan2(ut[2, 0], -ut[2, 1]) + np.pi
    alpha_ji = np.arctan2(ut[0, 2], -ut[1, 2]) + np.pi

    ell_ij = alpha_ij * ell / (2 * np.pi)
    ell_ji = alpha_ji * ell / (2 * np.pi)

    ell_ij = int(np.mod(np.round(ell_ij), ell))
    ell_ji = int(np.mod(np.round(ell_ji), ell))

    return ell_ij, ell_ji
