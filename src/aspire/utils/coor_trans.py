"""
General purpose math functions, mostly geometric in nature.
"""

import numpy as np
from numpy.linalg import norm
from scipy.linalg import svd

from aspire.utils.matlab_compat import randn, Random
from aspire.utils import ensure


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

    grid = np.ceil(np.arange(-n/2, n/2))

    if shifted and n % 2 == 0:
        grid = np.arange(-n/2+1/2, n/2+1/2)

    if normalized:
        if shifted and n % 2 == 0:
            grid = grid / (n/2-1/2)
        else:
            grid = grid / (n/2)

    x = np.meshgrid(grid)
    r = x

    return {
        'x': x,
        'r': r
    }


def grid_2d(n, shifted=False, normalized=True):
    """
    Generate two dimensional grid.

    :param n: the number of grid points in each dimension.
    :param shifted: shifted by half of grid or not when n is even.
    :param normalized: normalize the grid in the range of (-1, 1) or not.
    :return: the rectangular and polar coordinates of all grid points.
    """
    grid = np.ceil(np.arange(-n/2, n/2))

    if shifted and n % 2 == 0:
        grid = np.arange(-n/2+1/2, n/2+1/2)

    if normalized:
        if shifted and n % 2 == 0:
            grid = grid / (n/2-1/2)
        else:
            grid = grid / (n/2)

    x, y = np.meshgrid(grid, grid, indexing='ij')
    phi, r = cart2pol(x, y)

    return {
        'x': x,
        'y': y,
        'phi': phi,
        'r': r
    }


def grid_3d(n, shifted=False, normalized=True):
    """
    Generate three dimensional grid.

    :param n: the number of grid points in each dimension.
    :param shifted: shifted by half of grid or not when n is even.
    :param normalized: normalize the grid in the range of (-1, 1) or not.
    :return: the rectangular and spherical coordinates of all grid points.
    """
    grid = np.ceil(np.arange(-n/2, n/2))

    if shifted and n % 2 == 0:
        grid = np.arange(-n/2+1/2, n/2+1/2)

    if normalized:
        if shifted and n % 2 == 0:
            grid = grid / (n/2-1/2)
        else:
            grid = grid / (n/2)

    x, y, z = np.meshgrid(grid, grid, grid, indexing='ij')
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


def register_rotations(rots, rots_ref):
    """
    Register estimated orientations to reference ones.

    Finds the orthogonal transformation that best aligns the estimated rotations
    to the reference rotations. `regrot` are the estimated rotations after registering
    them to the reference ones, O is the optimal orthogonal 3x3 matrix to align
    the two sets. If flag==2 then J conjugacy is required.

    :param rots: The rotations to be aligned in the form of a 3-by-3-by-n array.
    :param rots_ref: The reference rotations to which we would like to align in
        the form of a 3-by-3-by-n array.
    :return: regrot, aligned rotation matrices;
             mse, mean squired error between the estimated and aligned matrices;
             diff, difference array between the estimated and aligned matrices;
             o_mat, optimal orthogonal 3x3 matrix to align the two sets;
            flag, flag==2 then J conjugacy is required.
    """

    ensure(rots.shape == rots_ref.shape,
           'Two sets of rotations must have same dimensions.')
    K = rots.shape[2]

    # Reflection matrix
    J = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])

    O1 = np.zeros((3, 3), dtype=rots.dtype)
    O2 = np.zeros((3, 3), dtype=rots.dtype)

    for k in range(K):
        R = rots[:, :, k]
        Rref = rots_ref[:, :, k]
        O1 = O1+R @ Rref.T
        O2 = O2+(J @ R @ J) @ Rref.T

    # Compute the two possible orthogonal matrices which register the
    # estimated rotations to the true ones.
    O1 = O1/K
    O2 = O2/K

    # We are registering one set of rotations (the estimated ones) to
    # another set of rotations (the true ones). Thus, the transformation
    # matrix between the two sets of rotations should be orthogonal. This
    # matrix is either O1 if we recover the non-reflected solution, or O2,
    # if we got the reflected one. In any case, one of them should be
    # orthogonal.

    err1 = norm(O1@O1.T - np.eye(3), ord='fro')
    err2 = norm(O2@O2.T - np.eye(3), ord='fro')

    # In any case, enforce the registering matrix O to be a rotation.
    if err1 < err2:
        # Use O1 as the registering matrix
        U, _, V = svd(O1)
        flag = 1
    else:
        # Use O2 as the registering matrix
        U, _, V = svd(O2)
        flag = 2

    o_mat = U @ V
    # estimate errors
    diff = np.zeros((K, 1))
    mse = 0
    regrot = np.zeros_like(rots)
    for k in range(K):
        R = rots[:, :, k]
        Rref = rots_ref[:, :, k]
        if flag == 2:
            R = J @ R @ J
        regrot[:, :, k] = o_mat.T @ R
        diff[k] = norm(o_mat.T @ R - Rref, ord='fro')
        mse = mse + diff[k]**2
    mse = mse/K

    return regrot, mse, diff, o_mat, flag
