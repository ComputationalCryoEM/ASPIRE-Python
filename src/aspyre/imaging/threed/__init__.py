import numpy as np

from aspyre.utils import ensure
from aspyre.utils.math import grid_2d
from aspyre.utils.fft import centered_ifft2, centered_fft2
from aspyre.utils.matlab_compat import m_reshape, m_flatten
from aspyre.nfft import Plan


class Volume(np.ndarray):
    """
    stack of volumes
    """
    def shift(self):
        raise NotImplementedError

    def rotate(self):
        raise NotImplementedError

    def denoise(self):
        raise NotImplementedError


class CartesianVolume(Volume):
    def expand(self, basis):
        return BasisVolume(basis)


class BasisVolume(Volume):
    def __init__(self, basis):
        self.basis = basis

    def evaluate(self):
        return CartesianVolume()


class FBBasisVolume(BasisVolume):
    pass


# TODO: The following functions likely all need to be moved inside the Volume class
def vol_project(vol, rot_matrices):
    L = vol.shape[0]
    n = rot_matrices.shape[-1]
    pts_rot = rotated_grids(L, rot_matrices)

    # TODO: rotated_grids might as well give us correctly shaped array in the first place
    pts_rot = m_reshape(pts_rot, (3, L**2*n))

    im_f = 1./L * Plan(vol.shape, pts_rot).transform(vol)
    im_f = m_reshape(im_f, (L, L, -1))

    if L % 2 == 0:
        im_f[0, :, :] = 0
        im_f[:, 0, :] = 0

    im = centered_ifft2(im_f)

    return np.real(im)


def rotated_grids(L, rot_matrices):
    """
    Generate rotated Fourier grids in 3D from rotation matrices
    :param L: The resolution of the desired grids.
    :param rot_matrices: An array of size 3-by-3-by-K containing K rotation matrices
    :return: A set of rotated Fourier grids in three dimensions as specified by the rotation matrices.
        Frequencies are in the range [-pi, pi].
    """
    # TODO: Flattening and reshaping at end may not be necessary!
    grid2d = grid_2d(L)
    num_pts = L**2
    num_rots = rot_matrices.shape[-1]
    pts = np.pi * np.vstack([grid2d['x'].flatten('F'), grid2d['y'].flatten('F'), np.zeros(num_pts)])
    pts_rot = np.zeros((3, num_pts, num_rots))
    for i in range(num_rots):
        pts_rot[:, :, i] = rot_matrices[:, :, i] @ pts

    pts_rot = m_reshape(pts_rot, (3, L, L, num_rots))
    return pts_rot


def im_backproject(im, rot_matrices):
    """
    Backproject images along rotation
    :param im: An L-by-L-by-n array of images to backproject.
    :param rot_matrices: An 3-by-3-by-n array of rotation matrices corresponding to viewing directions.
    :return: An L-by-L-by-L volumes corresponding to the sum of the backprojected images.
    """
    L, _, n = im.shape
    ensure(L == im.shape[1], "im must be LxLxK")
    ensure(n == rot_matrices.shape[2], "No. of rotation matrices must match the number of images")

    pts_rot = rotated_grids(L, rot_matrices)
    pts_rot = m_reshape(pts_rot, (3, -1))

    im_f = centered_fft2(im) / (L**2)
    if L % 2 == 0:
        im_f[0, :, :] = 0
        im_f[:, 0, :] = 0
    im_f = m_flatten(im_f)

    plan = Plan(
        sz=(L, L, L),
        fourier_pts=pts_rot
    )
    vol = np.real(plan.adjoint(im_f)) / L

    return vol
