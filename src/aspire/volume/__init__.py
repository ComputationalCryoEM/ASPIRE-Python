import numpy as np

import aspire.image
from aspire.nufft import nufft
from aspire.utils import ensure
from aspire.utils.coor_trans import grid_2d
from aspire.utils.fft import centered_fft2, centered_ifft2
from aspire.utils.matlab_compat import m_reshape
from aspire.utils.preprocess import downsample


class Volume:
    """
    Volume is an N x L x L x L array, along with associated utility methods.
    """
    def __init__(self, data):
        """
        Create a volume initialized with data.

        Volumes should be N x L x L x L,
        or L x L x L which implies N=1.

        :param data: Volume data

        :return: A volume instance.
        """

        if data.ndim == 3:
            data = data[np.newaxis, :, :, :]

        ensure(data.ndim == 4,
               'Volume data should be ndarray with shape NxLxLxL'
               ' or LxLxL.')

        ensure(data.shape[1] == data.shape[2] == data.shape[3],
               'Only cubed ndarrays are supported.')

        self.data = data
        self.n_vols = self.data.shape[0]
        self.dtype = self.data.dtype
        self.resolution = self.data.shape[1]
        self.shape = self.data.shape
        self.volume_shape = self.data.shape[1:]

    def __getitem__(self, item):
        # this is one reason why you might want Volume and VolumeStack classes...
        #return Volume(self.data[item])
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __len__(self):
        return self.n_vols

    def __add__(self, other):
        if isinstance(other, Volume):
            res = Volume(self.data + other.data)
        else:
            res = Volume(self.data + other)

        return res

    def __sub__(self, other):
        if isinstance(other, Volume):
            res = Volume(self.data - other.data)
        else:
            res = Volume(self.data - other)

        return res

    def __mul__(self, other):
        if isinstance(other, Volume):
            res = Volume(self.data * other.data)
        else:
            res = Volume(self.data * other)

        return res

    def project(self, vol_idx, rot_matrices):
        data = self[vol_idx].T  #RCOPT

        n = rot_matrices.shape[0]

        pts_rot = np.moveaxis(rotated_grids(self.resolution, rot_matrices), 1, 2)

        ## TODO: rotated_grids might as well give us correctly shaped array in the first place,
        pts_rot = m_reshape(pts_rot, (3, self.resolution**2*n))

        im_f = nufft(data, pts_rot) / self.resolution

        im_f = im_f.reshape(-1, self.resolution, self.resolution)

        if self.resolution % 2 == 0:
            im_f[:, 0, :] = 0
            im_f[:, :, 0] = 0

        im_f = centered_ifft2(im_f)

        return aspire.image.Image(np.real(im_f))

    def to_vec(self):
        """ Returns an N x resolution ** 3 array."""
        return m_reshape(self.data, (self.n_vols,) + (self.resolution**3,))
        #XXX reshape/flatten?

    @staticmethod
    def from_vec(vec):
        """ Returns a Volume instance from a N x resolution**3 array."""
        N = vec.shape[0]
        resolution = round(vec.shape[1] ** (1/3))
        assert resolution**3 == vec.shape[1]
        data = m_reshape(vec, (N,) + (resolution,)*3)
        return Volume(data)

    def downsample(self, szout, mask=None):
        if isinstance(szout, int):
            szout = (szout,)*3

        return Volume(downsample(self.data, szout, mask))

    def shift(self):
        raise NotImplementedError

    def rotate(self):
        raise NotImplementedError

    def denoise(self):
        raise NotImplementedError


class CartesianVolume(Volume):
    def expand(self, basis):
        return BasisVolume(basis)


class PolarVolume(Volume):
    def expand(self, basis):
        return BasisVolume(basis)


class BispecVolume(Volume):
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

def rotated_grids(L, rot_matrices):
    """
    Generate rotated Fourier grids in 3D from rotation matrices
    :param L: The resolution of the desired grids.
    :param rot_matrices: An array of size k-by-3-by-3 containing K rotation matrices
    :return: A set of rotated Fourier grids in three dimensions as specified by the rotation matrices.
        Frequencies are in the range [-pi, pi].
    """
    # TODO: Flattening and reshaping at end may not be necessary!
    grid2d = grid_2d(L)
    num_pts = L**2
    num_rots = rot_matrices.shape[0]
    pts = np.pi * np.vstack([grid2d['x'].flatten('F'), grid2d['y'].flatten('F'), np.zeros(num_pts)])
    pts_rot = np.zeros((3, num_pts, num_rots))
    for i in range(num_rots):
        pts_rot[:, :, i] = rot_matrices[i, :, :] @ pts

    pts_rot = m_reshape(pts_rot, (3, L, L, num_rots))
    return pts_rot
