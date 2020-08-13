import numpy as np

from aspire.image import Image
from aspire.nfft import Plan
from aspire.utils import ensure
from aspire.utils.coor_trans import grid_2d
from aspire.utils.fft import centered_fft2_C, centered_ifft2_C
from aspire.utils.fft import centered_fft2_F, centered_ifft2_F
from aspire.utils.matlab_compat import m_flatten, m_reshape


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
        self.N = self.data.shape[0]
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
        return self.N

    def __add__(self, other):
        return Volume(self.data + other.data)

    def __sub__(self, other):
        return Volume(self.data - other.data)

    # def __mul__(self, other):
    #     if isinstance(other, Volume):
    #         res = Volume(self.data * other.data)
    #     else:
    #         res = Volume(self.data * other)

    #     return res

    # def __truediv__(self, other):
    #     if isinstance(other, Volume):
    #         res = Volume(self.data / other.data)
    #     else:
    #         res = Volume(self.data / other)

    #     return res

    def _rotated_grids(self, rot_matrices):
        """
        Generate rotated Fourier grids in 3D from rotation matrices

        :param rot_matrices: An array of size k-by-3-by-3 containing K rotation matrices
        :return: A set of rotated Fourier grids in three dimensions as specified by the rotation matrices.
        Frequencies are in the range [-pi, pi].
        """
        grid2d = grid_2d(L)
        num_pts = self.resolution ** 2
        num_rots = rot_matrices.shape[0]
        pts = np.pi * np.vstack([grid2d['x'].flatten('F'), grid2d['y'].flatten('F'), np.zeros(num_pts)])
        pts_rot = np.zeros((3, num_pts, num_rots))
        for i in range(num_rots):
            pts_rot[:, :, i] = rot_matrices[i, :, :] @ pts
            #Note, previously pts_rot = m_reshape(pts_rot, (3, L, L, num_rots))
        return pts_rot


    def project(self, vol_idx, rot_matrices):
        data = self[vol_idx]

        n = rot_matrices.shape[0]

        pts_rot = rotated_grids(self.resolution, rot_matrices)

        # TODO: RCOPT come back and convert these methods internally to C order
        ## TODO: rotated_grids might as well give us correctly shaped array in the first place,
        #  yea... this and the related code look real funny to me.
        pts_rot = m_reshape(pts_rot, (3, self.resolution**2*n))

        im_f = (1./self.resolution *
                Plan(self.volume_shape, pts_rot).transform(data))

        im_f = m_reshape(im_f, (self.resolution, self.resolution, -1))

        if self.resolution % 2 == 0:
            im_f[0, :, :] = 0
            im_f[:, 0, :] = 0


        im = centered_ifft2_F(im_f)
        im_c = np.swapaxes(im, 0, -1)
        im_c = np.swapaxes(im_c, -2, -1)

        return Image(np.real(im_c))

    def to_vec(self):
        """ Returns an N x resolution ** 3 array."""
        return m_reshape(self.data, (self.N,) + (self.resolution**3,))
        #XXX reshape/flatten?

    @staticmethod
    def from_vec(vec):
        """ Returns a Volume instance from a N x resolution**3 array."""
        N = vec.shape[0]
        resolution = round(vec.shape[1] ** (1/3))
        assert resolution**3 == vec.shape[1]
        data = m_reshape(vec, (N,) + (resolution,)*3)
        return Volume(data)

    @staticmethod
    def from_backprojection(im, rot_matrices):
        """
        Backproject images along rotation
        :param im: An Image (stack) to backproject.
        :param rot_matrices: An n-by-3-by-3 array of rotation matrices \
        corresponding to viewing directions.

        :return: Volume instance corresonding to the backprojected images.
        """

        L = im.res

        ensure(im.n_images == rot_matrices.shape[0],
               "Number of rotation matrices must match the number of images")

        pts_rot = rotated_grids(L, rot_matrices)
        pts_rot = m_reshape(pts_rot, (3, -1))

        im_f = centered_fft2_C(im.data) / (L**2)
        if L % 2 == 0:
            im_f[:, 0, :] = 0
            im_f[:, :, 0] = 0

        # yikes
        im_f = np.swapaxes(im_f, -2, -1)
        im_f = im_f.flatten()

        plan = Plan(
            sz=(L, L, L),
            fourier_pts=pts_rot
        )
        vol = np.real(plan.adjoint(im_f)) / L

        return Volume(vol)

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


def im_backproject(im, rot_matrices):
    """
    Backproject images along rotation
    :param im: An L-by-L-by-n array of images to backproject.
    :param rot_matrices: An n-by-3-by-3 array of rotation matrices corresponding to viewing directions.
    :return: An L-by-L-by-L volumes corresponding to the sum of the backprojected images.
    """
    L, _, n = im.shape
    ensure(L == im.shape[1], "im must be LxLxK")
    ensure(n == rot_matrices.shape[0], "Number of rotation matrices must match the number of images")

    pts_rot = rotated_grids(L, rot_matrices)
    pts_rot = m_reshape(pts_rot, (3, -1))

    im_f = centered_fft2_F(im) / (L**2)
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
