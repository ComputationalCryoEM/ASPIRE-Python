import logging

import mrcfile
import numpy as np
from numpy.linalg import qr

import aspire.image
from aspire.nufft import nufft
from aspire.numeric import fft, xp
from aspire.utils import Rotation, crop_pad_3d, grid_2d, grid_3d, mat_to_vec, vec_to_mat
from aspire.utils.types import complex_type

logger = logging.getLogger(__name__)


def qr_vols_forward(sim, s, n, vols, k):
    """
    TODO: Write docstring
    TODO: Find a better place for this!

    :param sim:
    :param s:
    :param n:
    :param vols:
    :param k:
    :return:
    """
    ims = np.zeros((k, n, sim.L, sim.L), dtype=vols.dtype)
    for ell in range(k):
        ims[ell] = sim.vol_forward(Volume(vols[ell]), s, n).asnumpy()

    ims = np.swapaxes(ims, 1, 3)
    ims = np.swapaxes(ims, 0, 2)

    Q_vecs = np.zeros((sim.L**2, k, n), dtype=vols.dtype)
    Rs = np.zeros((k, k, n), dtype=vols.dtype)

    im_vecs = mat_to_vec(ims)
    for i in range(n):
        Q_vecs[:, :, i], Rs[:, :, i] = qr(im_vecs[:, :, i])
    Qs = vec_to_mat(Q_vecs)

    return Qs, Rs


class Volume:
    """
    Volume is an (N1 x ...) x L x L x L array, along with associated utility methods.
    """

    def __init__(self, data, dtype=None):
        """
        A stack of one or more volumes.

        This is a wrapper of numpy.ndarray which provides methods
        for common processing tasks.

        The stack can be multidimensional with `n_vols` equal
        to the product of the stack dimensions.  Singletons will be
        expanded into a stack with one entry.

        The last three axes represent the volume size,
        and are checked to be cubic.

        :param data: Numpy array containing volume data with shape
            `(..., resolution, resolution, resolution)`.
        :param dtype: Optionally cast `data` to this dtype.
            Defaults to `data.dtype`.

        :return: A Volume instance holding `data`.
        """

        if not isinstance(data, np.ndarray):
            raise ValueError("Volume should be instantiated with an ndarray")

        if data.ndim < 3:
            raise ValueError(
                "Volume data should be ndarray with shape (N1...)xLxLxL or LxLxL."
            )
        elif data.ndim == 3:
            data = np.expand_dims(data, axis=0)

        if dtype is None:
            self.dtype = data.dtype
        else:
            self.dtype = np.dtype(dtype)

        if not (data.shape[-1] == data.shape[-2] == data.shape[-3]):
            raise ValueError("Only cubed ndarrays are supported.")

        self._data = data.astype(self.dtype, copy=False)
        self.ndim = self._data.ndim
        self.shape = self._data.shape
        self.stack_ndim = self._data.ndim - 3
        self.stack_shape = self._data.shape[:-3]
        self.n_vols = np.prod(self.stack_shape)
        self.resolution = self._data.shape[-1]

    def asnumpy(self):
        """
        Return volume as a (n_vols, resolution, resolution, resolution) array.

        :return: ndarray
        """
        return self._data

    def astype(self, dtype, copy=True):
        """
        Return `Volume` instance with the prescribed dtype.

        :param dtype: Numpy dtype
        :param copy: Boolean, optionally avoid copying if Volume.dtype already matches.
            Defaults to True.
        :return: Volume instance
        """
        return Volume(self.asnumpy().astype(dtype, copy=copy))

    def _check_key_dims(self, key):
        if isinstance(key, tuple) and (len(key) > self._data.ndim):
            raise ValueError(
                f"Volume stack_dim is {self.stack_ndim}, slice length must be =< {self.ndim}"
            )

    def __getitem__(self, key):
        self._check_key_dims(key)
        return self._data[key]

    def __setitem__(self, key, value):
        self._check_key_dims(key)
        self._data[key] = value

    def stack_reshape(self, *args):
        """
        Reshape the stack axis.

        :*args: Integer(s) or tuple describing the intended shape.

        :returns: Volume instance
        """

        # If we're passed a tuple, use that
        if len(args) == 1 and isinstance(args[0], tuple):
            shape = args[0]
        else:
            # Otherwise use the variadic args
            shape = args

        # Sanity check the size
        if shape != (-1,) and np.prod(shape) != self.n_vols:
            raise ValueError(
                f"Number of volumes {self.n_vols} cannot be reshaped to {shape}."
            )

        return Volume(self._data.reshape(*shape, *self._data.shape[-3:]))

    def __repr__(self):
        msg = (
            f"{self.n_vols} {self.dtype} volumes arranged as a {self.stack_shape} stack"
        )
        msg += f" each of size {self.resolution}x{self.resolution}x{self.resolution}."
        return msg

    def __len__(self):
        return self.n_vols

    def __add__(self, other):
        if isinstance(other, Volume):
            res = Volume(self._data + other.asnumpy())
        else:
            res = Volume(self._data + other)

        return res

    def __radd__(self, otherL):
        return self + otherL

    def __sub__(self, other):
        if isinstance(other, Volume):
            res = Volume(self._data - other.asnumpy())
        else:
            res = Volume(self._data - other)

        return res

    def __rsub__(self, otherL):
        return Volume(otherL - self._data)

    def __mul__(self, other):
        if isinstance(other, Volume):
            res = Volume(self._data * other.asnumpy())
        else:
            res = Volume(self._data * other)

        return res

    def __rmul__(self, otherL):
        return self * otherL

    def project(self, vol_idx, rot_matrices):
        """
        Using the stack of rot_matrices,
        project images of Volume[vol_idx].

        :param vol_idx: Volume index
        :param rot_matrices: Stack of rotations. Rotation or ndarray instance.
        :return: `Image` instance.
        """
        # See Issue #727
        if self.stack_ndim > 1:
            raise NotImplementedError(
                "`project` is currently limited to 1D Volume stacks."
            )

        # If we are an ASPIRE Rotation, get the numpy representation.
        if isinstance(rot_matrices, Rotation):
            rot_matrices = rot_matrices.matrices

        if rot_matrices.dtype != self.dtype:
            logger.warning(
                f"{self.__class__.__name__}"
                f" rot_matrices.dtype {rot_matrices.dtype}"
                f" != self.dtype {self.dtype}."
                " In the future this will raise an error."
            )

        data = self[vol_idx]

        n = rot_matrices.shape[0]

        pts_rot = rotated_grids(self.resolution, rot_matrices)

        # TODO: rotated_grids might as well give us correctly shaped array in the first place
        pts_rot = pts_rot.reshape((3, n * self.resolution**2))

        im_f = nufft(data, pts_rot) / self.resolution

        im_f = im_f.reshape(-1, self.resolution, self.resolution)

        if self.resolution % 2 == 0:
            im_f[:, 0, :] = 0
            im_f[:, :, 0] = 0

        im_f = xp.asnumpy(fft.centered_ifft2(xp.asarray(im_f)))

        return aspire.image.Image(np.real(im_f))

    def to_vec(self):
        """Returns an N x resolution ** 3 array."""
        return self._data.reshape((self.n_vols, self.resolution**3))

    @staticmethod
    def from_vec(vec):
        """
        Returns a Volume instance from a (N, resolution**3) array or
        (resolution**3) array.

        :return: Volume instance.
        """

        if vec.ndim == 1:
            vec = vec[np.newaxis, :]

        n_vols = vec.shape[0]

        resolution = round(vec.shape[1] ** (1 / 3))
        assert resolution**3 == vec.shape[1]

        data = vec.reshape((n_vols, resolution, resolution, resolution))

        return Volume(data)

    def transpose(self):
        """
        Returns a new Volume instance with volume data axes tranposed.

        :return: Volume instance.
        """
        original_stack_shape = self.stack_shape
        v = self.stack_reshape(-1)
        vt = np.transpose(v._data, (0, -1, -2, -3))
        return Volume(vt).stack_reshape(original_stack_shape)

    @property
    def T(self):
        """
        Abbreviation for transpose.

        :return: Volume instance.
        """

        return self.transpose()

    def flatten(self):
        """
        Util function for flatten operation on volume data array.

        :return: ndarray
        """

        return self._data.flatten()

    def flip(self, axis=-3):
        """
        Flip volume stack data along axis using numpy.flip

        :param axis: Optionally specify axis as integer or tuple.
            Defaults to axis=-3.

        :return: Volume instance.
        """
        # Convert integer to tuple, so we can always loop.
        if isinstance(axis, int):
            axis = (axis,)

        for ax in axis:
            ax = ax % self.ndim  # modulo [0, ndim)
            if ax < self.stack_ndim:
                raise ValueError(
                    f"Cannot flip axis {ax}: stack axis. Did you mean {ax-4}?"
                )

        return Volume(np.flip(self._data, axis))

    def downsample(self, ds_res, mask=None):
        """
        Downsample each volume to a desired resolution (only cubic supported).

        :param ds_res: Desired resolution.
        :param mask: Optional NumPy array mask to multiply in Fourier space.
        """
        if mask is None:
            mask = 1.0

        original_stack_shape = self.stack_shape
        v = self.stack_reshape(-1)

        # take 3D Fourier transform of each volume in the stack
        fx = fft.fftshift(fft.fftn(v._data, axes=(1, 2, 3)))
        # crop each volume to the desired resolution in frequency space
        crop_fx = (
            np.array([crop_pad_3d(fx[i, :, :, :], ds_res) for i in range(self.n_vols)])
            * mask
        )
        # inverse Fourier transform of each volume
        out = fft.ifftn(fft.ifftshift(crop_fx), axes=(1, 2, 3)) * (
            ds_res**3 / self.resolution**3
        )
        # returns a new Volume object
        return Volume(np.real(out)).stack_reshape(original_stack_shape)

    def shift(self):
        raise NotImplementedError

    def rotate(self, rot_matrices, zero_nyquist=True):
        """
        Rotate volumes using a `Rotation` object. If the `Rotation` object
        is a single rotation, each volume will be rotated by that rotation.
        If the `Rotation` object is a stack of rotations of length n_vols,
        the ith volume is rotated by the ith rotation.

        :param rot_matrices: `Rotation` object of length 1 or n_vols.
        :param zero_nyquist: Option to keep or remove Nyquist frequency for even resolution.
            Defaults to zero_nyquist=True, removing the Nyquist frequency.

        :return: `Volume` instance.
        """
        if self.stack_ndim > 1:
            raise NotImplementedError(
                "`rotation` is currently limited to 1D Volume stacks."
            )

        assert isinstance(
            rot_matrices, Rotation
        ), f"Argument must be an instance of the Rotation class. {type(rot_matrices)} was supplied."

        # Get numpy representation of Rotation object.
        rot_matrices = rot_matrices.matrices

        K = len(rot_matrices)  # Rotation stack size
        assert K == self.n_vols or K == 1, "Rotation object must be length 1 or n_vols."

        if rot_matrices.dtype != self.dtype:
            logger.warning(
                f"{self.__class__.__name__}"
                f" rot_matrices.dtype {rot_matrices.dtype}"
                f" != self.dtype {self.dtype}."
                " In the future this will raise an error."
            )

        # If K = 1 we broadcast the single Rotation object across each volume.
        if K == 1:
            pts_rot = rotated_grids_3d(self.resolution, rot_matrices)
            vol_f = nufft(self.asnumpy(), pts_rot)
            vol_f = vol_f.reshape(-1, self.resolution, self.resolution, self.resolution)

        # If K = n_vols, we apply the ith rotation to ith volume.
        else:
            rot_matrices = rot_matrices.reshape((K, 1, 3, 3))
            pts_rot = np.zeros((K, 3, self.resolution**3), dtype=self.dtype)
            vol_f = np.empty(
                (self.n_vols, self.resolution**3), dtype=complex_type(self.dtype)
            )
            for i in range(K):
                pts_rot[i] = rotated_grids_3d(self.resolution, rot_matrices[i])

                vol_f[i] = nufft(self[i], pts_rot[i])

            vol_f = vol_f.reshape(-1, self.resolution, self.resolution, self.resolution)

        # If resolution is even, we zero out the nyquist frequency by default.
        if self.resolution % 2 == 0 and zero_nyquist is True:
            vol_f[:, 0, :, :] = 0
            vol_f[:, :, 0, :] = 0
            vol_f[:, :, :, 0] = 0

        vol = xp.asnumpy(
            np.real(fft.centered_ifftn(xp.asarray(vol_f), axes=(-3, -2, -1)))
        )

        return Volume(vol)

    def denoise(self):
        raise NotImplementedError

    def save(self, filename, overwrite=False):
        """
        Save volume to disk as mrc file

        :param filename: Filepath where volume will be saved

        :param overwrite: Option to overwrite file when set to True.
            Defaults to overwrite=False.
        """
        if self.stack_ndim > 1:
            raise NotImplementedError(
                "`save` is currently limited to 1D Volume stacks."
            )

        with mrcfile.new(filename, overwrite=overwrite) as mrc:
            mrc.set_data(self._data.astype(np.float32))

        if self.dtype != np.float32:
            logger.info(f"Volume with dtype {self.dtype} saved with dtype float32")

    @staticmethod
    def load(filename, permissive=True, dtype=np.float32):
        """
        Load an mrc file as a Volume instance.

        :param filename: Data filepath to load.
        :param permissive: Allows problematic files to load with warning when True.
            Defaults to permissive=True.
        :param dtype: Optionally specifiy data type. Defaults to dtype=np.float32.

        :return: Volume instance.
        """
        with mrcfile.open(filename, permissive=permissive) as mrc:
            loaded_data = mrc.data
        if loaded_data.dtype != dtype:
            logger.info(f"{filename} with dtype {loaded_data.dtype} loaded as {dtype}")
        return Volume(loaded_data.astype(dtype))


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

    grid2d = grid_2d(L, indexing="xy", dtype=rot_matrices.dtype)
    num_pts = L**2
    num_rots = rot_matrices.shape[0]
    pts = np.pi * np.vstack(
        [
            grid2d["x"].flatten(),
            grid2d["y"].flatten(),
            np.zeros(num_pts, dtype=rot_matrices.dtype),
        ]
    )
    pts_rot = np.zeros((3, num_rots, num_pts), dtype=rot_matrices.dtype)
    for i in range(num_rots):
        pts_rot[:, i, :] = rot_matrices[i, :, :] @ pts

    pts_rot = pts_rot.reshape((3, num_rots, L, L))

    return pts_rot


def rotated_grids_3d(L, rot_matrices):
    """
    Generate rotated Fourier grids in 3D from rotation matrices.

    :param L: The resolution of the desired grids.
    :param rot_matrices: An array of size k-by-3-by-3 containing K rotation matrices
    :return: A set of rotated Fourier grids in three dimensions as specified by the rotation matrices.
        Frequencies are in the range [-pi, pi].
    """

    grid3d = grid_3d(L, indexing="xyz", dtype=rot_matrices.dtype)
    num_pts = L**3
    num_rots = rot_matrices.shape[0]
    pts = np.pi * np.vstack(
        [
            grid3d["x"].flatten(),
            grid3d["y"].flatten(),
            grid3d["z"].flatten(),
        ]
    )
    pts_rot = np.zeros((3, num_rots, num_pts), dtype=rot_matrices.dtype)
    for i in range(num_rots):
        pts_rot[:, i, :] = rot_matrices[i, :, :] @ pts

    # Note we return grids as (Z,Y,X)
    return pts_rot.reshape(3, -1)
