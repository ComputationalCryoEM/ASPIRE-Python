import logging

import mrcfile
import numpy as np
from numpy.linalg import qr

import aspire.image
from aspire.nufft import nufft
from aspire.numeric import fft, xp
from aspire.utils import Rotation, mat_to_vec, vec_to_mat
from aspire.utils.coor_trans import grid_2d, grid_3d
from aspire.utils.matlab_compat import m_reshape
from aspire.utils.random import Random, randn
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
    Volume is an N x L x L x L array, along with associated utility methods.
    """

    def __init__(self, data):
        """
        Create a volume initialized with `data`.

        Volumes should be N x L x L x L,
        or L x L x L which implies N=1.

        :param data: Volume data

        :return: A volume instance.
        """

        if data.ndim == 3:
            data = data[np.newaxis, :, :, :]

        assert data.ndim == 4, (
            "Volume data should be ndarray with shape NxLxLxL" " or LxLxL."
        )

        assert (
            data.shape[1] == data.shape[2] == data.shape[3]
        ), "Only cubed ndarrays are supported."

        self._data = data
        self.n_vols = self._data.shape[0]
        self.dtype = self._data.dtype
        self.resolution = self._data.shape[1]
        self.shape = self._data.shape
        self.volume_shape = self._data.shape[1:]

    def asnumpy(self):
        """
        Return volume as a (n_vols, resolution, resolution, resolution) array.

        :return: ndarray
        """
        return self._data

    def __getitem__(self, item):
        # this is one reason why you might want Volume and VolumeStack classes...
        # return Volume(self._data[item])
        return self._data[item]

    def __setitem__(self, key, value):
        self._data[key] = value

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
        Returns a new Volume instance with volume data axis tranposed

        :return: Volume instance.
        """

        vol_t = np.empty_like(self._data)
        for n, v in enumerate(self._data):
            vol_t[n] = v.T

        return Volume(vol_t)

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

    def flip(self, axis=0):
        """
        Flip volume stack data along axis using numpy.flip

        :param axis: Optionally specify axis as integer or tuple.
        Defaults to axis=0.

        :return: Volume instance.
        """

        return Volume(np.flip(self._data, axis))

    def downsample(self, szout, mask=None):
        if isinstance(szout, int):
            szout = (szout,) * 3

        return Volume(aspire.image.downsample(self._data, szout, mask))

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
            pts_rot = np.zeros((K, 3, self.resolution**3))
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


def gaussian_blob_vols(L=8, C=2, K=16, symmetry_type=None, seed=None, dtype=np.float64):
    """
    Builds gaussian blob volumes with chosen symmetry type.

    :param L: The resolution of the volume.
    :param C: Number of volumes.
    :param K: The number of gaussian blobs used to generate the volume.
    :param symmetry_type: A string indicating the type of symmetry.
    :param seed: The random seed to produce centers and variances of the gaussian blobs.
    :param dtype: Data type.

    :return: A volume instance from an appropriate volume generator.
    """

    order = 1
    sym_type = None
    if symmetry_type is not None:
        # safer to make string consistent
        symmetry_type = symmetry_type.upper()
        # get the first letter
        sym_type = symmetry_type[0]
        # if there is a number denoting rotational symmetry, get that
        order = symmetry_type[1:] or None

    # map our sym_types to classes of Volumes
    map_sym_to_generator = {
        None: _gaussian_blob_Cn_vols,
        "C": _gaussian_blob_Cn_vols,
        # "D": gaussian_blob_Dn_vols,
        # "T": gaussian_blob_T_vols,
        # "O": gaussian_blob_O_vols,
    }

    sym_types = list(map_sym_to_generator.keys())
    if sym_type not in map_sym_to_generator.keys():
        raise NotImplementedError(
            f"{sym_type} type symmetry is not supported. The following symmetry types are currently supported: {sym_types}."
        )

    try:
        order = int(order)
    except Exception:
        raise NotImplementedError(
            f"{sym_type}{order} symmetry not supported. Only {sym_type}n symmetry, where n is an integer, is supported."
        )

    vols_generator = map_sym_to_generator[sym_type]

    return vols_generator(L=L, C=C, K=K, order=order, seed=seed, dtype=dtype)


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


def _gaussian_blob_Cn_vols(
    L=8, C=2, K=16, alpha=1, order=1, seed=None, dtype=np.float64
):
    """
    Generate Cn rotationally symmetric volumes composed of Gaussian blobs.
    The volumes are symmetric about the z-axis.

    Defaults to volumes with no symmetry.

    :param L: The size of the volumes
    :param C: The number of volumes to generate
    :param K: The number of blobs each volume is composed of.
    A Cn symmetric volume will be composed of n times K blobs.
    :param order: The order of cyclic symmetry.
    :param alpha: A scale factor of the blob widths

    :return: A Volume instance containing C Gaussian blob volumes with Cn symmetry.
    """

    # Apply symmetry to Q and mu by generating duplicates rotated by symmetry order.
    def _symmetrize_gaussians(Q, D, mu, order):
        angles = np.zeros(shape=(order, 3))
        angles[:, 2] = 2 * np.pi * np.arange(order) / order
        rot = Rotation.from_euler(angles).matrices

        K = Q.shape[0]
        Q_rot = np.zeros(shape=(order * K, 3, 3)).astype(dtype)
        D_sym = np.zeros(shape=(order * K, 3, 3)).astype(dtype)
        mu_rot = np.zeros(shape=(order * K, 3)).astype(dtype)
        idx = 0

        for j in range(order):
            for k in range(K):
                Q_rot[idx] = rot[j].T @ Q[k]
                D_sym[idx] = D[k]
                mu_rot[idx] = rot[j].T @ mu[k]
                idx += 1
        return Q_rot, D_sym, mu_rot

    vols = np.zeros(shape=(C, L, L, L)).astype(dtype)
    with Random(seed):
        for c in range(C):
            Q, D, mu = _gen_gaussians(K, alpha)
            Q_rot, D_sym, mu_rot = _symmetrize_gaussians(Q, D, mu, order)
            vols[c] = _eval_gaussians(L, Q_rot, D_sym, mu_rot, dtype=dtype)
    return Volume(vols)


def _eval_gaussians(L, Q, D, mu, dtype=np.float64):
    """
    Evaluate Gaussian blobs over a 3D grid with centers, mu, orientations, Q, and variances, D.

    :param L: Size of the volume to be populated with Gaussian blobs.
    :param Q: A stack of size (n_blobs) x 3 x 3 of rotation matrices,
        determining the orientation of each blob.
    :param D: A stack of size (n_blobs) x 3 x 3 diagonal matrices,
        whose diagonal entries are the variances of each blob.
    :param mu: An array of size (n_blobs) x 3 containing the centers for each blob.

    :return: An L x L x L array.
    """
    g = grid_3d(L, indexing="xyz", dtype=dtype)
    coords = np.array(
        [g["x"].flatten(), g["y"].flatten(), g["z"].flatten()], dtype=dtype
    )

    n_blobs = Q.shape[0]
    vol = np.zeros(shape=(1, coords.shape[-1])).astype(dtype)

    for k in range(n_blobs):
        coords_k = coords - mu[k, :, np.newaxis]
        coords_k = Q[k].T @ coords_k * np.sqrt(1 / np.diag(D[k, :, :]))[:, np.newaxis]

        vol += np.exp(-0.5 * np.sum(np.abs(coords_k) ** 2, axis=0))

    vol = np.reshape(vol, g["x"].shape)

    return vol


def _gen_gaussians(K, alpha, dtype=np.float64):
    """
    For K gaussians, generate random orientation (Q), mean (mu), and variance (D).

    :param K: Number of gaussians to generate.
    :param alpha: Scalar for peak of gaussians.

    :return: Orientations Q, Variances D, Means mu.
    """
    Q = np.zeros(shape=(K, 3, 3)).astype(dtype)
    D = np.zeros(shape=(K, 3, 3)).astype(dtype)
    mu = np.zeros(shape=(K, 3)).astype(dtype)

    for k in range(K):
        V = randn(3, 3).astype(dtype) / np.sqrt(3)
        Q[k, :, :] = qr(V)[0]
        D[k, :, :] = alpha**2 / 16 * np.diag(np.sum(abs(V) ** 2, axis=0))
        mu[k, :] = 0.5 * randn(3) / np.sqrt(3)

    return Q, D, mu


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
