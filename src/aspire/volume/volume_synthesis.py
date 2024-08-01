import abc

import numpy as np
from numpy.linalg import qr

from aspire.utils import bump_3d, grid_3d
from aspire.utils.random import Random, randn
from aspire.volume import (
    CnSymmetryGroup,
    DnSymmetryGroup,
    IdentitySymmetryGroup,
    OSymmetryGroup,
    TSymmetryGroup,
    Volume,
)


class SyntheticVolumeBase(abc.ABC):
    def __init__(self, L, C, pixel_size=None, seed=None, dtype=np.float64):
        self.L = L
        self.C = C
        self.seed = seed
        self.dtype = dtype
        self.pixel_size = pixel_size

    @abc.abstractmethod
    def generate(self):
        """
        Called to generate and return synthetic volumes.

        Each concrete subclass should implement this.
        """

    def __repr__(self):
        return f"{self.__class__.__name__} {self.__dict__}"


class GaussianBlobsVolume(SyntheticVolumeBase):
    """
    A base class for all volumes which are generated with randomized 3D Gaussians.
    """

    def __init__(
        self, L, C, K=16, alpha=1, pixel_size=None, seed=None, dtype=np.float64
    ):
        """
        :param L: Resolution of the Volume(s) in pixels.
        :param C: Number of Volumes to generate.
        :param K: Number of Gaussian blobs used to construct the Volume(s).
        :param alpha: Scaling factor for variance of Gaussian blobs. Default=1.
        :param pixel_size: Optional voxel_size in angstroms.
            When provided will be saved with `map`/`mrc` metadata.
            Default of `None` will not write to file,
            but will be considered unit pixels (1) for FSC.
        :param seed: Random seed for generating random Gaussian blobs.
        :param dtype: dtype for Volume(s)
        """
        self.K = int(K)
        self.alpha = float(alpha)
        super().__init__(L=L, C=C, pixel_size=pixel_size, seed=seed, dtype=dtype)
        self._set_symmetry_group()

    @abc.abstractproperty
    def n_blobs(self):
        """
        The total number of Gaussian blobs used to generate a Volume.
        This value differs from `self.K` as it accounts for the blobs
        which have been duplicated during `_symmetrize_gaussians`.
        """

    @property
    def symmetry_group(self):
        """
        SymmetryGroup object corresponding to the symmetry of the Volume.
        """
        return self._symmetry_group

    def generate(self):
        """
        Generates a Volume object with specified symmetry that is multiplied by a bump function
        to give compact support within the unit sphere.
        """
        vol = self._gaussian_blob_vols()
        bump_mask = bump_3d(self.L, spread=5, dtype=self.dtype)
        return Volume(
            bump_mask * vol,
            symmetry_group=self.symmetry_group,
            pixel_size=self.pixel_size,
        )

    def _gaussian_blob_vols(self):
        """
        Generates a 4D array representing a stack of volumes composed of Gaussian blobs.

        :return: An ndarray containing C Gaussian blob volumes.
        """
        vols = np.zeros(shape=((self.C,) + (self.L,) * 3)).astype(self.dtype)
        with Random(self.seed):
            for c in range(self.C):
                Q, D, mu = self._gen_gaussians()
                Q_rot, D_sym, mu_rot = self._symmetrize_gaussians(Q, D, mu)
                vols[c] = self._eval_gaussians(Q_rot, D_sym, mu_rot)
        return vols

    def _gen_gaussians(self):
        """
        For K gaussians, generate random orientation (Q), mean (mu), and variance (D).

        :return: Orientations Q, Variances D, Means mu.
        """
        Q = np.zeros(shape=(self.K, 3, 3)).astype(self.dtype)
        D = np.zeros(shape=(self.K, 3, 3)).astype(self.dtype)
        mu = np.zeros(shape=(self.K, 3)).astype(self.dtype)

        for k in range(self.K):
            V = randn(3, 3).astype(self.dtype) / np.sqrt(3)
            Q[k, :, :] = qr(V)[0]
            D[k, :, :] = (
                self.alpha**2 / self.n_blobs * np.diag(np.sum(abs(V) ** 2, axis=0))
            )
            mu[k, :] = 0.5 * randn(3) / np.sqrt(3)

        return Q, D, mu

    def _symmetrize_gaussians(self, Q, D, mu):
        """
        Called to add symmetry to Volumes by generating for each Gaussian blob duplicates in symmetric positions.
        """
        rots = self.symmetry_group.matrices

        Q_rot = np.zeros(shape=(self.n_blobs, 3, 3)).astype(self.dtype)
        D_sym = np.zeros(shape=(self.n_blobs, 3, 3)).astype(self.dtype)
        mu_rot = np.zeros(shape=(self.n_blobs, 3)).astype(self.dtype)
        idx = 0

        for rot in rots:
            for k in range(self.K):
                Q_rot[idx] = rot.T @ Q[k]
                D_sym[idx] = D[k]
                mu_rot[idx] = rot.T @ mu[k]
                idx += 1
        return Q_rot, D_sym, mu_rot

    def _eval_gaussians(self, Q, D, mu):
        """
        Evaluate Gaussian blobs over a 3D grid with centers, mu, orientations, Q, and variances, D.

        :param Q: A stack of size (n_blobs) x 3 x 3 of rotation matrices,
            determining the orientation of each blob.
        :param D: A stack of size (n_blobs) x 3 x 3 diagonal matrices,
            whose diagonal entries are the variances of each blob.
        :param mu: An array of size (n_blobs) x 3 containing the centers for each blob.

        :return: An L x L x L array.
        """
        g = grid_3d(self.L, indexing="zyx", dtype=self.dtype)
        coords = np.array(
            [g["x"].flatten(), g["y"].flatten(), g["z"].flatten()], dtype=self.dtype
        )

        n_blobs = Q.shape[0]
        vol = np.zeros(shape=(1, coords.shape[-1])).astype(self.dtype)

        for k in range(n_blobs):
            coords_k = coords - mu[k, :, np.newaxis]
            coords_k = (
                Q[k].T @ coords_k * np.sqrt(1 / np.diag(D[k, :, :]))[:, np.newaxis]
            )

            vol += np.exp(-0.5 * np.sum(np.abs(coords_k) ** 2, axis=0))

        vol = np.reshape(vol, g["x"].shape)

        return vol


class CnSymmetricVolume(GaussianBlobsVolume):
    """
    A Volume object with cyclically symmetric volumes constructed of random 3D Gaussian blobs.
    """

    def __init__(
        self, L, C, order, K=16, alpha=1, pixel_size=None, seed=None, dtype=np.float64
    ):
        """
        :param L: Resolution of the Volume(s) in pixels.
        :param C: Number of Volumes to generate.
        :param order: An integer representing the cyclic order of the Volume(s).
        :param K: Number of Gaussian blobs used to construct the Volume(s).
        :param pixel_size: Optional voxel_size in angstroms.
            When provided will be saved with `map`/`mrc` metadata.
            Default of `None` will not write to file,
            but will be considered unit pixels (1) for FSC.
        :param seed: Random seed for generating random Gaussian blobs.
        :param dtype: dtype for Volume(s)
        """
        self.order = int(order)
        self._check_order()
        super().__init__(
            L=L, C=C, K=K, alpha=alpha, pixel_size=pixel_size, seed=seed, dtype=dtype
        )

    def _check_order(self):
        if self.order < 2:
            raise ValueError(
                f"For a {self.__class__.__name__} the cyclic order must be greater than 1. Provided order was {self.order}"
            )

    def _set_symmetry_group(self):
        self._symmetry_group = CnSymmetryGroup(order=self.order, dtype=self.dtype)

    @property
    def n_blobs(self):
        return self.order * self.K


class DnSymmetricVolume(CnSymmetricVolume):
    """
    A Volume object with n-fold dihedral symmetry constructed of random 3D Gaussian blobs.
    """

    def _set_symmetry_group(self):
        self._symmetry_group = DnSymmetryGroup(order=self.order, dtype=self.dtype)

    @property
    def n_blobs(self):
        return 2 * self.order * self.K


class TSymmetricVolume(GaussianBlobsVolume):
    """
    A Volume object with tetrahedral symmetry constructed of random 3D Gaussian blobs.
    """

    def _set_symmetry_group(self):
        self._symmetry_group = TSymmetryGroup(dtype=self.dtype)

    @property
    def n_blobs(self):
        return 12 * self.K


class OSymmetricVolume(GaussianBlobsVolume):
    """
    A Volume object with octahedral symmetry constructed of random 3D Gaussian blobs.
    """

    def _set_symmetry_group(self):
        self._symmetry_group = OSymmetryGroup(dtype=self.dtype)

    @property
    def n_blobs(self):
        return 24 * self.K


class AsymmetricVolume(CnSymmetricVolume):
    """
    An asymmetric Volume constructed of random 3D Gaussian blobs with compact support in the unit sphere.
    """

    def __init__(self, L, C, K=64, pixel_size=None, seed=None, dtype=np.float64):
        super().__init__(
            L=L, C=C, K=K, order=1, pixel_size=pixel_size, seed=seed, dtype=dtype
        )

    def _check_order(self):
        if self.order != 1:
            raise ValueError(
                f"An {self.__class__.__name__} must have order=1. Provided order was {self.order}"
            )

    def _set_symmetry_group(self):
        self._symmetry_group = IdentitySymmetryGroup(dtype=self.dtype)

    def _symmetrize_gaussians(self, Q, D, mu):
        return Q, D, mu


class LegacyVolume(AsymmetricVolume):
    """
    An asymmetric Volume object used for testing of legacy code.
    """

    def __init__(self, L, C=2, K=16, pixel_size=None, seed=0, dtype=np.float64):
        super().__init__(L=L, C=C, K=K, pixel_size=pixel_size, seed=seed, dtype=dtype)

    def generate(self):
        """
        Generates an asymmetric volume composed of random 3D Gaussian blobs.
        """
        vols = self._gaussian_blob_vols()

        # Swap axes to retain Legacy xyz-indexing.
        vols = np.swapaxes(vols, 1, 3)

        return Volume(vols, pixel_size=self.pixel_size)
