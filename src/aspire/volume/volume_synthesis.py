import abc

import numpy as np
from numpy.linalg import qr

from aspire.utils import Rotation, bump_3d, grid_3d
from aspire.utils.random import Random, randn
from aspire.volume import Volume


class SyntheticVolumeBase(abc.ABC):
    def __init__(self, L, C, seed=None, dtype=np.float64):
        self.L = L
        self.C = C
        self.seed = seed
        self.dtype = dtype

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

    def __init__(self, L, C, order, K=16, seed=None, dtype=np.float64):
        """
        :param L: Resolution of the Volume(s) in pixels.
        :param C: Number of Volumes to generate.
        :param order: An integer representing the cyclic order of the Volume(s).
        :param K: Number of Gaussian blobs used to construct the Volume(s).
        :param seed: Random seed for generating random Gaussian blobs.
        :param dtype: dtype for Volume(s)
        """
        self.order = int(order)
        self._check_order()
        self.K = int(K)
        super().__init__(L=L, C=C, seed=seed, dtype=dtype)

    def _check_order(self):
        if self.order < 2:
            raise ValueError(
                f"For a {self.__class__.__name__} the cyclic order must be greater than 1. Provided order was {self.order}"
            )

    @abc.abstractmethod
    def _symmetrize_gaussians(self):
        """
        Called to add symmetry to Volumes by generating for each Gaussian blob duplicates in symmetric positions.
        """

    def generate(self):
        """
        Generates a Volume object with specified symmetry that is multiplied by a bump function
        to give compact support within the unit sphere.
        """
        vol = self._gaussian_blob_Cn_vols()

        bump_mask = bump_3d(self.L, spread=5, dtype=self.dtype)
        vol = np.multiply(bump_mask, vol)

        return Volume(vol)

    def _gaussian_blob_Cn_vols(self):
        """
        Generate Cn rotationally symmetric volumes composed of Gaussian blobs.
        The volumes are symmetric about the z-axis.

        Defaults to volumes with no symmetry.

        :return: A Volume instance containing C Gaussian blob volumes with Cn symmetry.
        """
        vols = np.zeros(shape=((self.C,) + (self.L,) * 3)).astype(self.dtype)
        with Random(self.seed):
            for c in range(self.C):
                Q, D, mu = self._gen_gaussians()
                Q_rot, D_sym, mu_rot = self._symmetrize_gaussians(Q, D, mu)
                vols[c] = self._eval_gaussians(Q_rot, D_sym, mu_rot)
        return Volume(vols)

    def _gen_gaussians(self):
        """
        For K gaussians, generate random orientation (Q), mean (mu), and variance (D).

        :return: Orientations Q, Variances D, Means mu.
        """
        alpha = 1
        Q = np.zeros(shape=(self.K, 3, 3)).astype(self.dtype)
        D = np.zeros(shape=(self.K, 3, 3)).astype(self.dtype)
        mu = np.zeros(shape=(self.K, 3)).astype(self.dtype)

        for k in range(self.K):
            V = randn(3, 3).astype(self.dtype) / np.sqrt(3)
            Q[k, :, :] = qr(V)[0]
            D[k, :, :] = alpha**2 / 16 * np.diag(np.sum(abs(V) ** 2, axis=0))
            mu[k, :] = 0.5 * randn(3) / np.sqrt(3)

        return Q, D, mu

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
        g = grid_3d(self.L, indexing="xyz", dtype=self.dtype)
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
    A cyclically symmetric volume constructed of random 3D Gaussian blobs.
    """

    def _symmetrize_gaussians(self, Q, D, mu):
        angles = np.zeros(shape=(self.order, 3))
        angles[:, 2] = 2 * np.pi * np.arange(self.order) / self.order
        rot = Rotation.from_euler(angles).matrices

        Q_rot = np.zeros(shape=(self.order * self.K, 3, 3)).astype(self.dtype)
        D_sym = np.zeros(shape=(self.order * self.K, 3, 3)).astype(self.dtype)
        mu_rot = np.zeros(shape=(self.order * self.K, 3)).astype(self.dtype)
        idx = 0

        for j in range(self.order):
            for k in range(self.K):
                Q_rot[idx] = rot[j].T @ Q[k]
                D_sym[idx] = D[k]
                mu_rot[idx] = rot[j].T @ mu[k]
                idx += 1
        return Q_rot, D_sym, mu_rot


class AsymmetricVolume(CnSymmetricVolume):
    """
    An asymmetric Volume constructed of random 3D Gaussian blobs with compact support in the unit sphere.
    """

    def __init__(self, L, C, K=16, seed=None, dtype=np.float64):
        super().__init__(L=L, C=C, order=1, seed=seed, dtype=dtype)

    def _check_order(self):
        if self.order != 1:
            raise ValueError(
                f"An {self.__class__.__name__} must have order=1. Provided order was {self.order}"
            )

    def _symmetrize_gaussians(self, Q, D, mu):
        return Q, D, mu


class LegacyVolume(AsymmetricVolume):
    def generate(self):
        """
        Generates a SyntheticVolume with specified symmetry that is multiplied by a bump function
        to give compact support within the unit sphere.
        """
        return self._gaussian_blob_Cn_vols()
