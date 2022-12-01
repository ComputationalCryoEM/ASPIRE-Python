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

    def __init__(self, L, C, K=16, alpha=1, seed=None, dtype=np.float64):
        """
        :param L: Resolution of the Volume(s) in pixels.
        :param C: Number of Volumes to generate.
        :param K: Number of Gaussian blobs used to construct the Volume(s).
        :param alpha: Scaling factor for variance of Gaussian blobs. Default=1.
        :param seed: Random seed for generating random Gaussian blobs.
        :param dtype: dtype for Volume(s)
        """
        self.K = int(K)
        self.alpha = float(alpha)
        super().__init__(L=L, C=C, seed=seed, dtype=dtype)

    @abc.abstractproperty
    def n_blobs(self):
        """
        The total number of Gaussian blobs used to generate a Volume.
        This value differs from `self.K` as it accounts for the blobs
        which have been duplicated during `_symmetrize_gaussians`.
        """

    @abc.abstractproperty
    def symmetry_group(self):
        """
        This property will be implemented by each subclass.
        """

    def generate(self):
        """
        Generates a Volume object with specified symmetry that is multiplied by a bump function
        to give compact support within the unit sphere.
        """
        vol = self._gaussian_blob_vols()

        bump_mask = bump_3d(self.L, spread=5, dtype=self.dtype)
        vol = np.multiply(bump_mask, vol)

        return Volume(vol)

    def _gaussian_blob_vols(self):
        """
        Generates a Volume object composed of Gaussian blobs.

        :return: A Volume instance containing C Gaussian blob volumes.
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
    A Volume object with cyclically symmetric volumes constructed of random 3D Gaussian blobs.
    """

    def __init__(self, L, C, order, K=16, alpha=1, seed=None, dtype=np.float64):
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
        super().__init__(L=L, C=C, K=K, alpha=alpha, seed=seed, dtype=dtype)

    def _check_order(self):
        if self.order < 2:
            raise ValueError(
                f"For a {self.__class__.__name__} the cyclic order must be greater than 1. Provided order was {self.order}"
            )

    @property
    def n_blobs(self):
        return self.order * self.K

    @property
    def symmetry_group(self):
        """
        The Cn symmetry group contains all rotations about the z-axis
        by multiples of 2pi/n.

        In the case of an AsymmetricVolume or LegacyVolume, `symmetry_group`
        contains only the identity.

        :return: Rotation object containing the Cn symmetry group and the identity.
        """
        angles = np.zeros((self.order, 3), dtype=self.dtype)
        angles[:, 2] = 2 * np.pi * np.arange(self.order) / self.order
        return Rotation.from_euler(angles, dtype=self.dtype)


class DnSymmetricVolume(CnSymmetricVolume):
    """
    A Volume object with n-fold dihedral symmetry constructed of random 3D Gaussian blobs.
    """

    @property
    def n_blobs(self):
        return 2 * self.order * self.K

    @property
    def symmetry_group(self):
        """
        The Dn symmetry group contains all elements of the Cn symmetry group.
        In addition, for each element of the Cn symmetric group we rotate by
        pi about a perpendicular axis, in this case the y-axis.

        :return: Rotation object containing the Dn symmetry group and the identity.
        """
        # Rotations to induce cyclic symmetry
        angles = 2 * np.pi * np.arange(self.order, dtype=self.dtype) / self.order
        rot_z = Rotation.about_axis("z", angles).matrices

        # Perpendicular rotation to induce dihedral symmetry
        rot_perp = Rotation.about_axis("y", np.pi, dtype=self.dtype).matrices

        # Full set of rotations.
        rots = np.concatenate((rot_z, rot_z @ rot_perp[0].T), dtype=self.dtype)

        return Rotation(rots)


class TSymmetricVolume(GaussianBlobsVolume):
    """
    A Volume object with tetrahedral symmetry constructed of random 3D Gaussian blobs.
    """

    @property
    def n_blobs(self):
        return 12 * self.K

    @property
    def symmetry_group(self):
        """
        A tetrahedron has C3 symmetry along the 4 axes through each vertex and
        perpendicular to the opposite face, and C2 symmetry along the axes through
        the midpoints of opposite edges. We convert from axis-angle representation of
        the symmetry group elements into rotation vectors to generate the rotation
        matrices via the `from_rotvec()` method.

        :return: Rotation object containing the tetrahedral symmetry group and the identity.
        """
        # C3 rotation vectors, ie. angle * axis.
        axes_C3 = np.array(
            [[1, 1, 1], [-1, -1, 1], [1, -1, -1], [-1, 1, -1]], dtype=self.dtype
        )
        axes_C3 /= np.linalg.norm(axes_C3, axis=-1)[..., np.newaxis]
        angles_C3 = np.array([2 * np.pi / 3, 4 * np.pi / 3], dtype=self.dtype)
        rot_vecs_C3 = np.array(
            [angle * axes_C3 for angle in angles_C3], dtype=self.dtype
        ).reshape((8, 3))

        # C2 rotation vectors.
        axes_C2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=self.dtype)
        rot_vecs_C2 = np.pi * axes_C2

        # The full set of rotation vectors inducing tetrahedral symmetry.
        rot_vec_I = np.zeros((1, 3), dtype=self.dtype)
        rot_vecs = np.concatenate(
            (rot_vec_I, rot_vecs_C3, rot_vecs_C2), dtype=self.dtype
        )

        # Return rotations.
        return Rotation.from_rotvec(rot_vecs, dtype=self.dtype)


class OSymmetricVolume(GaussianBlobsVolume):
    """
    A Volume object with octahedral symmetry constructed of random 3D Gaussian blobs.
    """

    @property
    def n_blobs(self):
        return 24 * self.K

    @property
    def symmetry_group(self):
        """
        The symmetry group elements of the octahedral symmetry group O are the identity,
        the elements of 3 C4 rotation groups whose axes pass through two opposite vertices of the
        regular octahedron, 4 C3 rotation groups whose axes pass through the midpoints of two of
        its opposite faces, and 6 C2 rotation groups whose axes pass through the midpoints of two of
        its opposite edges.

        :return: Rotation object containing the octahedral symmetry group and the identity.
        """

        # C4 rotation vectors, ie angle * axis
        axes_C4 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=self.dtype)
        angles_C4 = np.array([np.pi / 2, np.pi, 3 * np.pi / 2], dtype=self.dtype)
        rot_vecs_C4 = np.array(
            [angle * axes_C4 for angle in angles_C4], dtype=self.dtype
        ).reshape((9, 3))

        # C3 rotation vectors
        axes_C3 = np.array(
            [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=self.dtype
        )
        axes_C3 /= np.linalg.norm(axes_C3, axis=-1)[..., np.newaxis]
        angles_C3 = np.array([2 * np.pi / 3, 4 * np.pi / 3], dtype=self.dtype)
        rot_vecs_C3 = np.array(
            [angle * axes_C3 for angle in angles_C3], dtype=self.dtype
        ).reshape((8, 3))

        # C2 rotation vectors
        axes_C2 = np.array(
            [[1, 1, 0], [-1, 1, 0], [1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1]],
            dtype=self.dtype,
        )
        axes_C2 /= np.linalg.norm(axes_C2, axis=-1)[..., np.newaxis]
        rot_vecs_C2 = np.pi * axes_C2

        # The full set of rotation vectors inducing octahedral symmetry.
        rot_vec_I = np.zeros((1, 3), dtype=self.dtype)
        rot_vecs = np.concatenate(
            (rot_vec_I, rot_vecs_C4, rot_vecs_C3, rot_vecs_C2), dtype=self.dtype
        )

        # Return rotations.
        return Rotation.from_rotvec(rot_vecs, dtype=self.dtype)


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
    """
    An asymmetric Volume object used for testing of legacy code.
    """

    def generate(self):
        """
        Generates an asymmetric volume composed of random 3D Gaussian blobs.
        """
        return self._gaussian_blob_vols()
