import abc

import numpy as np

from aspire.volume import gaussian_blob_vols


class SyntheticVolumeBase(abc.ABC):
    def __init__(self, L, C, symmetry_type, seed=None, dtype=np.float64):
        self.L = L
        self.C = C
        self.symmetry_type = symmetry_type
        self.seed = seed
        self.dtype = dtype

    @abc.abstractmethod
    def generate(self):
        """
        Called to generate and return the synthetic volumes.

        Each concrete subclass should impliment this.
        """

    def __repr__(self):
        # return (f'L={self.L} C={self.C} symmetry_type={self.symmetry_type}'
        #         f' seed={self.seed} dtype={self.dtype}')
        return f"{self.__dict__}"


class LegacyGaussianBlob(SyntheticVolumeBase):
    """
    Legacy Aspire Gaussian Blobs.

    Suffers from too large point variances.
    Included for migration of legacy unit tests.
    """

    def __init__(self, L, C, symmetry_type, K=16, seed=None, dtype=np.float64):
        super().__init__(L, C, symmetry_type, seed=seed, dtype=dtype)
        self.K = K

    def generate(self):
        # transfer the legacy gaussian blobs stuff here.
        return gaussian_blob_vols(
            L=self.L,
            C=self.C,
            symmetry_type=self.symmetry_type,
            seed=self.seed,
            dtype=self.dtype,
        )


class BumpBlobs(SyntheticVolumeBase):
    """
    Similar to LegacyGaussianBlob, but used 3d Bump function.
    """

    def __init__(self, L, C, symmetry_type, K=16, seed=None, dtype=np.float64):
        super().__init__(L, C, symmetry_type, seed=seed, dtype=dtype)
        self.K = K


class CnSymmetricGaussianBlob(LegacyGaussianBlob):
    """
    Cn Symmetric ...
    """

    # Note this class can actually inherit everything from LegacyGaussianBlob for now.
    def __init__(self, L, C, symmetry_type, K=16, seed=None, dtype=np.float64):
        super().__init__(L, C, symmetry_type, K=K, seed=seed, dtype=dtype)
        assert self.symmetry_type is not None, "bleep Bloop eRoR"


class DnSymmetricGaussianBlob(SyntheticVolumeBase):
    """
    Dn Symmetric ...
    """

    def __init__(self, L, C, symmetry_type, K=16, seed=None, dtype=np.float64):
        super().__init__(L, C, symmetry_type, seed=seed, dtype=dtype)
        self.K = K


class TSymmetricGaussianBlob(SyntheticVolumeBase):
    """
    T Symmetric ...
    """

    def __init__(self, L, C, symmetry_type, K=16, seed=None, dtype=np.float64):
        super().__init__(L, C, symmetry_type, seed=seed, dtype=dtype)
        self.K = K


class OSymmetricGaussianBlob(SyntheticVolumeBase):
    """
    O Symmetric ...
    """

    def __init__(self, L, C, symmetry_type, K=16, seed=None, dtype=np.float64):
        super().__init__(L, C, symmetry_type, seed=seed, dtype=dtype)
        self.K = K


class PointBasedBlobs(SyntheticVolumeBase):
    """
    Designed to take in iterable of ((x,y,z), sigma), or generate a random one.
    """

    def __init__(self, L, C, symmetry_type, points=None, seed=None, dtype=np.float64):
        """
        ...
        `points` can an integer to induce generation of that number random points,
        or points can be a iterable of form described above.

        """
        super().__init__(L, C, symmetry_type, seed=seed, dtype=dtype)

        if points is None:
            points = 16

        if isinstance(points, int):
            points = self._gen_points(points)

        self.points = points
        self.K = len(self.points)


class PDBBlobs(PointBasedBlobs):
    """
    Take in a pdb-ish file and translate into points to be used
    in PointBasedBlobs.
    """

    def __init__(self, L, C, symmetry_type, filename, seed=None, dtype=np.float64):
        points = self._load_pdb(filename)
        super().__init__(L, C, symmetry_type, points=points, seed=seed, dtype=dtype)

    def _load_pdb(self, filename):
        """
        Return points loaded from file.
        """
        pass


# # example fail
# A = BumpBlobs()
