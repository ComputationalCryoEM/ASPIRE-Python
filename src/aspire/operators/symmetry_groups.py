import logging
from abc import ABC, abstractmethod, abstractproperty

import numpy as np

from aspire.utils import Rotation

logger = logging.getLogger(__name__)


class SymmetryGroup(ABC):
    """
    Base class for symmetry groups.
    """

    def __init__(self, dtype=None):
        """
        :param dtype: Numpy dtype to be used for rotation matrices.
        """

        if dtype is None:
            raise RuntimeError("You must supply a dtype for rotations.")
        else:
            self.dtype = np.dtype(dtype)

        self._rotations = None
        self.rotations = self.generate_rotations()

    @abstractmethod
    def generate_rotations(self):
        """
        Method for generating a Rotation object for the symmetry group.
        """

    @property
    def rotations(self):
        return self._rotations

    @rotations.setter
    def rotations(self, value):
        if self._rotations is not None:
            raise RuntimeError("Rotations are read-only.")
        self._rotations = value

    @property
    def matrices(self):
        return self.rotations.matrices

    def __repr__(self):
        return f"{self.__class__.__name__}, {self.__dict__}"

    @abstractproperty
    def symmetry_type(self):
        """String denoting the symmetry type."""

    def __str__(self):
        return f"{self.symmetry_type}"


class CyclicSymmetryGroup(SymmetryGroup):
    """
    Cyclic Symmetry Group.
    """

    def __init__(self, order, dtype):
        self.order = int(order)
        super().__init__(dtype=dtype)

    @property
    def symmetry_type(self):
        return "C" + str(self.order)

    def generate_rotations(self):
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


class DihedralSymmetryGroup(SymmetryGroup):
    """
    Dihedral Symmetry Group.
    """

    def __init__(self, order, dtype):
        self.order = int(order)
        super().__init__(dtype=dtype)

    @property
    def symmetry_type(self):
        return "D" + str(self.order)

    def generate_rotations(self):
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


class TetrahedralSymmetryGroup(SymmetryGroup):
    """
    Tetrahedral Symmetry Group.
    """

    def __init__(self, dtype):
        super().__init__(dtype=dtype)

    @property
    def symmetry_type(self):
        return "T"

    def generate_rotations(self):
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


class OctahedralSymmetryGroup(SymmetryGroup):
    """
    Octahedral Symmetry Group.
    """

    def __init__(self, dtype):
        super().__init__(dtype=dtype)

        self._symmetry_group = self.generate_rotations()

    @property
    def symmetry_type(self):
        return "O"

    def generate_rotations(self):
        """
        The symmetry group elements of the octahedral symmetry group O are the identity,
        the elements of 3 C4 rotation groups whose axes pass through two opposite vertices of the
        regular octahedron, 4 C3 rotation groups whose axes pass through the midpoints of two of
        its opposite faces, and 6 C2 rotation groups whose axes pass through the midpoints of two
        of its opposite edges.

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
