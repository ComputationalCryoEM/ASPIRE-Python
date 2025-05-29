import logging
from abc import ABC, abstractproperty

import numpy as np
from scipy.spatial.transform import Rotation as R

from aspire.utils import Rotation

logger = logging.getLogger(__name__)


class SymmetryGroup(ABC):
    """
    Base class for symmetry groups.
    """

    def __init__(self):
        """
        Abstract class for symmetry groups.
        """
        self.dtype = np.float64
        self.rotations = self.generate_rotations()

    def generate_rotations(self):
        """
        Method for generating a Rotation object for the symmetry group.
        """
        rots = R.create_group(self.to_string).as_matrix()
        return Rotation(rots.astype(self.dtype, copy=False))

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return str(self) == str(other)
        return False

    @property
    def matrices(self):
        return self.rotations.matrices

    def __repr__(self):
        return f"{self.__class__.__name__}, {self.__dict__}"

    @abstractproperty
    def to_string(self):
        """String denoting the symmetry type."""

    def __str__(self):
        return f"{self.to_string}"

    @staticmethod
    def parse(symmetry):
        """
        Takes a SymmetryGroup instance or a string, ie. 'C1', 'C7', 'D3', 'T', 'O',
        and returns a concrete SymmetryGroup object.

        :param symmetry: A string (or SymmetryGroup instance) indicating the symmetry of a molecule.
        :return: Concrete SymmetryGroup object.
        """

        if symmetry is None:
            return IdentitySymmetryGroup()

        if isinstance(symmetry, SymmetryGroup):
            return symmetry

        if not isinstance(symmetry, str):
            raise TypeError(
                f"`symmetry` must be a string or `SymmetryGroup` instance. Found {type(symmetry)}"
            )

        # Parse symmetry provided as a string.
        symmetry = symmetry.upper()
        if symmetry == "C1":
            return IdentitySymmetryGroup()

        symmetry_type = symmetry[0]
        symmetric_order = symmetry[1:]

        map_to_sym_group = {
            "C": CnSymmetryGroup,
            "D": DnSymmetryGroup,
            "T": TSymmetryGroup,
            "O": OSymmetryGroup,
            "I": ISymmetryGroup,
        }
        if symmetry_type not in map_to_sym_group.keys():
            raise ValueError(
                f"Symmetry type {symmetry_type} not supported. Try: {*map_to_sym_group.keys(), }."
            )

        symmetry_group = map_to_sym_group[symmetry_type]
        group_kwargs = dict()
        if symmetric_order:
            group_kwargs["order"] = int(symmetric_order)

        return symmetry_group(**group_kwargs)


class CnSymmetryGroup(SymmetryGroup):
    """
    Cyclic symmetry group.
    """

    def __init__(self, order):
        """
        `CnSymmetryGroup` instance that serves up a `Rotation` object
        containing rotation matrices of the symmetry group (including
        the identity) accessed via the `matrices` attribute.

        :param order: The cyclic order for the symmetry group (int).
        """

        self.order = int(order)
        super().__init__()

    @property
    def to_string(self):
        return "C" + str(self.order)

    def generate_rotations(self):
        """
        The Cn symmetry group contains all rotations about the z-axis
        by multiples of 2pi/n.

        In the case of an AsymmetricVolume or LegacyVolume, `symmetry_group`
        contains only the identity.

        :return: Rotation object containing the Cn symmetry group and the identity.
        """
        return super().generate_rotations()


class IdentitySymmetryGroup(CnSymmetryGroup):
    """
    The identity symmetry group.
    """

    def __init__(self):
        """
        `IdentitySymmetryGroup` instance that serves up a `Rotation` object
        containing the identity matrix.
        """

        super().__init__(order=1)


class DnSymmetryGroup(SymmetryGroup):
    """
    Dihedral symmetry group.
    """

    def __init__(self, order):
        """
        `DnSymmetryGroup` instance that serves up a `Rotation` object
        containing rotation matrices of the symmetry group (including
        the Identity) accessed via the `matrices` attribute. Note, this
        is the chiral dihedral symmetry group which does contain reflections.

        :param order: The cyclic order for the symmetry group (int).
        """

        self.order = int(order)
        super().__init__()

    @property
    def to_string(self):
        return "D" + str(self.order)

    def generate_rotations(self):
        """
        The Dn symmetry group contains all elements of the Cn symmetry group.
        In addition, for each element of the Cn symmetric group we rotate by
        pi about a perpendicular axis, in this case the y-axis.

        :return: Rotation object containing the Dn symmetry group and the identity.
        """
        return super().generate_rotations()


class TSymmetryGroup(SymmetryGroup):
    """
    Tetrahedral symmetry group.
    """

    def __init__(self):
        """
        `TSymmetryGroup` instance that serves up a `Rotation` object
        containing rotation matrices of the symmetry group (including the
        Identity) accessed via the `matrices` attribute. Note, this is the
        chiral tetrahedral symmetry group which does not contain reflections.
        """

        super().__init__()

    @property
    def to_string(self):
        return "T"

    def generate_rotations(self):
        """
        A tetrahedron has C3 symmetry along the 4 axes through each vertex and
        perpendicular to the opposite face, and C2 symmetry along the axes through
        the midpoints of opposite edges.

        :return: Rotation object containing the tetrahedral symmetry group and the identity.
        """
        return super().generate_rotations()


class OSymmetryGroup(SymmetryGroup):
    """
    Octahedral symmetry group.
    """

    def __init__(self):
        """
        `OSymmetryGroup` instance that serves up a `Rotation` object
        containing rotation matrices of the symmetry group (including the
        Identity) accessed via the `matrices` attribute. Note, this is the
        chiral octahedral symmetry group which does not contain reflections.
        """
        super().__init__()

        self._symmetry_group = self.generate_rotations()

    @property
    def to_string(self):
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
        return super().generate_rotations()


class ISymmetryGroup(SymmetryGroup):
    """
    Icosahedral symmetry group.
    """

    def __init__(self):
        """
        `ISymmetryGroup` instance that serves up a `Rotation` object
        containing rotation matrices of the symmetry group (including the
        Identity) accessed via the `matrices` attribute. Note, this is the
        chiral icosahedral symmetry group which does not contain reflections.
        """
        super().__init__()

        self._symmetry_group = self.generate_rotations()

    @property
    def to_string(self):
        return "I"

    def generate_rotations(self):
        """
        Icosahedral rotation group I (60 elements):
        - 1 identity rotation (order 1)
        - 24 rotations of order 5: ±72°, ±144° around 6 face-center axes
        - 20 rotations of order 3: ±120° around 10 vertex axes
        - 15 rotations of order 2: 180° around 15 edge-midpoint axes

        :return: Rotation object containing the icosahedral symmetry group and the identity.
        """
        return super().generate_rotations()
