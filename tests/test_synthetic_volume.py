from unittest import TestCase

import numpy as np
from parameterized import parameterized_class

from aspire.source.simulation import Simulation
from aspire.utils import Rotation, grid_3d
from aspire.volume import (
    AsymmetricVolume,
    CnSymmetricVolume,
    DnSymmetricVolume,
    LegacyVolume,
    OSymmetricVolume,
    TSymmetricVolume,
)


class Base:
    def setUp(self):
        self.dtype = np.float32
        self.C = 1
        self.seed = 0
        vol_kwargs = dict(
            L=self.L,
            C=self.C,
            seed=self.seed,
            dtype=self.dtype,
        )
        if hasattr(self, "order"):
            vol_kwargs["order"] = self.order

        self.vol_obj = self.vol_class(**vol_kwargs)
        self.vol = self.vol_obj.generate()

    def testVolumeRepr(self):
        """Test Synthetic Volume repr"""
        self.assertTrue(
            repr(self.vol_obj).startswith(f"{self.vol_obj.__class__.__name__}")
        )

    def testVolumeGenerate(self):
        """Test that a volume is generated"""
        _ = self.vol

    def testSimulationInit(self):
        """Test that a Simulation initializes provided a synthetic Volume."""
        _ = Simulation(L=self.L, vols=self.vol)

    def testCompactSupport(self):
        """Test that volumes have compact support."""
        if self.vol_class != LegacyVolume:
            # Mask to check support
            g_3d = grid_3d(self.L, dtype=self.dtype)
            inside = g_3d["r"] < (self.L - 1) / self.L
            outside = g_3d["r"] > 1

            # Check that volume is zero outside of support and positive inside.
            self.assertTrue(self.vol[0][outside].all() == 0)
            self.assertTrue((self.vol[0][inside] > 0).all())

    def testVolumeSymmetry(self):
        """Test that volumes have intended symmetry."""
        vol = self.vol

        # Rotations in symmetry group, excluding the Identity.
        rots = self.vol_obj.symmetry_group.matrices[1:]

        for rot in rots:
            # Rotate volume by an element of the symmetric group.
            rot_vol = vol.rotate(Rotation(rot), zero_nyquist=False)

            # Check that correlation is close to 1.
            corr = np.dot(rot_vol[0].flatten(), vol[0].flatten()) / np.dot(
                vol[0].flatten(), vol[0].flatten()
            )
            self.assertTrue(abs(corr - 1) < 1e-5)


@parameterized_class(
    ("L", "order"),
    [
        (20, 2),
        (21, 2),
        (30, 3),
        (31, 3),
        (40, 4),
        (41, 4),
        (52, 5),
        (53, 5),
        (64, 6),
        (65, 6),
    ],
)
class CnSymmetricVolumeCase(Base, TestCase):
    vol_class = CnSymmetricVolume
    L = 20
    order = 2


@parameterized_class(
    ("L", "order"),
    [
        (20, 2),
        (21, 2),
        (40, 3),
        (41, 3),
        (42, 4),
        (43, 4),
        (55, 5),
        (56, 5),
        (64, 6),
        (65, 6),
    ],
)
class DnSymmetricVolumeCase(Base, TestCase):
    vol_class = DnSymmetricVolume
    L = 20
    order = 2


@parameterized_class(("L"), [(20,), (21,)])
class TSymmetricVolumeCase(Base, TestCase):
    vol_class = TSymmetricVolume
    L = 20


@parameterized_class(("L"), [(20,), (21,)])
class OSymmetricVolumeCase(Base, TestCase):
    vol_class = OSymmetricVolume
    L = 20


@parameterized_class(("L"), [(21,)])
class AsymmetricVolumeCase(Base, TestCase):
    vol_class = AsymmetricVolume
    L = 20


@parameterized_class(("L"), [(21,)])
class LegacyVolumeCase(Base, TestCase):
    vol_class = LegacyVolume
    L = 20
