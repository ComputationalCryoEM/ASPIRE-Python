from unittest import TestCase

import numpy as np
from parameterized import parameterized_class

from aspire.source.simulation import Simulation
from aspire.utils import grid_3d
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
        self.L = 10
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
            inside = g_3d["r"] < 1
            outside = g_3d["r"] > 1

            # Check that volume is zero outside of support and positive inside.
            self.assertTrue(self.vol[0][outside].all() == 0)
            self.assertTrue((self.vol[0][inside] > 0).all())


@parameterized_class(("order"), [(3,), (4,), (5,), (6,)])
class CnSymmetricVolumeCase(Base, TestCase):
    vol_class = CnSymmetricVolume
    order = 2


@parameterized_class(("order"), [(3,), (4,), (5,), (6,)])
class DnSymmetricVolumeCase(Base, TestCase):
    vol_class = DnSymmetricVolume
    order = 2


class TSymmetricVolumeCase(Base, TestCase):
    vol_class = TSymmetricVolume


class OSymmetricVolumeCase(Base, TestCase):
    vol_class = OSymmetricVolume


class AsymmetricVolumeCase(Base, TestCase):
    vol_class = AsymmetricVolume


class LegacyVolumeCase(Base, TestCase):
    vol_class = LegacyVolume
