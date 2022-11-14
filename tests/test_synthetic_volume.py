from unittest import TestCase

import numpy as np
from parameterized import parameterized_class

from aspire.source.simulation import Simulation
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

        self.vol = self.vol_class(**vol_kwargs)

    def testVolumeRepr(self):
        """Test Synthetic Volume repr"""
        self.assertTrue(repr(self.vol).startswith(f"{self.vol.__class__.__name__}"))

    def testVolumeGenerate(self):
        """Test that a volume is generated"""
        _ = self.vol.generate()

    def testSimulationInit(self):
        """Test that a Simulation initializes provided a synthetic Volume."""
        _ = Simulation(L=self.L, vols=self.vol.generate())


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
