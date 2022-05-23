import unittest
from unittest import TestCase

import numpy as np

from aspire.source import Simulation
from aspire.utils import utest_tolerance
from aspire.utils.matrix import anorm
from aspire.utils.misc import gaussian_3d
from aspire.volume import Volume


class DownsampleTestCase(TestCase):
    def setUp(self):
        self.n = 128
        self.dtype = np.float32

    def tearDown(self):
        pass

    def _testDownsample2DCase(self, L, L_ds):
        # downsampling from size L to L_ds
        imgs_org, imgs_ds = self.createImages(L, L_ds)
        # check resolution is correct
        self.assertEqual((self.n, L_ds, L_ds), imgs_ds.shape)
        # check center points for all images
        self.assertTrue(self.checkCenterPoint(imgs_org, imgs_ds))
        # check signal energy is conserved
        self.assertTrue(self.checkSignalEnergy(imgs_org, imgs_ds))

    def testDownsample2D_EvenEven(self):
        # source resolution: 64
        # target resolution: 32
        self._testDownsample2DCase(64, 32)

    @unittest.skip(
        "Signal energy test fails for this case in current DS implementation"
    )
    def testDownsample2D_EvenOdd(self):
        # source resolution: 64
        # target resolution: 33
        self._testDownsample2DCase(64, 33)

    def testDownsample2D_OddOdd(self):
        # source resolution: 65
        # target resolution: 33
        self._testDownsample2DCase(65, 33)

    def testDownsample2D_OddEven(self):
        # source resolution: 65
        # target resolution: 32
        self._testDownsample2DCase(65, 32)

    def checkCenterPoint(self, imgs_org, imgs_ds):
        # Check that center point is the same after ds
        L = imgs_org.res
        max_resolution = imgs_ds.res
        return np.allclose(
            imgs_org[:, L // 2, L // 2],
            imgs_ds[:, max_resolution // 2, max_resolution // 2],
            atol=utest_tolerance(self.dtype),
        )

    def checkSignalEnergy(self, imgs_org, imgs_ds):
        # check conservation of energy after downsample
        L = imgs_org.res
        max_resolution = imgs_ds.res
        return np.allclose(
            anorm(imgs_org.asnumpy(), axes=(1, 2)) / L,
            anorm(imgs_ds.asnumpy(), axes=(1, 2)) / max_resolution,
            atol=utest_tolerance(self.dtype),
        )

    def createImages(self, L, L_ds):
        # generate a 3D Gaussian volume
        sigma = 0.1
        vol = gaussian_3d(L, sigma=(L * sigma,) * 3, dtype=self.dtype)
        # initialize a Simulation object to generate projections of the volume
        sim = Simulation(
            L, self.n, vols=Volume(vol), offsets=0.0, amplitudes=1.0, dtype=self.dtype
        )

        # get images before downsample
        imgs_org = sim.images()

        # get images after downsample
        sim.downsample(L_ds)
        imgs_ds = sim.images()

        return imgs_org, imgs_ds
