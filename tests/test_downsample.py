from itertools import product
from unittest import TestCase

import numpy as np

from aspire.image import Image
from aspire.source import Simulation
from aspire.utils import utest_tolerance
from aspire.utils.matrix import anorm
from aspire.utils.misc import gaussian_3d
from aspire.volume import Volume


class DownsampleTestCase(TestCase):
    def setUp(self):
        self.n = 27
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

    def _testDownsample3DCase(self, L, L_ds):
        # downsampling from size L to L_ds
        imgs_org, imgs_ds = self.createVolumes(L, L_ds)
        # check resolution is correct
        self.assertEqual((self.n, L_ds, L_ds, L_ds), imgs_ds.shape)
        # check center points for all volumes
        self.assertTrue(self.checkCenterPoint(imgs_org, imgs_ds))
        # check signal energy is conserved
        self.assertTrue(self.checkSignalEnergy(imgs_org, imgs_ds))

    def testDownsample2D_EvenEven(self):
        # source resolution: 64
        # target resolution: 32
        self._testDownsample2DCase(64, 32)

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

    def testDownsample3D_EvenEven(self):
        # source resolution: 64
        # target resolution: 32
        self._testDownsample3DCase(64, 32)

    def testDownsample3D_EvenOdd(self):
        # source resolution: 64
        # target resolution: 33
        self._testDownsample3DCase(64, 33)

    def testDownsample3D_OddOdd(self):
        # source resolution: 65
        # target resolution: 33
        self._testDownsample3DCase(65, 33)

    def testDownsample3D_OddEven(self):
        # source resolution: 65
        # target resolution: 32
        self._testDownsample3DCase(64, 32)

    def checkCenterPoint(self, imgs_org, imgs_ds):
        # Check that center point is the same after ds
        L = imgs_org.shape[-1]
        L_ds = imgs_ds.shape[-1]
        # grab the center point via tuple of length 2 or 3 (image or volume)
        center_org, center_ds = (L // 2, L // 2), (L_ds // 2, L_ds // 2)
        if isinstance(imgs_org, Volume):
            center_org += (L // 2,)
            center_ds += (L_ds // 2,)
        return np.allclose(
            imgs_org[(..., *center_org)],
            imgs_ds[(..., *center_ds)],
            atol=1e-3,  # this indeterminacy only appears with the 3D centerpoints..
        )

    def checkSignalEnergy(self, imgs_org, imgs_ds):
        # check conservation of energy after downsample
        if isinstance(imgs_org, Image):
            L = imgs_org.res
            max_resolution = imgs_ds.res
            return np.allclose(
                anorm(imgs_org.asnumpy(), axes=(1, 2)) / L,
                anorm(imgs_ds.asnumpy(), axes=(1, 2)) / max_resolution,
                atol=utest_tolerance(self.dtype),
            )
        elif isinstance(imgs_org, Volume):
            L = imgs_org.resolution
            L_ds = imgs_ds.resolution
            return np.allclose(
                anorm(imgs_org.asnumpy(), axes=(1, 2, 3)) / (L ** (3 / 2)),
                anorm(imgs_ds.asnumpy(), axes=(1, 2, 3)) / (L_ds ** (3 / 2)),
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
        imgs_org = sim.images(start=0, num=self.n)

        # get images after downsample
        sim.downsample(L_ds)
        imgs_ds = sim.images(start=0, num=self.n)

        return imgs_org, imgs_ds

    def createVolumes(self, L, L_ds):
        # generate a set of volumes with various anisotropic spreads
        sigmas = list(product([L * 0.1, L * 0.2, L * 0.3], repeat=3))
        vol = Volume(
            np.array([gaussian_3d(L, sigma=s, dtype=self.dtype) for s in sigmas])
        )
        return vol, vol.downsample(L_ds)
