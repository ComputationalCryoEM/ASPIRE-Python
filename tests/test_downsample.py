from itertools import product
from unittest import TestCase

import numpy as np
from parameterized import parameterized

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

    @parameterized.expand([[64, 32], [64, 33], [65, 33], [65, 32]])
    def testDownsample2DCase(self, L, L_ds):
        # downsampling from size L to L_ds
        imgs_org, imgs_ds = self.createImages(L, L_ds)
        # check resolution is correct
        self.assertEqual((self.n, L_ds, L_ds), imgs_ds.shape)
        # check center points for all images
        self.assertTrue(self.checkCenterPoint(imgs_org, imgs_ds))
        # check signal energy is conserved
        self.assertTrue(self.checkSignalEnergy(imgs_org, imgs_ds))

    @parameterized.expand([[64, 32], [64, 33], [65, 33], [65, 32]])
    def testDownsample3DCase(self, L, L_ds):
        # downsampling from size L to L_ds
        vols_org, vols_ds = self.createVolumes(L, L_ds)
        # check resolution is correct
        self.assertEqual((self.n, L_ds, L_ds, L_ds), vols_ds.shape)
        # check center points for all volumes
        self.assertTrue(self.checkCenterPoint(vols_org, vols_ds))
        # check signal energy is conserved
        self.assertTrue(self.checkSignalEnergy(vols_org, vols_ds))

    def checkCenterPoint(self, data_org, data_ds):
        # Check that center point is the same after ds
        L = data_org.shape[-1]
        L_ds = data_ds.shape[-1]
        # grab the center point via tuple of length 2 or 3 (image or volume)
        center_org, center_ds = (L // 2, L // 2), (L_ds // 2, L_ds // 2)
        # different tolerances for 2d vs 3d ...
        tolerance = utest_tolerance(self.dtype)
        if isinstance(data_org, Volume):
            center_org += (L // 2,)
            center_ds += (L_ds // 2,)
            # indeterminacy for 3D
            tolerance = 5e-2
        return np.allclose(
            data_org[(..., *center_org)],
            data_ds[(..., *center_ds)],
            atol=tolerance,
        )

    def checkSignalEnergy(self, data_org, data_ds):
        # check conservation of energy after downsample
        L = data_org.shape[-1]
        L_ds = data_ds.shape[-1]
        if isinstance(data_org, Image):
            return np.allclose(
                anorm(data_org.asnumpy(), axes=(1, 2)) / L,
                anorm(data_ds.asnumpy(), axes=(1, 2)) / L_ds,
                atol=utest_tolerance(self.dtype),
            )
        elif isinstance(data_org, Volume):
            return np.allclose(
                anorm(data_org.asnumpy(), axes=(1, 2, 3)) / (L ** (3 / 2)),
                anorm(data_ds.asnumpy(), axes=(1, 2, 3)) / (L_ds ** (3 / 2)),
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
        imgs_org = sim.images[: self.n]

        # get images after downsample
        sim.downsample(L_ds)
        imgs_ds = sim.images[: self.n]

        return imgs_org, imgs_ds

    def createVolumes(self, L, L_ds):
        # generate a set of volumes with various anisotropic spreads
        sigmas = list(product([L * 0.1, L * 0.2, L * 0.3], repeat=3))
        # get volumes before downsample
        vols_org = Volume(
            np.array([gaussian_3d(L, sigma=s, dtype=self.dtype) for s in sigmas])
        )

        # get volumes after downsample
        vols_ds = vols_org.downsample(L_ds)

        return vols_org, vols_ds
