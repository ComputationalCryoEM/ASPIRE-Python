import os.path
from unittest import TestCase

import numpy as np

from aspire.noise import AnisotropicNoiseEstimator
from aspire.operators.filters import FunctionFilter, RadialCTFFilter, ScalarFilter
from aspire.source import ArrayImageSource
from aspire.source.simulation import Simulation
from aspire.utils import utest_tolerance
from aspire.utils.coor_trans import grid_2d, grid_3d
from aspire.volume import Volume

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class PreprocessPLTestCase(TestCase):
    def setUp(self):

        self.L = 64
        self.n = 128
        self.dtype = np.float32
        self.noise_filter = FunctionFilter(lambda x, y: np.exp(-(x ** 2 + y ** 2) / 2))

        self.sim = Simulation(
            L=self.L,
            n=self.n,
            unique_filters=[
                RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)
            ],
            noise_filter=self.noise_filter,
            dtype=self.dtype,
        )
        self.imgs_org = self.sim.images(start=0, num=self.n)

    def testPhaseFlip(self):
        self.sim.phase_flip()
        imgs_pf = self.sim.images(start=0, num=self.n)

        # check energy conservation
        self.assertTrue(np.allclose(self.imgs_org.norm(), imgs_pf.norm()))

    def testDownsample(self):
        # generate a 3D map with density decays as Gaussian function
        g3d = grid_3d(self.L, dtype=self.dtype)
        coords = np.array([g3d["x"].flatten(), g3d["y"].flatten(), g3d["z"].flatten()])
        sigma = 0.2
        vol = np.exp(-0.5 * np.sum(np.abs(coords / sigma) ** 2, axis=0)).astype(
            self.dtype
        )
        vol = np.reshape(vol, g3d["x"].shape)
        vols = Volume(vol)
        print("volume:")
        print(vols[0][16:49, 16:49, 16:49])

        # set noise to zero and CFT filters to unity for simulation object
        noise_var = 0
        noise_filter = ScalarFilter(dim=2, value=noise_var)
        sim = Simulation(
            L=self.L,
            n=self.n,
            vols=vols,
            offsets=0.0,
            amplitudes=1.0,
            unique_filters=[
                ScalarFilter(dim=2, value=1) for d in np.linspace(1.5e4, 2.5e4, 7)
            ],
            noise_filter=noise_filter,
            dtype=self.dtype,
        )
        # get images before downsample
        print("Rotation:")
        print(sim.rots[0])
        imgs_clean = sim.projections()
        print("Clean image:")
        print(imgs_clean[0][16:49, 16:49])
        print("CTF indices:")
        print(sim.filter_indices)
        imgs_ctf_clean = sim.clean_images()
        print("Clean image with CTF:")
        print(imgs_ctf_clean[0][16:49, 16:49])
        imgs_org = sim.images(start=0, num=self.n)
        print("Noise image:")
        print(imgs_org[0][16:49, 16:49])
        # get images after downsample
        max_resolution = 32
        sim.downsample(max_resolution)
        imgs_ds = sim.images(start=0, num=self.n)

        # Check individual grid points
        self.assertTrue(
            np.allclose(
                imgs_org[:, 32, 32],
                imgs_ds[:, 16, 16],
                atol=utest_tolerance(self.dtype),
            )
        )
        # check resolution
        self.assertTrue(np.allclose(max_resolution, imgs_ds.shape[1]))
        # check energy conservation after downsample
        delta = imgs_org.norm() / self.L - imgs_ds.norm() / max_resolution
        print("Downsample image:")
        print(imgs_ds[0][8:24, 8:24])
        print(imgs_org.norm() / self.L, imgs_ds.norm() / max_resolution, delta)
        self.assertTrue(
            np.allclose(
                imgs_org.norm() / self.L,
                imgs_ds.norm() / max_resolution,
                atol=1.0e-4,
            )
        )

    def testNormBackground(self):
        bg_radius = 1.0
        grid = grid_2d(self.L)
        mask = grid["r"] > bg_radius
        self.sim.normalize_background()
        imgs_nb = self.sim.images(start=0, num=self.n).asnumpy()
        new_mean = np.mean(imgs_nb[:, mask])
        new_variance = np.var(imgs_nb[:, mask])

        # new mean of noise should be close to zero and variance should be close to 1
        self.assertTrue(new_mean < 1e-7 and abs(new_variance - 1) < 1e-7)

    def testWhiten(self):
        noise_estimator = AnisotropicNoiseEstimator(self.sim)
        self.sim.whiten(noise_estimator.filter)
        imgs_wt = self.sim.images(start=0, num=self.n).asnumpy()

        # calculate correlation between two neighboring pixels from background
        corr_coef = np.corrcoef(
            imgs_wt[:, self.L - 1, self.L - 1], imgs_wt[:, self.L - 2, self.L - 1]
        )

        # correlation matrix should be close to identity
        self.assertTrue(np.allclose(np.eye(2), corr_coef, atol=1e-1))

    def testInvertContrast(self):
        sim1 = self.sim
        imgs1 = sim1.images(start=0, num=128)
        sim1.invert_contrast()
        imgs1_rc = sim1.images(start=0, num=128)
        # need to set the negative images to the second simulation object
        sim2 = ArrayImageSource(-imgs1)
        sim2.invert_contrast()
        imgs2_rc = sim2.images(start=0, num=128)

        # all images should be the same after inverting contrast
        self.assertTrue(np.allclose(imgs1_rc.asnumpy(), imgs2_rc.asnumpy()))
