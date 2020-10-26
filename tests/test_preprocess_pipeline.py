import os.path
from unittest import TestCase

import numpy as np

from aspire.estimation.noise import WhiteNoiseEstimator
from aspire.source.simulation import Simulation
from aspire.utils.coor_trans import grid_2d
from aspire.utils.filters import RadialCTFFilter, ScalarFilter
from aspire.utils.matrix import anorm
from aspire.volume import Volume

DATA_DIR = os.path.join(os.path.dirname(__file__), 'saved_test_data')


class PreprocessPLTestCase(TestCase):
    def setUp(self):

        self.L = 64
        self.n = 128

        self.noise_variance = 0.004
        self.noise_filter = ScalarFilter(dim=2, value=self.noise_variance)

        self.sim = Simulation(
            L=self.L,
            n=self.n,
            unique_filters=[RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)],
            noise_filter=self.noise_filter
        )
        self.imgs_org = self.sim.images(start=0, num=self.n).asnumpy()

    def testPhaseFlip(self):
        self.sim.phase_flip()
        imgs_pf = self.sim.images(start=0, num=self.n).asnumpy()

        # check energy conservation
        self.assertTrue(anorm(self.imgs_org), anorm(imgs_pf))

    def testDownsample(self):
        max_resolution = 8
        self.sim.downsample(max_resolution)
        imgs_ds = self.sim.images(start=0, num=self.n).asnumpy()

        # check resolution
        self.assertTrue(max_resolution, imgs_ds.shape[1])
        # check energy conservation after downsample
        self.assertTrue(anorm(self.imgs_org), anorm(imgs_ds))

    def testNormBackground(self):
        bg_radius = 1.0
        grid = grid_2d(self.L)
        mask = (grid['r'] > bg_radius)
        self.sim.normalize_background()
        imgs_nb = self.sim.images(start=0, num=self.n).asnumpy()
        new_mean = np.mean(imgs_nb[:, mask])
        new_variance = np.var(imgs_nb[:, mask])

        # new mean of noise should close to zero and 1 for variance
        self.assertTrue(new_mean < 1e-7 and abs(new_variance-1) < 1e-7)

    def testWhiten(self):
        noise_estimator = WhiteNoiseEstimator(self.sim)
        self.sim.whiten(noise_estimator.filter)
        imgs_wt = self.sim.images(start=0, num=self.n).asnumpy()

        # calculate correlation between two neighboring pixels from background
        corr_coef = np.corrcoef(imgs_wt[:, self.L-1, self.L-1],
                                imgs_wt[:, self.L-2, self.L-1])

        # correlation should be low
        self.assertTrue(np.abs(corr_coef[0, 1]) < 1e-1)

    def testInvertContrast(self):
        vols = Volume(np.load(os.path.join(DATA_DIR, 'clean70SRibosome_vol.npy')))
        vols1 = vols.downsample((8*np.ones(3, dtype=int)))
        vols2 = -1.0*vols1
        noise_var = 1.3957e-4
        noise_filter = ScalarFilter(dim=2, value=noise_var)
        sim1 = Simulation(
            L=8,
            n=128,
            vols=vols1,
            unique_filters=[RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)],
            noise_filter=noise_filter
        )
        sim2 = Simulation(
            L=8,
            n=128,
            vols=vols2,
            unique_filters=[RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)],
            noise_filter=noise_filter
        )

        imgs1 = sim1.images(start=0, num=128).asnumpy()
        # need to set the negative images to the second simulation object
        sim2._cached_im = -imgs1

        sim1.invert_contrast()
        sim2.invert_contrast()
        imgs1_rc = sim1.images(start=0, num=128).asnumpy()
        imgs2_rc = sim2.images(start=0, num=128).asnumpy()

        # all images should be the same after inverting contract
        self.assertTrue(np.allclose(imgs1_rc, imgs2_rc))
