import os
import numpy as np

from unittest import TestCase

from aspire.source.simulation import Simulation
from aspire.utils.filters import RadialCTFFilter
from aspire.utils.preprocess import downsample
from aspire.utils.coor_trans import qrand_rots
from aspire.utils.preprocess import vol2img
from aspire.orientation.commonline_sync import CommLineSync

import os.path
DATA_DIR = os.path.join(os.path.dirname(__file__), 'saved_test_data')


class OrientSyncTestCase(TestCase):
    def setUp(self):

        L = 16
        n = 32
        C = 1
        SNR = 1
        pixel_size = 5
        voltage = 200
        defocus_min = 1.5e4
        defocus_max = 2.5e4
        defocus_ct = 7
        Cs = 2.0
        alpha = 0.1

        filters = [RadialCTFFilter(pixel_size, voltage, defocus=d, Cs=2.0, alpha=0.1) for d in
                   np.linspace(defocus_min, defocus_max, defocus_ct)]

        sim = Simulation(
            L=L,
            n=n,
            C=C,
            filters=filters
        )

        vols = np.load(os.path.join(DATA_DIR, 'clean70SRibosome_vol.npy'))
        vols = vols[..., np.newaxis]
        vols = downsample(vols, (L*np.ones(3, dtype=int)))
        sim.vols = vols

        # use new methods to generate random rotations and clean images
        sim.rots = qrand_rots(n, seed=0)
        imgs_clean = vol2img(vols[..., 0], sim.rots)
        sim.cache(imgs_clean)

        self.orient_est = CommLineSync(sim, L//2, 180)
        self.orient_est.build_clmatrix()
        self.orient_est.syncmatrix_vote()
        self.orient_est.estimate_rotations()
        self.est_rots = self.orient_est.rotations
        np.random.seed(0)
        self.est_shifts, _ = self.orient_est.estimate_shifts()

    def tearDown(self):
        pass

    def test01BuildCLmatrix(self):
        results = np.load(os.path.join(DATA_DIR, 'orient_est_clmatrix.npy'),
                          allow_pickle=True)
        self.assertTrue(np.allclose(results, self.orient_est.clmatrix))

    def test02SyncMatrixVote(self):
        results = np.load(os.path.join(DATA_DIR, 'orient_est_smatrix.npy'),
                          allow_pickle=True)
        self.assertTrue(np.allclose(results, self.orient_est.syncmatrix))

    def test03EstRotations(self):
        results = np.load(os.path.join(DATA_DIR, 'orient_est_rots.npy'),
                          allow_pickle=True)
        self.assertTrue(np.allclose(results, self.orient_est.rotations))

    def test04EstShifts(self):
        results = np.load(os.path.join(DATA_DIR, 'orient_est_shifts.npy'),
                          allow_pickle=True)
        self.assertTrue(np.allclose(results, self.est_shifts))
