import os
import numpy as np

from unittest import TestCase

from aspire.orientation.commonline_sync import CLSyncVoting
from aspire.source.simulation import Simulation
from aspire.utils.filters import RadialCTFFilter
from aspire.utils.matlab_compat import Random
from aspire.utils.preprocess_F import downsample
from aspire.volume import Volume

import os.path
DATA_DIR = os.path.join(os.path.dirname(__file__), 'saved_test_data')


class OrientSyncTestCase(TestCase):

    def setUp(self):
        L = 32
        n = 64
        pixel_size = 5
        voltage = 200
        defocus_min = 1.5e4
        defocus_max = 2.5e4
        defocus_ct = 7
        Cs = 2.0
        alpha = 0.1

        filters = [RadialCTFFilter(pixel_size, voltage, defocus=d, Cs=Cs, alpha=alpha) for d in
                   np.linspace(defocus_min, defocus_max, defocus_ct)]

        vols = np.load(os.path.join(DATA_DIR, 'clean70SRibosome_vol.npy'))
        vols = vols[..., np.newaxis]
        vols = downsample(vols, (L*np.ones(3, dtype=int)))
        vols = Volume(np.moveaxis(vols, -1, 0)) #RCOPT

        sim = Simulation(
            L=L,
            n=n,
            vols=vols,
            filters=filters
        )

        self.orient_est = CLSyncVoting(sim, L // 2, 36)

    def tearDown(self):
        pass

    def testBuildCLmatrix(self):
        self.orient_est.build_clmatrix()
        results = np.load(os.path.join(DATA_DIR, 'orient_est_clmatrix.npy'))
        self.assertTrue(np.allclose(results, self.orient_est.clmatrix))

    def testSyncMatrixVote(self):
        self.orient_est.syncmatrix_vote()
        results = np.load(os.path.join(DATA_DIR, 'orient_est_smatrix.npy'))
        self.assertTrue(np.allclose(results, self.orient_est.syncmatrix))

    def testEstRotations(self):
        self.orient_est.estimate_rotations()
        results = np.load(os.path.join(DATA_DIR, 'orient_est_rots.npy'))
        self.assertTrue(np.allclose(results, self.orient_est.rotations))

    def testEstShifts(self):
        # need to rerun explicitly the estimation of rotations
        self.orient_est.estimate_rotations()
        with Random(0):
            self.est_shifts = self.orient_est.estimate_shifts()
        results = np.load(os.path.join(DATA_DIR, 'orient_est_shifts.npy'))
        self.assertTrue(np.allclose(results, self.est_shifts))
