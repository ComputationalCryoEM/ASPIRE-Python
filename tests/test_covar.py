import os
import numpy as np
from scipy.cluster.vq import kmeans2

from unittest import TestCase
from unittest.mock import patch
import pytest

from aspyre.source import SourceFilter
from aspyre.source.simulation import Simulation
from aspyre.basis.fb_3d import FBBasis3D
from aspyre.imaging.filters import RadialCTFFilter
from aspyre.estimation.mean import MeanEstimator
from aspyre.estimation.covar import CovarianceEstimator
from aspyre.utils.matrix import eigs
from aspyre.utils.misc import src_wiener_coords
from aspyre.utils.matlab_compat import Random

import os.path
DATA_DIR = os.path.join(os.path.dirname(__file__), 'saved_test_data')


class CovarianceTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = Simulation(
            n=1024,
            filters=SourceFilter(
                [RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)],
                n=1024
            )
        )
        basis = FBBasis3D((8, 8, 8))
        cls.noise_variance = 0.0030762743633643615

        cls.mean_estimator = MeanEstimator(cls.sim, basis)
        cls.mean_est = np.load(os.path.join(DATA_DIR, 'mean_8_8_8.npy'))

        # Passing in a mean_kernel argument to the following constructor speeds up some calculations
        cls.covar_estimator = CovarianceEstimator(cls.sim, basis, mean_kernel=cls.mean_estimator.kernel, preconditioner='none')
        cls.covar_estimator_with_preconditioner = CovarianceEstimator(cls.sim, basis, mean_kernel=cls.mean_estimator.kernel, preconditioner='circulant')

    def tearDown(self):
        pass

    @pytest.mark.expensive
    def testCovariance(self):
        covar_est = self.covar_estimator_with_preconditioner.estimate(self.mean_est, self.noise_variance)

        self.assertTrue(np.allclose(
            np.array([
                [0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -4.97200141e-17,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                [0.00000000e+00,  0.00000000e+00, -1.12423131e-02, -7.82940670e-03, -5.25730644e-02, -5.12982911e-02,  5.46448594e-03,  0.00000000e+00],
                [0.00000000e+00, -1.45263802e-02,  1.74773651e-02,  2.37465013e-02, -5.82802387e-02, -2.18693634e-02, -8.17610441e-03, -5.34913194e-02],
                [0.00000000e+00, -1.25388002e-02,  3.32579746e-02, -2.47232520e-02, -1.45779671e-01, -9.90902473e-02, -1.21664500e-01, -1.86567008e-01],
                [5.23168570e-17,  1.48361175e-02,  6.39768940e-02,  2.31061220e-01,  1.14505159e-01, -1.27282900e-01, -1.20426781e-01, -9.83754536e-02],
                [0.00000000e+00, -2.77886166e-02, -2.70706646e-02,  3.27305040e-01,  3.52852148e-01,  1.95510582e-03, -5.53571860e-02, -2.08399248e-02],
                [0.00000000e+00, -2.60699879e-02, -1.84686293e-02,  1.30268283e-01,  1.36522253e-01,  8.11090183e-02,  3.50443711e-02, -1.21283276e-02],
                [0.00000000e+00,  0.00000000e+00,  6.67517637e-02,  1.12721933e-01, -8.87693429e-03,  2.99613531e-02,  4.14024319e-02,  0.00000000e+00]
            ]),
            covar_est[:, :, 4, 4, 4, 4],
            atol=1e-4
        ))

    @patch('scipy.sparse.linalg.cg')
    def testCovariance1(self, cg):
        cg_return_value = np.load(os.path.join(DATA_DIR, 'cg_return_value.npy'))
        cg.return_value = cg_return_value, 0  # 0 = convergence success

        covar_est = self.covar_estimator.estimate(self.mean_est, self.noise_variance)

        # Since we're only mocking a linear system solver, ensure that we did return the solution
        # for the argument we got called with.
        # 'call_args' is a tuple with the first member being the ordered arguments of the Mock call
        # In our case (in order) - the LinearOperator and 'b' (the RHS of the linear system)
        op, b = cg.call_args[0]
        self.assertTrue(np.allclose(b, op(cg_return_value), atol=1e-5))

        self.assertTrue(np.allclose(
            np.array([
                [0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -4.97200141e-17,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                [0.00000000e+00,  0.00000000e+00, -1.12423131e-02, -7.82940670e-03, -5.25730644e-02, -5.12982911e-02,  5.46448594e-03,  0.00000000e+00],
                [0.00000000e+00, -1.45263802e-02,  1.74773651e-02,  2.37465013e-02, -5.82802387e-02, -2.18693634e-02, -8.17610441e-03, -5.34913194e-02],
                [0.00000000e+00, -1.25388002e-02,  3.32579746e-02, -2.47232520e-02, -1.45779671e-01, -9.90902473e-02, -1.21664500e-01, -1.86567008e-01],
                [5.23168570e-17,  1.48361175e-02,  6.39768940e-02,  2.31061220e-01,  1.14505159e-01, -1.27282900e-01, -1.20426781e-01, -9.83754536e-02],
                [0.00000000e+00, -2.77886166e-02, -2.70706646e-02,  3.27305040e-01,  3.52852148e-01,  1.95510582e-03, -5.53571860e-02, -2.08399248e-02],
                [0.00000000e+00, -2.60699879e-02, -1.84686293e-02,  1.30268283e-01,  1.36522253e-01,  8.11090183e-02,  3.50443711e-02, -1.21283276e-02],
                [0.00000000e+00,  0.00000000e+00,  6.67517637e-02,  1.12721933e-01, -8.87693429e-03,  2.99613531e-02,  4.14024319e-02,  0.00000000e+00]
            ]),
            covar_est[:, :, 4, 4, 4, 4],
            atol=1e-4
        ))

    @patch('scipy.sparse.linalg.cg')
    def testCovariance2(self, cg):
        # Essentially the same as above, except that our estimator now has a preconditioner
        cg_return_value = np.load(os.path.join(DATA_DIR, 'cg_return_value.npy'))
        cg.return_value = cg_return_value, 0  # 0 = convergence success

        covar_est = self.covar_estimator_with_preconditioner.estimate(self.mean_est, self.noise_variance)

        self.assertTrue(np.allclose(
            np.array([
                [0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -4.97200141e-17,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                [0.00000000e+00,  0.00000000e+00, -1.12423131e-02, -7.82940670e-03, -5.25730644e-02, -5.12982911e-02,  5.46448594e-03,  0.00000000e+00],
                [0.00000000e+00, -1.45263802e-02,  1.74773651e-02,  2.37465013e-02, -5.82802387e-02, -2.18693634e-02, -8.17610441e-03, -5.34913194e-02],
                [0.00000000e+00, -1.25388002e-02,  3.32579746e-02, -2.47232520e-02, -1.45779671e-01, -9.90902473e-02, -1.21664500e-01, -1.86567008e-01],
                [5.23168570e-17,  1.48361175e-02,  6.39768940e-02,  2.31061220e-01,  1.14505159e-01, -1.27282900e-01, -1.20426781e-01, -9.83754536e-02],
                [0.00000000e+00, -2.77886166e-02, -2.70706646e-02,  3.27305040e-01,  3.52852148e-01,  1.95510582e-03, -5.53571860e-02, -2.08399248e-02],
                [0.00000000e+00, -2.60699879e-02, -1.84686293e-02,  1.30268283e-01,  1.36522253e-01,  8.11090183e-02,  3.50443711e-02, -1.21283276e-02],
                [0.00000000e+00,  0.00000000e+00,  6.67517637e-02,  1.12721933e-01, -8.87693429e-03,  2.99613531e-02,  4.14024319e-02,  0.00000000e+00]
            ]),
            covar_est[:, :, 4, 4, 4, 4],
            atol=1e-4
        ))

    def testMeanEvaluation(self):
        metrics = self.sim.eval_mean(self.mean_est)
        self.assertAlmostEqual(2.6641160559507631, metrics['err'], places=4)
        self.assertAlmostEqual(0.17659437048516261, metrics['rel_err'], places=4)
        self.assertAlmostEqual(0.9849211540734224, metrics['corr'], places=4)

    def testCovarEvaluation(self):
        covar_est = np.load(os.path.join(DATA_DIR, 'covar_8_8_8_8_8_8.npy'))
        metrics = self.sim.eval_covar(covar_est)
        self.assertAlmostEqual(13.322721549011165, metrics['err'], places=4)
        self.assertAlmostEqual(0.59589360739385577, metrics['rel_err'], places=4)
        self.assertAlmostEqual(0.84053472877416313, metrics['corr'], places=4)

    def testEigsEvaluation(self):
        covar_est = np.load(os.path.join(DATA_DIR, 'covar_8_8_8_8_8_8.npy'))
        eigs_est, lambdas_est = eigs(covar_est, 16)

        # No. of distinct volumes
        C = 2

        # Eigenvalues and their corresponding eigenvectors are returned in descending order
        # We take the highest C-1 entries, since C-1 is the rank of the population covariance matrix.
        eigs_est_trunc = eigs_est[:, :, :, :C-1]
        lambdas_est_trunc = lambdas_est[:C-1, :C-1]

        metrics = self.sim.eval_eigs(eigs_est_trunc, lambdas_est_trunc)
        self.assertAlmostEqual(13.09420492368651, metrics['err'], places=4)
        self.assertAlmostEqual(0.58567250265489856, metrics['rel_err'], places=4)
        self.assertAlmostEqual(0.85473300555263432, metrics['corr'], places=4)

    def testClustering(self):
        covar_est = np.load(os.path.join(DATA_DIR, 'covar_8_8_8_8_8_8.npy'))
        eigs_est, lambdas_est = eigs(covar_est, 16)

        C = 2

        eigs_est_trunc = eigs_est[:, :, :, :C-1]
        lambdas_est_trunc = lambdas_est[:C-1, :C-1]

        # Estimate the coordinates in the eigenbasis. Given the images, we find the coordinates in the basis that
        # minimize the mean squared error, given the (estimated) covariances of the volumes and the noise process.
        coords_est = src_wiener_coords(self.sim, self.mean_est, eigs_est_trunc, lambdas_est_trunc, self.noise_variance)

        # Cluster the coordinates using k-means. Again, we know how many volumes we expect, so we can use this parameter
        # here. Typically, one would take the number of clusters to be one plus the number of eigenvectors extracted.

        # Since kmeans2 relies on randomness for initialization, important to push random seed to context manager here.
        with Random(0):
            centers, vol_idx = kmeans2(coords_est.T, C)

        clustering_accuracy = self.sim.eval_clustering(vol_idx)
        self.assertEqual(clustering_accuracy, 1)
