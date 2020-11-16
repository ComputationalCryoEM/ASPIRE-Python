from unittest import TestCase

import numpy as np

from aspire.basis.ffb_2d import FFBBasis2D
from aspire.estimation.covar2d import BatchedRotCov2D, RotCov2D
from aspire.source.simulation import Simulation
from aspire.utils.filters import RadialCTFFilter, ScalarFilter
from aspire.utils.types import utest_tolerance


class BatchedRotCov2DTestCase(TestCase):
    def setUp(self):
        n = 32
        L = 8
        filters = [
            RadialCTFFilter(5, 200, defocus=d, Cs=2.0, alpha=0.1)
            for d in np.linspace(1.5e4, 2.5e4, 7)
        ]
        self.dtype = np.float32
        self.noise_var = 0.1848
        noise_filter = ScalarFilter(dim=2, value=self.noise_var)

        self.src = Simulation(
            L, n, unique_filters=filters, dtype=self.dtype, noise_filter=noise_filter
        )
        self.basis = FFBBasis2D((L, L), dtype=self.dtype)
        self.coeff = self.basis.evaluate_t(self.src.images(0, self.src.n))

        self.ctf_idx = self.src.filter_indices
        self.ctf_fb = [f.fb_mat(self.basis) for f in self.src.unique_filters]

        self.cov2d = RotCov2D(self.basis)
        self.bcov2d = BatchedRotCov2D(self.src, self.basis, batch_size=7)

    def tearDown(self):
        pass

    def blk_diag_allclose(self, blk_diag_a, blk_diag_b, atol=1e-8):
        close = True
        for blk_a, blk_b in zip(blk_diag_a, blk_diag_b):
            close = close and np.allclose(blk_a, blk_b, atol=atol)
        return close

    def testMeanCovar(self):
        # Test basic functionality against RotCov2D.

        mean_cov2d = self.cov2d.get_mean(
            self.coeff, ctf_fb=self.ctf_fb, ctf_idx=self.ctf_idx
        )
        covar_cov2d = self.cov2d.get_covar(
            self.coeff,
            mean_coeff=mean_cov2d,
            ctf_fb=self.ctf_fb,
            ctf_idx=self.ctf_idx,
            noise_var=self.noise_var,
        )

        mean_bcov2d = self.bcov2d.get_mean()
        covar_bcov2d = self.bcov2d.get_covar(noise_var=self.noise_var)

        self.assertTrue(
            np.allclose(mean_cov2d, mean_bcov2d, atol=utest_tolerance(self.dtype))
        )

        self.assertTrue(
            self.blk_diag_allclose(
                covar_cov2d, covar_bcov2d, atol=utest_tolerance(self.dtype)
            )
        )

    def testZeroMean(self):
        # Make sure it works with zero mean (pure second moment).
        zero_coeff = np.zeros((self.basis.count,), dtype=self.dtype)

        covar_cov2d = self.cov2d.get_covar(
            self.coeff, mean_coeff=zero_coeff, ctf_fb=self.ctf_fb, ctf_idx=self.ctf_idx
        )

        covar_bcov2d = self.bcov2d.get_covar(mean_coeff=zero_coeff)

        self.assertTrue(
            self.blk_diag_allclose(
                covar_cov2d, covar_bcov2d, atol=utest_tolerance(self.dtype)
            )
        )

    def testAutoMean(self):
        # Make sure it automatically calls get_mean if needed.
        covar_cov2d = self.cov2d.get_covar(
            self.coeff, ctf_fb=self.ctf_fb, ctf_idx=self.ctf_idx
        )

        covar_bcov2d = self.bcov2d.get_covar()

        self.assertTrue(
            self.blk_diag_allclose(
                covar_cov2d, covar_bcov2d, atol=utest_tolerance(self.dtype)
            )
        )

    def testShrink(self):
        # Make sure it properly shrinks the right-hand side if specified.
        covar_est_opt = {
            "shrinker": "frobenius_norm",
            "verbose": 0,
            "max_iter": 250,
            "iter_callback": [],
            "store_iterates": False,
            "rel_tolerance": 1e-12,
            "precision": self.dtype,
        }

        covar_cov2d = self.cov2d.get_covar(
            self.coeff,
            ctf_fb=self.ctf_fb,
            ctf_idx=self.ctf_idx,
            covar_est_opt=covar_est_opt,
        )

        covar_bcov2d = self.bcov2d.get_covar(covar_est_opt=covar_est_opt)

        self.assertTrue(self.blk_diag_allclose(covar_cov2d, covar_bcov2d))

    def testAutoBasis(self):
        # Make sure basis is automatically created if not specified.
        nbcov2d = BatchedRotCov2D(self.src)

        covar_bcov2d = self.bcov2d.get_covar()
        covar_nbcov2d = nbcov2d.get_covar()

        self.assertTrue(
            self.blk_diag_allclose(
                covar_bcov2d, covar_nbcov2d, atol=utest_tolerance(self.dtype)
            )
        )

    def testCWFCoeff(self):
        # Calculate CWF coefficients using Cov2D base class
        mean_cov2d = self.cov2d.get_mean(
            self.coeff, ctf_fb=self.ctf_fb, ctf_idx=self.ctf_idx
        )
        covar_cov2d = self.cov2d.get_covar(
            self.coeff,
            ctf_fb=self.ctf_fb,
            ctf_idx=self.ctf_idx,
            noise_var=self.noise_var,
        )

        coeff_cov2d = self.cov2d.get_cwf_coeffs(
            self.coeff,
            self.ctf_fb,
            self.ctf_idx,
            mean_coeff=mean_cov2d,
            covar_coeff=covar_cov2d,
            noise_var=self.noise_var,
        )

        # Calculate CWF coefficients using Batched Cov2D class
        mean_bcov2d = self.bcov2d.get_mean()
        covar_bcov2d = self.bcov2d.get_covar(noise_var=self.noise_var)

        coeff_bcov2d = self.bcov2d.get_cwf_coeffs(
            self.coeff,
            self.ctf_fb,
            self.ctf_idx,
            mean_bcov2d,
            covar_bcov2d,
            noise_var=self.noise_var,
        )
        self.assertTrue(
            self.blk_diag_allclose(
                coeff_cov2d,
                coeff_bcov2d,
                # Note, the Batched class has reduced resolution,
                #  compared to the non batched method.
                atol=utest_tolerance(self.dtype),
            )
        )
