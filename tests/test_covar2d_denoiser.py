from unittest import TestCase

import numpy as np

from aspire.basis.ffb_2d import FFBBasis2D
from aspire.denoising.denoiser_cov2d import DenoiserCov2D
from aspire.estimation.covar2d import BatchedRotCov2D
from aspire.source.simulation import Simulation
from aspire.utils.filters import RadialCTFFilter, ScalarFilter


class BatchedRotCov2DTestCase(TestCase):
    def setUp(self):
        n = 32
        L = 8
        self.dtype = np.float32

        self.noise_var = 0.1848
        noise_filter = ScalarFilter(dim=2, value=self.noise_var)

        pixel_size = 5
        voltage = 200
        defocus_min = 1.5e4
        defocus_max = 2.5e4
        defocus_ct = 7

        filters = [
            RadialCTFFilter(pixel_size, voltage, defocus=d, Cs=2.0, alpha=0.1)
            for d in np.linspace(defocus_min, defocus_max, defocus_ct)
        ]

        src = Simulation(
            L, n, unique_filters=filters, dtype=self.dtype, noise_filter=noise_filter
        )

        basis = FFBBasis2D((L, L), dtype=self.dtype)

        unique_filters = src.unique_filters
        self.ctf_idx = src.filter_indices
        self.ctf_fb = [f.fb_mat(basis) for f in unique_filters]

        im = src.images(0, src.n)
        self.coeff = basis.evaluate_t(im)

        self.bcov2d = BatchedRotCov2D(src, basis, batch_size=7)
        self.denoisor = DenoiserCov2D(src, basis, self.noise_var)
        self.denoised_src = self.denoisor.denoise(batch_size=7)
        self.src = src
        self.basis = basis
        self.covar_est_opt = {
            "shrinker": "frobenius_norm",
            "verbose": 0,
            "max_iter": 250,
            "iter_callback": [],
            "store_iterates": False,
            "rel_tolerance": 1e-12,
            "precision": self.dtype,
        }

    def blk_diag_allclose(self, blk_diag_a, blk_diag_b, atol=1e-8):
        close = True
        for blk_a, blk_b in zip(blk_diag_a, blk_diag_b):
            close = close and np.allclose(blk_a, blk_b, atol=atol)
        return close

    def testMean(self):
        mean_bcov2d = self.bcov2d.get_mean()
        mean_denoisor = self.denoisor.mean_est

        self.assertTrue(np.allclose(mean_denoisor, mean_bcov2d))

    def testCovar(self):
        covar_bcov2d = self.bcov2d.get_covar(
            noise_var=self.noise_var, covar_est_opt=self.covar_est_opt
        )
        covar_denoisor = self.denoisor.covar_est

        self.assertTrue(self.blk_diag_allclose(covar_denoisor, covar_bcov2d))

    def testCWFCeoffs(self):
        mean_bcov2d = self.bcov2d.get_mean()
        covar_bcov2d = self.bcov2d.get_covar(
            noise_var=self.noise_var, covar_est_opt=self.covar_est_opt
        )
        coeffs_bcov2d = self.bcov2d.get_cwf_coeffs(
            self.coeff,
            self.ctf_fb,
            self.ctf_idx,
            mean_coeff=mean_bcov2d,
            covar_coeff=covar_bcov2d,
            noise_var=self.noise_var,
        )
        imgs_denoised_bcov2d = self.basis.evaluate(coeffs_bcov2d)
        imgs_denoised_denoisor = self.denoised_src.images(0, self.src.n)

        self.assertTrue(
            np.allclose(
                imgs_denoised_bcov2d.asnumpy(), imgs_denoised_denoisor.asnumpy()
            )
        )
