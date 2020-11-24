from unittest import TestCase

import numpy as np

from aspire.basis.ffb_2d import FFBBasis2D
from aspire.denoising.denoiser_cov2d import DenoiserCov2D
from aspire.covariance.covar2d import BatchedRotCov2D
from aspire.source.simulation import Simulation
from aspire.operators.filters import RadialCTFFilter, ScalarFilter


class BatchedRotCov2DTestCase(TestCase):
    def setUp(self):
        n = 32
        L = 8
        dtype = np.float32

        noise_var = 0.1848
        noise_filter = ScalarFilter(dim=2, value=noise_var)
        filters = [
            RadialCTFFilter(5, 200, defocus=d, Cs=2.0, alpha=0.1)
            for d in np.linspace(1.5e4, 2.5e4, 7)
        ]

        src = Simulation(
            L, n, unique_filters=filters, dtype=dtype, noise_filter=noise_filter
        )
        im = src.images(0, src.n)

        self.dtype = dtype
        self.basis = FFBBasis2D((L, L), dtype=self.dtype)
        self.noise_var = noise_var
        self.noise_filter = noise_filter
        self.filters = filters
        self.ctf_idx = src.filter_indices
        self.ctf_fb = [f.fb_mat(self.basis) for f in src.unique_filters]
        self.coeff = self.basis.evaluate_t(im)
        self.bcov2d = BatchedRotCov2D(src, self.basis, batch_size=7)
        self.denoisor = DenoiserCov2D(src, self.basis, self.noise_var)
        self.denoised_src = self.denoisor.denoise(batch_size=7)

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
        imgs_denoised_denoisor = self.denoised_src.images(0, self.denoised_src.n)

        self.assertTrue(
            np.allclose(
                imgs_denoised_bcov2d.asnumpy(), imgs_denoised_denoisor.asnumpy()
            )
        )

    def testMSE(self):
        # need larger numbers of images and higher resolution for good MSE
        img_size = 64
        num_imgs = 1024

        # set simulation object
        sim = Simulation(
            L=img_size,
            n=num_imgs,
            unique_filters=self.filters,
            offsets=0.0,
            amplitudes=1.0,
            dtype=self.dtype,
            noise_filter=self.noise_filter,
        )
        imgs_clean = sim.projections()

        # Specify the fast FB basis method for expending the 2D images
        ffbbasis = FFBBasis2D((img_size, img_size), dtype=self.dtype)
        denoisor = DenoiserCov2D(sim, ffbbasis, self.noise_var)
        denoised_src = denoisor.denoise(batch_size=64)
        imgs_denoised = denoised_src.images(0, num_imgs)
        # Calculate the normalized RMSE of the estimated images.
        nrmse_ims = (imgs_denoised - imgs_clean).norm() / imgs_clean.norm()

        self.assertTrue(nrmse_ims < 0.25)
