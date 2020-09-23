import os
from collections import OrderedDict
from unittest import TestCase

import numpy as np

from aspire.basis.ffb_2d import FFBBasis2D
from aspire.estimation.covar2d import BatchedRotCov2D, RotCov2D
from aspire.source.simulation import Simulation
from aspire.utils.filters import RadialCTFFilter


class BatchedRotCov2DTestCase(TestCase):
    def setUp(self):
        n = 32
        L = 8

        noise_var = 0.1848

        pixel_size = 5
        voltage = 200
        defocus_min = 1.5e4
        defocus_max = 2.5e4
        defocus_ct = 7
        Cs = 2.0
        alpha = 0.1

        filters = [RadialCTFFilter(pixel_size, voltage, defocus=d, Cs=2.0, alpha=0.1)
                   for d in np.linspace(defocus_min, defocus_max, defocus_ct)]

        # Since FFBBasis2D doesn't yet implement dtype, we'll set this to double to match its built in types.
        src = Simulation(L, n, filters=filters, dtype='double')

        basis = FFBBasis2D((L, L))

        unique_filters = list(OrderedDict.fromkeys(src.filters))
        ctf_idx = np.array([unique_filters.index(f) for f in src.filters])

        ctf_fb = [f.fb_mat(basis) for f in unique_filters]

        im = src.images(0, src.n)
        coeff = basis.evaluate_t(im.data).astype(src.dtype)

        cov2d = RotCov2D(basis)
        bcov2d = BatchedRotCov2D(src, basis, batch_size=7)

        self.src = src
        self.basis = basis
        self.ctf_fb = ctf_fb
        self.ctf_idx = ctf_idx

        self.cov2d = cov2d
        self.bcov2d = bcov2d

        self.coeff = coeff

    def tearDown(self):
        pass

    def blk_diag_allclose(self, blk_diag_a, blk_diag_b):
        close = True
        for blk_a, blk_b in zip(blk_diag_a, blk_diag_b):
            close = (close and np.allclose(blk_a, blk_b))
        return close

    def test01(self):
        # Test basic functionality against RotCov2D.
        noise_var = 0.1848

        mean_cov2d = self.cov2d.get_mean(self.coeff, ctf_fb=self.ctf_fb,
                                         ctf_idx=self.ctf_idx)
        covar_cov2d = self.cov2d.get_covar(self.coeff, mean_coeff=mean_cov2d,
                                      ctf_fb=self.ctf_fb, ctf_idx=self.ctf_idx,
                                      noise_var=noise_var)

        mean_bcov2d = self.bcov2d.get_mean()
        covar_bcov2d = self.bcov2d.get_covar(noise_var=noise_var)

        self.assertTrue(np.allclose(mean_cov2d, mean_bcov2d))
        self.assertTrue(self.blk_diag_allclose(covar_cov2d, covar_bcov2d))

    def test02(self):
        # Make sure it works with zero mean (pure second moment).
        zero_coeff = np.zeros((self.basis.count,))

        covar_cov2d = self.cov2d.get_covar(self.coeff, mean_coeff=zero_coeff,
                                      ctf_fb=self.ctf_fb, ctf_idx=self.ctf_idx)

        covar_bcov2d = self.bcov2d.get_covar(mean_coeff=zero_coeff)

        self.assertTrue(self.blk_diag_allclose(covar_cov2d, covar_bcov2d))

    def test03(self):
        # Make sure it automatically calls get_mean if needed.
        covar_cov2d = self.cov2d.get_covar(self.coeff, ctf_fb=self.ctf_fb,
                                           ctf_idx=self.ctf_idx)

        covar_bcov2d = self.bcov2d.get_covar()

        self.assertTrue(self.blk_diag_allclose(covar_cov2d, covar_bcov2d))

    def test04(self):
        # Make sure it properly shrinks the right-hand side if specified.
        covar_est_opt = {'shrinker': 'frobenius_norm',
                         'verbose': 0,
                         'max_iter': 250,
                         'iter_callback': [],
                         'store_iterates': False,
                         'rel_tolerance': 1e-12,
                         'precision': 'float64'}

        covar_cov2d = self.cov2d.get_covar(self.coeff, ctf_fb=self.ctf_fb,
                                           ctf_idx=self.ctf_idx,
                                           covar_est_opt=covar_est_opt)

        covar_bcov2d = self.bcov2d.get_covar(covar_est_opt=covar_est_opt)

        self.assertTrue(self.blk_diag_allclose(covar_cov2d, covar_bcov2d))

    def test05(self):
        # Make sure basis is automatically created if not specified.
        nbcov2d = BatchedRotCov2D(self.src)

        covar_bcov2d = self.bcov2d.get_covar()
        covar_nbcov2d = nbcov2d.get_covar()

        self.assertTrue(self.blk_diag_allclose(covar_bcov2d, covar_nbcov2d))
