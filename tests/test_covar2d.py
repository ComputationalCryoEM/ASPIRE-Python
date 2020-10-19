import os
import os.path
from unittest import TestCase

import numpy as np

from aspire.basis.ffb_2d import FFBBasis2D
from aspire.estimation.covar2d import RotCov2D
from aspire.source.simulation import Simulation
from aspire.utils.filters import RadialCTFFilter, ScalarFilter
from aspire.volume import Volume

DATA_DIR = os.path.join(os.path.dirname(__file__), 'saved_test_data')


class Cov2DTestCase(TestCase):
    def setUp(self):
        self.dtype = np.float32

        L = 8
        n = 32
        pixel_size = 5.0*65/L
        voltage = 200
        defocus_min = 1.5e4
        defocus_max = 2.5e4
        defocus_ct = 7

        self.noise_var = 1.3957e-4
        noise_filter = ScalarFilter(dim=2, value=self.noise_var)

        unique_filters = [RadialCTFFilter(pixel_size, voltage, defocus=d, Cs=2.0, alpha=0.1) for d in
                   np.linspace(defocus_min, defocus_max, defocus_ct)]

        vols = Volume(np.load(os.path.join(DATA_DIR, 'clean70SRibosome_vol.npy')).astype(self.dtype)) # RCOPT
        vols = vols.downsample((L*np.ones(3, dtype=int)))*1.0e3
        # Since FFBBasis2D doesn't yet implement dtype, we'll set this to double to match its built in types.
        sim = Simulation(
            n=n,
            L=L,
            vols=vols,
            unique_filters=unique_filters,
            offsets=0.0,
            amplitudes=1.0,
            dtype=self.dtype,
            noise_filter=noise_filter
        )

        self.basis = FFBBasis2D((L, L), dtype=self.dtype)

        self.h_idx = sim.filter_indices
        self.h_ctf_fb = [filt.fb_mat(self.basis) for filt in unique_filters]

        self.imgs_clean = sim.projections()
        self.imgs_ctf_clean = sim.clean_images()
        self.imgs_ctf_noise = sim.images(start=0, num=n)

        self.cov2d = RotCov2D(self.basis)
        self.coeff_clean = self.basis.evaluate_t(self.imgs_clean)
        self.coeff = self.basis.evaluate_t(self.imgs_ctf_noise)

    def tearDown(self):
        pass

    def testGetMean(self):
        results = np.load(os.path.join(DATA_DIR, 'clean70SRibosome_cov2d_mean.npy'))
        self.mean_coeff = self.cov2d._get_mean(self.coeff_clean)
        self.assertTrue(np.allclose(results, self.mean_coeff))

    def testGetCovar(self):
        results = np.load(os.path.join(DATA_DIR, 'clean70SRibosome_cov2d_covar.npy'))
        self.covar_coeff = self.cov2d._get_covar(self.coeff_clean)

        for im, mat in enumerate(results.tolist()):
            self.assertTrue(np.allclose(mat, self.covar_coeff[im]))

    def testGetMeanCTF(self):
        results = np.load(os.path.join(DATA_DIR, 'clean70SRibosome_cov2d_meanctf.npy'))
        self.mean_coeff_ctf = self.cov2d.get_mean(self.coeff, self.h_ctf_fb, self.h_idx)
        self.assertTrue(np.allclose(results, self.mean_coeff_ctf))

    def testGetCovarCTF(self):
        results = np.load(os.path.join(DATA_DIR, 'clean70SRibosome_cov2d_covarctf.npy'))
        self.covar_coeff_ctf = self.cov2d.get_covar(self.coeff, self.h_ctf_fb, self.h_idx,
                                                    noise_var=self.noise_var)
        for im, mat in enumerate(results.tolist()):
            self.assertTrue(np.allclose(mat, self.covar_coeff_ctf[im]))

    def testGetCovarCTFShrink(self):
        results = np.load(os.path.join(DATA_DIR, 'clean70SRibosome_cov2d_covarctf_shrink.npy'))
        covar_opt = {'shrinker': 'frobenius_norm', 'verbose': 0, 'max_iter': 250, 'iter_callback': [],
                     'store_iterates': False, 'rel_tolerance': 1e-12, 'precision': 'float32'}
        self.covar_coeff_ctf_shrink = self.cov2d.get_covar(self.coeff, self.h_ctf_fb, self.h_idx,
                                                           noise_var=self.noise_var, covar_est_opt=covar_opt)

        for im, mat in enumerate(results.tolist()):
            self.assertTrue(np.allclose(mat, self.covar_coeff_ctf_shrink[im]))

    def testGetCWFCoeffs(self):
        results = np.load(os.path.join(DATA_DIR, 'clean70SRibosome_cov2d_cwf_coeff.npy'))
        self.coeff_cwf = self.cov2d.get_cwf_coeffs(self.coeff, self.h_ctf_fb, self.h_idx,
                                                   noise_var=self.noise_var)
        self.assertTrue(np.allclose(results, self.coeff_cwf,
                                    atol=1e-6 if self.dtype == np.float32 else 1e-8))

    def testGetCWFCoeffsIdentityCTF(self):
        results = np.load(os.path.join(DATA_DIR, 'clean70SRibosome_cov2d_cwf_coeff_noCTF.npy'))
        self.coeff_cwf_noCTF = self.cov2d.get_cwf_coeffs(self.coeff, noise_var=self.noise_var)
        self.assertTrue(np.allclose(results, self.coeff_cwf_noCTF,
                                    atol=1e-6 if self.dtype == np.float32 else 1e-8))

    def testGetCWFCoeffsClean(self):
        results = np.load(os.path.join(DATA_DIR, 'clean70SRibosome_cov2d_cwf_coeff_clean.npy'))
        self.coeff_cwf_clean = self.cov2d.get_cwf_coeffs(self.coeff_clean, noise_var=0)
        self.assertTrue(np.allclose(results, self.coeff_cwf_clean))
