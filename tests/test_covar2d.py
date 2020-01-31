import os
import numpy as np
from unittest import TestCase

from aspire.source.simulation import Simulation
from aspire.basis.ffb_2d import FFBBasis2D
from aspire.utils.filters import RadialCTFFilter
from aspire.utils.preprocess import downsample
from aspire.utils.coor_trans import qrand_rots
from aspire.utils.preprocess import vol2img
from aspire.utils.blk_diag_func import radial_filter2fb_mat
from aspire.utils.matrix import anorm
from aspire.utils.matlab_compat import randn

from aspire.estimation.covar2d import RotCov2D


import os.path
DATA_DIR = os.path.join(os.path.dirname(__file__), 'saved_test_data')


class Cov2DTestCase(TestCase):
    def setUp(self):

        L = 8
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
            n=n,
            C=C,
            filters=filters
        )

        vols = np.load(os.path.join(DATA_DIR, 'clean70SRibosome_vol.npy'))
        vols = vols[..., np.newaxis]
        vols = downsample(vols, (L*np.ones(3, dtype=int)))
        sim.vols = vols

        self.basis = FFBBasis2D((L, L))
        # use new methods to generate random rotations and clean images
        sim.rots = qrand_rots(n, seed=0)
        self.imgs_clean = vol2img(vols[..., 0], sim.rots)

        self.h_idx = np.array([filters.index(f) for f in sim.filters])
        self.filters = filters
        self.h_ctf_fb = [radial_filter2fb_mat(filt.evaluate, self.basis) for filt in self.filters]

        self.imgs_ctf_clean = sim.eval_filters(self.imgs_clean)

        sim.cache(self.imgs_ctf_clean)

        power_clean = anorm(self.imgs_ctf_clean)**2/np.size(self.imgs_ctf_clean)
        self.noise_var = power_clean/SNR
        self.imgs_ctf_noise = self.imgs_ctf_clean + np.sqrt(self.noise_var)*randn(L, L, n, seed=0)

        self.cov2d = RotCov2D(self.basis)
        self.coeff_clean = self.basis.evaluate_t(self.imgs_clean)
        self.coeff = self.basis.evaluate_t(self.imgs_ctf_noise)

    def tearDown(self):
        pass

    def test01GetMean(self):
        results = np.load(os.path.join(DATA_DIR, 'clean70SRibosome_cov2d_mean.npy'))
        self.mean_coeff = self.cov2d._get_mean(self.coeff_clean)
        self.assertTrue(np.allclose(results.flatten(), self.mean_coeff))

    def test02GetCovar(self):
        results = np.load(os.path.join(DATA_DIR, 'clean70SRibosome_cov2d_covar.npy'))
        self.covar_coeff = self.cov2d._get_covar(self.coeff_clean)
        im = 0
        for mat in results[0].tolist():
            self.assertTrue(np.allclose(mat, self.covar_coeff[im]))
            im += 1

    def test03GetMeanCTF(self):
        results = np.load(os.path.join(DATA_DIR, 'clean70SRibosome_cov2d_meanctf.npy'))
        self.mean_coeff_ctf = self.cov2d.get_mean(self.coeff, self.h_ctf_fb, self.h_idx)
        self.assertTrue(np.allclose(results.flatten(), self.mean_coeff_ctf))

    def test04GetCovarCTF(self):
        results = np.load(os.path.join(DATA_DIR, 'clean70SRibosome_cov2d_covarctf.npy'))
        self.covar_coeff_ctf = self.cov2d.get_covar(self.coeff, self.h_ctf_fb, self.h_idx,
                                                    noise_var=self.noise_var)
        im = 0
        for mat in results.tolist():
            self.assertTrue(np.allclose(mat, self.covar_coeff_ctf[im]))
            im += 1

    def test05GetCovarCTFShrink(self):
        results = np.load(os.path.join(DATA_DIR, 'clean70SRibosome_cov2d_covarctf_shrink.npy'))
        covar_opt = {'shrinker': 'frobenius_norm', 'verbose': 0, 'max_iter': 250, 'iter_callback': [],
                     'store_iterates': False, 'rel_tolerance': 1e-12, 'precision': 'float64'}
        self.covar_coeff_ctf_shrink = self.cov2d.get_covar(self.coeff, self.h_ctf_fb, self.h_idx,
                                                           noise_var=self.noise_var, covar_est_opt=covar_opt)
        im = 0
        for mat in results.tolist():
            self.assertTrue(np.allclose(mat, self.covar_coeff_ctf_shrink[im]))
            im += 1

    def test06GetCWFCoeffs(self):
        results = np.load(os.path.join(DATA_DIR, 'clean70SRibosome_cov2d_cwf_coeff.npy'))
        self.coeff_cwf = self.cov2d.get_cwf_coeffs(self.coeff, self.h_ctf_fb, self.h_idx,
                                                   noise_var=self.noise_var)
        self.assertTrue(np.allclose(results, self.coeff_cwf))

    def test07GetCWFCoeffsIdentityCTF(self):
        results = np.load(os.path.join(DATA_DIR, 'clean70SRibosome_cov2d_cwf_coeff_noCTF.npy'))
        self.coeff_cwf_noCTF = self.cov2d.get_cwf_coeffs(self.coeff, noise_var=self.noise_var)
        self.assertTrue(np.allclose(results, self.coeff_cwf_noCTF))

    def test08GetCWFCoeffsClean(self):
        results = np.load(os.path.join(DATA_DIR, 'clean70SRibosome_cov2d_cwf_coeff_clean.npy'))
        self.coeff_cwf_clean = self.cov2d.get_cwf_coeffs(self.coeff_clean, noise_var=0)
        self.assertTrue(np.allclose(results, self.coeff_cwf_clean))
