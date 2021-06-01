import os
import os.path
from unittest import TestCase

import numpy as np
from parameterized import parameterized
from pytest import raises

from aspire.basis import FFBBasis2D
from aspire.covariance import RotCov2D
from aspire.operators import RadialCTFFilter, ScalarFilter
from aspire.source.simulation import Simulation
from aspire.utils import utest_tolerance
from aspire.volume import Volume

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class Cov2DTestCase(TestCase):
    # These class variables support parameterized arg checking in `testShrinkers`
    shrinkers = [(None,), "frobenius_norm", "operator_norm", "soft_threshold"]
    bad_shrinker_inputs = ["None", "notashrinker", ""]

    def setUp(self):
        self.dtype = np.float32

        L = 8
        n = 32
        pixel_size = 5.0 * 65 / L
        voltage = 200
        defocus_min = 1.5e4
        defocus_max = 2.5e4
        defocus_ct = 7

        self.noise_var = 1.3957e-4
        noise_filter = ScalarFilter(dim=2, value=self.noise_var)

        unique_filters = [
            RadialCTFFilter(pixel_size, voltage, defocus=d, Cs=2.0, alpha=0.1)
            for d in np.linspace(defocus_min, defocus_max, defocus_ct)
        ]

        vols = Volume(
            np.load(os.path.join(DATA_DIR, "clean70SRibosome_vol.npy")).astype(
                self.dtype
            )
        )  # RCOPT
        vols = vols.downsample((L * np.ones(3, dtype=int))) * 1.0e3
        # Since FFBBasis2D doesn't yet implement dtype, we'll set this to double to match its built in types.
        sim = Simulation(
            n=n,
            L=L,
            vols=vols,
            unique_filters=unique_filters,
            offsets=0.0,
            amplitudes=1.0,
            dtype=self.dtype,
            noise_filter=noise_filter,
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
        results = np.load(os.path.join(DATA_DIR, "clean70SRibosome_cov2d_mean.npy"))
        self.mean_coeff = self.cov2d._get_mean(self.coeff_clean)
        self.assertTrue(np.allclose(results, self.mean_coeff))

    def testGetCovar(self):
        results = np.load(
            os.path.join(DATA_DIR, "clean70SRibosome_cov2d_covar.npy"),
            allow_pickle=True,
        )
        self.covar_coeff = self.cov2d._get_covar(self.coeff_clean)

        for im, mat in enumerate(results.tolist()):
            self.assertTrue(np.allclose(mat, self.covar_coeff[im]))

    def testGetMeanCTF(self):
        results = np.load(os.path.join(DATA_DIR, "clean70SRibosome_cov2d_meanctf.npy"))
        self.mean_coeff_ctf = self.cov2d.get_mean(self.coeff, self.h_ctf_fb, self.h_idx)
        self.assertTrue(np.allclose(results, self.mean_coeff_ctf))

    def testGetCovarCTF(self):
        results = np.load(
            os.path.join(DATA_DIR, "clean70SRibosome_cov2d_covarctf.npy"),
            allow_pickle=True,
        )
        self.covar_coeff_ctf = self.cov2d.get_covar(
            self.coeff, self.h_ctf_fb, self.h_idx, noise_var=self.noise_var
        )
        for im, mat in enumerate(results.tolist()):
            self.assertTrue(np.allclose(mat, self.covar_coeff_ctf[im]))

    def testGetCovarCTFShrink(self):
        results = np.load(
            os.path.join(DATA_DIR, "clean70SRibosome_cov2d_covarctf_shrink.npy"),
            allow_pickle=True,
        )
        covar_opt = {
            "shrinker": "frobenius_norm",
            "verbose": 0,
            "max_iter": 250,
            "iter_callback": [],
            "store_iterates": False,
            "rel_tolerance": 1e-12,
            "precision": self.dtype,
        }
        self.covar_coeff_ctf_shrink = self.cov2d.get_covar(
            self.coeff,
            self.h_ctf_fb,
            self.h_idx,
            noise_var=self.noise_var,
            covar_est_opt=covar_opt,
        )

        for im, mat in enumerate(results.tolist()):
            self.assertTrue(np.allclose(mat, self.covar_coeff_ctf_shrink[im]))

    def testGetCWFCoeffs(self):
        results = np.load(
            os.path.join(DATA_DIR, "clean70SRibosome_cov2d_cwf_coeff.npy")
        )
        self.coeff_cwf = self.cov2d.get_cwf_coeffs(
            self.coeff, self.h_ctf_fb, self.h_idx, noise_var=self.noise_var
        )
        self.assertTrue(
            np.allclose(results, self.coeff_cwf, atol=utest_tolerance(self.dtype))
        )

    def testGetCWFCoeffsIdentityCTF(self):
        results = np.load(
            os.path.join(DATA_DIR, "clean70SRibosome_cov2d_cwf_coeff_noCTF.npy")
        )
        self.coeff_cwf_noCTF = self.cov2d.get_cwf_coeffs(
            self.coeff, noise_var=self.noise_var
        )
        self.assertTrue(
            np.allclose(results, self.coeff_cwf_noCTF, atol=utest_tolerance(self.dtype))
        )

    def testGetCWFCoeffsClean(self):
        results = np.load(
            os.path.join(DATA_DIR, "clean70SRibosome_cov2d_cwf_coeff_clean.npy")
        )
        self.coeff_cwf_clean = self.cov2d.get_cwf_coeffs(self.coeff_clean, noise_var=0)
        self.assertTrue(
            np.allclose(results, self.coeff_cwf_clean, atol=utest_tolerance(self.dtype))
        )

    def testGetCWFCoeffsCleanCTF(self):
        """
        Test case of clean images (coeff_clean and noise_var=0)
        while using a non Identity CTF.

        This case may come up when a developer switches between
        clean and dirty images.
        """

        coeff_cwf = self.cov2d.get_cwf_coeffs(
            self.coeff_clean, self.h_ctf_fb, self.h_idx, noise_var=0
        )

        # Reconstruct images from output of get_cwf_coeffs
        img_est = self.basis.evaluate(coeff_cwf)
        # Compare with clean images
        delta = np.mean(np.square((self.imgs_clean - img_est).asnumpy()))
        self.assertTrue(delta < 0.02)

    def testGetCWFCoeffsCTFargs(self):
        """
        Test we raise when user supplies incorrect CTF arguments,
        and that the error message matches.
        """

        with raises(RuntimeError, match=r".*Given ctf_fb.*"):
            _ = self.cov2d.get_cwf_coeffs(
                self.coeff, self.h_ctf_fb, None, noise_var=self.noise_var
            )

    # Note, parameterized module can be removed at a later date
    # and replaced with pytest if ASPIRE-Python moves away from
    # the TestCase class style tests.
    # Paramaterize over known shrinkers and some bad values
    @parameterized.expand(shrinkers + bad_shrinker_inputs)
    def testShrinkers(self, shrinker):
        """Test all the shrinkers we know about run without crashing,
        and check we raise with specific message for unsupporting shrinker arg."""

        if shrinker in self.bad_shrinker_inputs:
            with raises(AssertionError, match="Unsupported shrink method"):
                _ = self.cov2d.get_covar(
                    self.coeff_clean, covar_est_opt={"shrinker": shrinker}
                )
            return

        results = np.load(
            os.path.join(DATA_DIR, "clean70SRibosome_cov2d_covar.npy"),
            allow_pickle=True,
        )

        covar_coeff = self.cov2d.get_covar(
            self.coeff_clean, covar_est_opt={"shrinker": shrinker}
        )

        for im, mat in enumerate(results.tolist()):
            self.assertTrue(
                np.allclose(mat, covar_coeff[im], atol=utest_tolerance(self.dtype))
            )
