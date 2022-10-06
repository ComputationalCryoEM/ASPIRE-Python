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
    """
    Cov2D Test without CTFFilters populated.
    """

    unique_filters = None
    h_idx = None
    h_ctf_fb = None

    # These class variables support parameterized arg checking in `testShrinkers`
    shrinkers = [(None,), "frobenius_norm", "operator_norm", "soft_threshold"]
    bad_shrinker_inputs = ["None", "notashrinker", ""]

    def setUp(self):
        self.dtype = np.float32

        self.L = L = 8
        n = 32

        self.noise_var = 1.3957e-4
        noise_filter = ScalarFilter(dim=2, value=self.noise_var)

        vols = Volume(
            np.load(os.path.join(DATA_DIR, "clean70SRibosome_vol.npy")).astype(
                self.dtype
            )
        )  # RCOPT
        vols = vols.downsample(L) * 1.0e3
        # Since FFBBasis2D doesn't yet implement dtype, we'll set this to double to match its built in types.
        self.sim = Simulation(
            n=n,
            L=L,
            vols=vols,
            unique_filters=self.unique_filters,
            offsets=0.0,
            amplitudes=1.0,
            dtype=self.dtype,
            noise_filter=noise_filter,
        )

        self.basis = FFBBasis2D((L, L), dtype=self.dtype)

        self.imgs_clean = self.sim.projections[:]
        self.imgs_ctf_clean = self.sim.clean_images[:]
        self.imgs_ctf_noise = self.sim.images[:n]

        self.cov2d = RotCov2D(self.basis)
        self.coeff_clean = self.basis.evaluate_t(self.imgs_clean)
        self.coeff = self.basis.evaluate_t(self.imgs_ctf_noise)

    def tearDown(self):
        pass

    def testGetMean(self):
        results = np.load(os.path.join(DATA_DIR, "clean70SRibosome_cov2d_mean.npy"))
        mean_coeff = self.cov2d._get_mean(self.coeff_clean)
        self.assertTrue(np.allclose(results, mean_coeff))

    def testGetCovar(self):
        results = np.load(
            os.path.join(DATA_DIR, "clean70SRibosome_cov2d_covar.npy"),
            allow_pickle=True,
        )
        covar_coeff = self.cov2d._get_covar(self.coeff_clean)

        for im, mat in enumerate(results.tolist()):
            self.assertTrue(np.allclose(mat, covar_coeff[im]))

    def testGetMeanCTF(self):
        """
        Compare `get_mean` (no CTF args) with `_get_mean` (no CTF model).
        """
        mean_coeff_ctf = self.cov2d.get_mean(self.coeff, self.h_ctf_fb, self.h_idx)
        mean_coeff = self.cov2d._get_mean(self.coeff_clean)
        self.assertTrue(np.allclose(mean_coeff_ctf, mean_coeff, atol=0.002))

    def testGetCWFCoeffsClean(self):
        results = np.load(
            os.path.join(DATA_DIR, "clean70SRibosome_cov2d_cwf_coeff_clean.npy")
        )
        coeff_cwf_clean = self.cov2d.get_cwf_coeffs(self.coeff_clean, noise_var=0)
        self.assertTrue(
            np.allclose(results, coeff_cwf_clean, atol=utest_tolerance(self.dtype))
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


class Cov2DTestCaseCTF(Cov2DTestCase):
    """
    Cov2D Test with CTFFilters populated.
    """

    @property
    def unique_filters(self):
        return [
            RadialCTFFilter(5.0 * 65 / self.L, 200, defocus=d, Cs=2.0, alpha=0.1)
            for d in np.linspace(1.5e4, 2.5e4, 7)
        ]

    @property
    def h_idx(self):
        return self.sim.filter_indices

    @property
    def h_ctf_fb(self):
        return [filt.fb_mat(self.basis) for filt in self.unique_filters]

    def testGetCWFCoeffsCTFargs(self):
        """
        Test we raise when user supplies incorrect CTF arguments,
        and that the error message matches.
        """

        with raises(RuntimeError, match=r".*Given ctf_fb.*"):
            _ = self.cov2d.get_cwf_coeffs(
                self.coeff, self.h_ctf_fb, None, noise_var=self.noise_var
            )

    def testGetMeanCTF(self):
        """
        Compare `get_mean` with saved legacy cov2d results.
        """
        results = np.load(os.path.join(DATA_DIR, "clean70SRibosome_cov2d_meanctf.npy"))
        mean_coeff_ctf = self.cov2d.get_mean(self.coeff, self.h_ctf_fb, self.h_idx)
        self.assertTrue(np.allclose(results, mean_coeff_ctf))

    def testGetCWFCoeffs(self):
        """
        Tests `get_cwf_coeffs` with poulated CTF.
        """
        results = np.load(
            os.path.join(DATA_DIR, "clean70SRibosome_cov2d_cwf_coeff.npy")
        )
        coeff_cwf = self.cov2d.get_cwf_coeffs(
            self.coeff, self.h_ctf_fb, self.h_idx, noise_var=self.noise_var
        )
        self.assertTrue(
            np.allclose(results, coeff_cwf, atol=utest_tolerance(self.dtype))
        )

    # Note, I think this file is incorrectly named...
    #   It appears to have come from operations on images with ctf applied.
    def testGetCWFCoeffsNoCTF(self):
        """
        Tests `get_cwf_coeffs` without providing CTF.  (Internally uses IdentityCTF).
        """
        results = np.load(
            os.path.join(DATA_DIR, "clean70SRibosome_cov2d_cwf_coeff_noCTF.npy")
        )
        coeff_cwf_noCTF = self.cov2d.get_cwf_coeffs(
            self.coeff, noise_var=self.noise_var
        )

        self.assertTrue(
            np.allclose(results, coeff_cwf_noCTF, atol=utest_tolerance(self.dtype))
        )

    def testGetCovarCTF(self):
        results = np.load(
            os.path.join(DATA_DIR, "clean70SRibosome_cov2d_covarctf.npy"),
            allow_pickle=True,
        )
        covar_coeff_ctf = self.cov2d.get_covar(
            self.coeff, self.h_ctf_fb, self.h_idx, noise_var=self.noise_var
        )
        for im, mat in enumerate(results.tolist()):
            self.assertTrue(np.allclose(mat, covar_coeff_ctf[im]))

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
        covar_coeff_ctf_shrink = self.cov2d.get_covar(
            self.coeff,
            self.h_ctf_fb,
            self.h_idx,
            noise_var=self.noise_var,
            covar_est_opt=covar_opt,
        )

        for im, mat in enumerate(results.tolist()):
            self.assertTrue(np.allclose(mat, covar_coeff_ctf_shrink[im]))
