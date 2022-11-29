from unittest import TestCase
from unittest.mock import patch

import numpy as np
from parameterized import parameterized
from pytest import raises

from aspire import __version__
from aspire.utils import (
    all_pairs,
    all_triplets,
    get_full_version,
    mem_based_cpu_suggestion,
    num_procs_suggestion,
    pairs_to_linear,
    physical_core_cpu_suggestion,
    powerset,
    utest_tolerance,
    virtual_core_cpu_suggestion,
)
from aspire.utils.misc import (
    bump_3d,
    fuzzy_mask,
    gaussian_1d,
    gaussian_2d,
    gaussian_3d,
    grid_3d,
)


class UtilsTestCase(TestCase):
    def testGetFullVersion(self):
        """Test typical version string response is coherent with package."""

        self.assertTrue(get_full_version().startswith(__version__))

    @patch("os.path.isdir")
    def testGetFullVersionPath(self, d_mock):
        """Test not isdir case of get_full_version."""

        d_mock.return_value = False

        self.assertTrue(get_full_version() == __version__)

    @patch("subprocess.check_output")
    def testGetFullVersionSrc(self, p_mock):
        """Test subprocess exception case of get_full_version."""

        p_mock.side_effect = FileNotFoundError

        self.assertTrue(get_full_version() == __version__ + ".src")

    @patch("subprocess.check_output")
    def testGetFullVersionUnexpected(self, p_mock):
        """Test unexpected exception case of get_full_version."""

        p_mock.side_effect = RuntimeError

        self.assertTrue(get_full_version() == __version__ + ".x")

    def testPowerset(self):
        ref = sorted([(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)])
        s = range(1, 4)
        self.assertTrue(sorted(list(powerset(s))) == ref)

    def testGetTestTol(self):
        self.assertEqual(1e-8, utest_tolerance(np.float64))
        self.assertEqual(1e-5, utest_tolerance(np.float32))
        with raises(TypeError):
            utest_tolerance(int)

    @parameterized.expand([("yx",), ("xy",)])
    def testGaussian2d(self, indexing):
        L = 100
        # Note, `mu` and `sigma` are in (x, y) order.
        mu = (7, -3)
        sigma = (5, 6)

        g = gaussian_2d(L, mu=mu, sigma=sigma, indexing=indexing)

        # The normalized sum across an axis should correspond to a 1d gaussian with appropriate mu, sigma, peak.
        # Set axes based on 'indexing'.
        x, y = 0, 1
        if indexing == "yx":
            x, y = y, x

        g_x = np.sum(g, axis=y) / np.sum(g)
        g_y = np.sum(g, axis=x) / np.sum(g)

        # Corresponding 1d gaussians
        peak_x = 1 / np.sqrt(2 * np.pi * sigma[0] ** 2)
        peak_y = 1 / np.sqrt(2 * np.pi * sigma[1] ** 2)
        g_1d_x = peak_x * gaussian_1d(L, mu=mu[0], sigma=sigma[0])
        g_1d_y = peak_y * gaussian_1d(L, mu=mu[1], sigma=sigma[1])

        # Assert all-close
        self.assertTrue(np.allclose(g_x, g_1d_x))
        self.assertTrue(np.allclose(g_y, g_1d_y))

        # Test errors are raised with improper `mu` and `sigma` length.
        with raises(ValueError, match="`mu` must be len(2)*"):
            gaussian_2d(L, mu=(1,), sigma=sigma, indexing=indexing)
        with raises(ValueError, match="`sigma` must be*"):
            gaussian_2d(L, mu=mu, sigma=(1, 2, 3), indexing=indexing)

    @parameterized.expand([("zyx",), ("xyz")])
    def testGaussian3d(self, indexing):
        L = 100
        # Note, `mu` and `sigma` are in (x, y, z) order.
        mu = (0, 5, 10)
        sigma = (5, 7, 9)

        G = gaussian_3d(L, mu, sigma, indexing=indexing)

        # The normalized sum across two axes should correspond to a 1d gaussian with appropriate mu, sigma, peak.
        # Set axes based on 'indexing'.
        x, y, z = 0, 1, 2
        if indexing == "zyx":
            x, y, z = z, y, x

        G_x = np.sum(G, axis=(y, z)) / np.sum(G)
        G_y = np.sum(G, axis=(x, z)) / np.sum(G)
        G_z = np.sum(G, axis=(x, y)) / np.sum(G)

        # Corresponding 1d gaussians
        peak_x = 1 / np.sqrt(2 * np.pi * sigma[0] ** 2)
        peak_y = 1 / np.sqrt(2 * np.pi * sigma[1] ** 2)
        peak_z = 1 / np.sqrt(2 * np.pi * sigma[2] ** 2)
        g_1d_x = peak_x * gaussian_1d(L, mu=mu[0], sigma=sigma[0])
        g_1d_y = peak_y * gaussian_1d(L, mu=mu[1], sigma=sigma[1])
        g_1d_z = peak_z * gaussian_1d(L, mu=mu[2], sigma=sigma[2])

        # Assert all-close
        self.assertTrue(np.allclose(G_x, g_1d_x))
        self.assertTrue(np.allclose(G_y, g_1d_y))
        self.assertTrue(np.allclose(G_z, g_1d_z))

        # Test errors are raised with improper `mu` and `sigma` length.
        with raises(ValueError, match="`mu` must be len(3)*"):
            gaussian_3d(L, mu=(1, 2), sigma=sigma, indexing=indexing)
        with raises(ValueError, match="`sigma` must be*"):
            gaussian_3d(L, mu=mu, sigma=(1, 2), indexing=indexing)

    def testAllPairs(self):
        n = 25
        pairs = all_pairs(n)
        nchoose2 = n * (n - 1) // 2
        self.assertTrue(len(pairs) == nchoose2)
        self.assertTrue(len(pairs[0]) == 2)

    def testPairsToLinear(self):
        n = 10
        pairs = all_pairs(n)
        all_pairs_index = np.zeros(len(pairs))
        pairs_to_linear_index = np.zeros(len(pairs))
        for idx, (i, j) in enumerate(pairs):
            all_pairs_index[idx] = pairs.index((i, j))
            pairs_to_linear_index[idx] = pairs_to_linear(n, i, j)
        self.assertTrue(np.allclose(all_pairs_index, pairs_to_linear_index))

    def testAllTriplets(self):
        n = 25
        triplets = all_triplets(n)
        nchoose3 = n * (n - 1) * (n - 2) // 6
        self.assertTrue(len(triplets) == nchoose3)
        self.assertTrue(len(triplets[0]) == 3)

    def testGaussianScalarParam(self):
        L = 100
        sigma = 5
        mu_2d = (2, 3)
        sigma_2d = (sigma, sigma)
        mu_3d = (2, 3, 5)
        sigma_3d = (sigma, sigma, sigma)

        g_2d = gaussian_2d(L, mu_2d, sigma_2d)
        g_2d_scalar = gaussian_2d(L, mu_2d, sigma)

        g_3d = gaussian_3d(L, mu_3d, sigma_3d)
        g_3d_scalar = gaussian_3d(L, mu_3d, sigma)

        self.assertTrue(np.allclose(g_2d, g_2d_scalar))
        self.assertTrue(np.allclose(g_3d, g_3d_scalar))

    @parameterized.expand([(29,), (30,)])
    def testBump3d(self, L):
        L = L
        dtype = np.float64
        a = 10

        # Build volume of 1's and apply bump function
        volume = np.ones((L,) * 3, dtype=dtype)
        bump = bump_3d(L, spread=a, dtype=dtype)
        bumped_volume = np.multiply(bump, volume)

        # Define support for volume
        g = grid_3d(L, dtype=dtype)
        inside = g["r"] < (L - 1) / L
        outside = g["r"] >= 1

        # Test that volume is zero outside of support
        self.assertTrue(bumped_volume[outside].all() == 0)

        # Test that volume is positive inside support
        self.assertTrue((bumped_volume[inside] > 0).all())

        # Test that the center is still 1
        self.assertTrue(np.allclose(bumped_volume[(L // 2,) * 3], 1))

    def testFuzzyMask(self):
        results = np.array(
            [
                [
                    2.03406033e-06,
                    7.83534653e-05,
                    9.19567967e-04,
                    3.73368194e-03,
                    5.86559882e-03,
                    3.73368194e-03,
                    9.19567967e-04,
                    7.83534653e-05,
                ],
                [
                    7.83534653e-05,
                    2.35760928e-03,
                    2.15315317e-02,
                    7.15226076e-02,
                    1.03823087e-01,
                    7.15226076e-02,
                    2.15315317e-02,
                    2.35760928e-03,
                ],
                [
                    9.19567967e-04,
                    2.15315317e-02,
                    1.48272439e-01,
                    3.83057355e-01,
                    5.00000000e-01,
                    3.83057355e-01,
                    1.48272439e-01,
                    2.15315317e-02,
                ],
                [
                    3.73368194e-03,
                    7.15226076e-02,
                    3.83057355e-01,
                    7.69781837e-01,
                    8.96176913e-01,
                    7.69781837e-01,
                    3.83057355e-01,
                    7.15226076e-02,
                ],
                [
                    5.86559882e-03,
                    1.03823087e-01,
                    5.00000000e-01,
                    8.96176913e-01,
                    9.94134401e-01,
                    8.96176913e-01,
                    5.00000000e-01,
                    1.03823087e-01,
                ],
                [
                    3.73368194e-03,
                    7.15226076e-02,
                    3.83057355e-01,
                    7.69781837e-01,
                    8.96176913e-01,
                    7.69781837e-01,
                    3.83057355e-01,
                    7.15226076e-02,
                ],
                [
                    9.19567967e-04,
                    2.15315317e-02,
                    1.48272439e-01,
                    3.83057355e-01,
                    5.00000000e-01,
                    3.83057355e-01,
                    1.48272439e-01,
                    2.15315317e-02,
                ],
                [
                    7.83534653e-05,
                    2.35760928e-03,
                    2.15315317e-02,
                    7.15226076e-02,
                    1.03823087e-01,
                    7.15226076e-02,
                    2.15315317e-02,
                    2.35760928e-03,
                ],
            ]
        )
        fmask = fuzzy_mask((8, 8), 2, 2)
        self.assertTrue(np.allclose(results, fmask, atol=1e-7))


class MultiProcessingUtilsTestCase(TestCase):
    """
    Smoke tests.
    """

    def testMemSuggestion(self):
        self.assertTrue(isinstance(mem_based_cpu_suggestion(), int))

    def testPhySuggestion(self):
        self.assertTrue(isinstance(physical_core_cpu_suggestion(), int))

    def testVrtSuggestion(self):
        self.assertTrue(isinstance(virtual_core_cpu_suggestion(), int))

    def testGetNumMultiProcs(self):
        self.assertTrue(isinstance(num_procs_suggestion(), int))
