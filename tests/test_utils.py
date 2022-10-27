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
from aspire.utils.misc import gaussian_1d, gaussian_2d, gaussian_3d


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

    def testGaussian2d(self):
        L = 100
        mu = (7, -3)
        sigma = (5, 6)

        g = gaussian_2d(L, mu=mu, sigma=sigma)

        # The normalized sum across an axis should correspond to a 1d gaussian with appropriate mu, sigma, peak.
        g_x = np.sum(g, axis=0) / np.sum(g)
        g_y = np.sum(g, axis=1) / np.sum(g)

        # Corresponding 1d gaussians
        peak_x = 1 / np.sqrt(2 * np.pi * sigma[0] ** 2)
        peak_y = 1 / np.sqrt(2 * np.pi * sigma[1] ** 2)
        g_1d_x = peak_x * gaussian_1d(L, mu=mu[0], sigma=sigma[0])
        g_1d_y = peak_y * gaussian_1d(L, mu=mu[1], sigma=sigma[1])

        # Assert all-close
        self.assertTrue(np.allclose(g_x, g_1d_x))
        self.assertTrue(np.allclose(g_y, g_1d_y))

    @parameterized.expand([("zyx",), ("xyz")])
    def testGaussian3d(self, indexing):
        L = 100
        mu = (0, 5, 10)
        sigma = (5, 7, 9)

        G = gaussian_3d(L, mu, sigma, indexing=indexing)

        # The normalized sum across two axes should correspond to a 1d gaussian with appropriate mu, sigma, peak.
        G_x = np.sum(G, axis=(1, 2)) / np.sum(G)
        G_y = np.sum(G, axis=(0, 2)) / np.sum(G)
        G_z = np.sum(G, axis=(0, 1)) / np.sum(G)

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
