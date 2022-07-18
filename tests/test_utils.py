from unittest import TestCase
from unittest.mock import patch

import numpy as np
from pytest import raises

from aspire import __version__
from aspire.utils import get_full_version, powerset, utest_tolerance
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
        mu_x, mu_y = 7, -3
        s_x, s_y = 5, 6

        g = gaussian_2d(L, x0=mu_x, y0=mu_y, sigma_x=s_x, sigma_y=s_y)

        # The normalized sum across an axis should correspond to a 1d gaussian with appropriate mu, sigma, peak.
        g_x = np.sum(g, axis=0) / np.sum(g)
        g_y = np.sum(g, axis=1) / np.sum(g)

        # Corresponding 1d gaussians
        peak_x = 1 / np.sqrt(2 * np.pi * s_x**2)
        peak_y = 1 / np.sqrt(2 * np.pi * s_y**2)
        g_1d_x = gaussian_1d(L, mu=mu_x, sigma=s_x, peak=peak_x)
        g_1d_y = gaussian_1d(L, mu=mu_y, sigma=s_y, peak=peak_y)

        # Assert all-close
        self.assertTrue(np.allclose(g_x, g_1d_x))
        self.assertTrue(np.allclose(g_y, g_1d_y))

    def testGaussian3d(self):
        L = 100
        mu = (0, 5, 10)
        sigma = (5, 7, 9)

        G = gaussian_3d(L, mu, sigma)

        # The normalized sum across two axes should correspond to a 1d gaussian with appropriate mu, sigma, peak.
        G_x = np.sum(G, axis=(0, 1)) / np.sum(G)
        G_y = np.sum(G, axis=(0, 2)) / np.sum(G)
        G_z = np.sum(G, axis=(1, 2)) / np.sum(G)

        # Corresponding 1d gaussians
        peak_x = 1 / np.sqrt(2 * np.pi * sigma[0] ** 2)
        peak_y = 1 / np.sqrt(2 * np.pi * sigma[1] ** 2)
        peak_z = 1 / np.sqrt(2 * np.pi * sigma[2] ** 2)
        g_1d_x = gaussian_1d(L, mu=mu[0], sigma=sigma[0], peak=peak_x)
        g_1d_y = gaussian_1d(L, mu=mu[1], sigma=sigma[1], peak=peak_y)
        g_1d_z = gaussian_1d(L, mu=mu[2], sigma=sigma[2], peak=peak_z)

        # Assert all-close
        self.assertTrue(np.allclose(G_x, g_1d_x))
        self.assertTrue(np.allclose(G_y, g_1d_y))
        self.assertTrue(np.allclose(G_z, g_1d_z))
