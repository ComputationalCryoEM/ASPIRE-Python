from unittest import TestCase
from unittest.mock import patch

import numpy as np
from pytest import raises
from scipy.stats import norm

from aspire import __version__
from aspire.utils import get_full_version, powerset, utest_tolerance
from aspire.utils.misc import gaussian_2d, gaussian_3d


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
        s_x, s_y = 10, 15

        g = gaussian_2d(L, x0=mu_x, y0=mu_y, sigma_x=s_x, sigma_y=s_y)

        # The normalized sum across an axis should correspond to a 1d pdf with associated mu, sigma.
        g_x = np.sum(g, axis=1) / np.sum(g)
        g_y = np.sum(g, axis=0) / np.sum(g)

        # Corresponding 1d pdf's.
        pdf_x = norm.pdf(np.arange(L), L // 2 + mu_x, s_x)
        pdf_y = norm.pdf(np.arange(L), L // 2 + mu_y, s_y)

        # Assert that the root mean squared error is small.
        tol = 0.02
        self.assertTrue(np.sqrt(np.sum((g_x - pdf_x) ** 2) / L) < tol)
        self.assertTrue(np.sqrt(np.sum((g_y - pdf_y) ** 2) / L) < tol)

    def testGaussian3d(self):
        L = 100
        mu = (0, 5, 10)
        sigma = (5, 10, 15)

        G = gaussian_3d(L, mu, sigma)

        # The normalized sum across two axes should correspond to a 1d pdf with proper mu, sigma.
        G_x = np.sum(G, axis=(1, 2)) / np.sum(G)
        G_y = np.sum(G, axis=(0, 2)) / np.sum(G)
        G_z = np.sum(G, axis=(0, 1)) / np.sum(G)

        # Corresponding 1d pdf's.
        pdf_x = norm.pdf(np.arange(L), L // 2 + mu[0], sigma[0])
        pdf_y = norm.pdf(np.arange(L), L // 2 + mu[1], sigma[1])
        pdf_z = norm.pdf(np.arange(L), L // 2 + mu[2], sigma[2])

        # Assert that the root mean squared error is small..
        tol = 0.02
        self.assertTrue(np.sqrt(np.sum((G_x - pdf_x) ** 2) / L) < tol)
        self.assertTrue(np.sqrt(np.sum((G_y - pdf_y) ** 2) / L) < tol)
        self.assertTrue(np.sqrt(np.sum((G_z - pdf_z) ** 2) / L) < tol)
