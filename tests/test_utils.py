from unittest import TestCase
from unittest.mock import patch

import numpy as np
from pytest import raises

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
        L = 42
        X = gaussian_2d(L, x0=2, y0=5, sigma_x=1, sigma_y=2)
        Y = gaussian_2d(L, x0=0, y0=1, sigma_x=2, sigma_y=3)

        # For jointly distributed gaussians, var(X + Y) = var(X) + var(Y) + 2covar(X,Y)
        var_X = np.var(X)
        var_Y = np.var(Y)
        cov_XY = np.cov(X.flatten(), Y.flatten())[0][1]
        self.assertTrue(
            np.allclose(np.var(X + Y), var_X + var_Y + 2 * cov_XY, atol=1e-5)
        )

    def testGaussian3d(self):
        L = 42
        X = gaussian_3d(L, mu=(0, 1, 2), sigma=(1, 2, 3))
        Y = gaussian_3d(L, mu=(1, 3, 5), sigma=(3, 1, 5))

        # For jointly distributed gaussians, var(X + Y) = var(X) + var(Y) + 2covar(X,Y)
        var_X = np.var(X)
        var_Y = np.var(Y)
        cov_XY = np.cov(X.flatten(), Y.flatten())[0][1]
        self.assertTrue(
            np.allclose(np.var(X + Y), var_X + var_Y + 2 * cov_XY, atol=1e-8)
        )
