import logging
import os.path
from unittest import TestCase
from unittest.case import SkipTest

import numpy as np
from parameterized import parameterized_class
from scipy.special import jv

from aspire.basis import FFBBasis2D
from aspire.image import Image
from aspire.source import Simulation
from aspire.utils import utest_tolerance
from aspire.utils.misc import grid_2d
from aspire.volume import Volume

from ._basis_util import Steerable2DMixin, UniversalBasisMixin

logger = logging.getLogger(__name__)
DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


# NOTE: Class with default values is already present, so don't list it below.
@parameterized_class(
    ("L", "dtype"),
    [
        (8, np.float64),
        (16, np.float32),
        (16, np.float64),
        (32, np.float32),
        (32, np.float64),
    ],
)
class FFBBasis2DTestCase(TestCase, Steerable2DMixin, UniversalBasisMixin):
    L = 8
    dtype = np.float32

    def setUp(self):
        self.basis = FFBBasis2D((self.L, self.L), dtype=self.dtype)
        self.seed = 9161341

    def tearDown(self):
        pass

    def _testElement(self, ell, k, sgn):
        indices = self.basis.indices()
        ells = indices["ells"]
        sgns = indices["sgns"]
        ks = indices["ks"]

        g2d = grid_2d(self.L, dtype=self.dtype)
        mask = g2d["r"] < 1

        r0 = self.basis.r0[k, ell]

        # TODO: Figure out where these factors of 1 / 2 are coming from.
        # Intuitively, the grid should go from -L / 2 to L / 2, not -L / 2 to
        # L / 4. Furthermore, there's an extra factor of 1 / 2 in the
        # definition of `im` below that may be related.
        r = g2d["r"] * self.L / 4

        im = np.zeros((self.L, self.L), dtype=self.dtype)
        im[mask] = (
            (-1) ** k
            * np.sqrt(np.pi)
            * r0
            * jv(ell, 2 * np.pi * r[mask])
            / ((2 * np.pi * r[mask]) ** 2 - r0**2)
        )

        if sgn == 1:
            im *= np.sqrt(2) * np.cos(ell * g2d["phi"])
        else:
            im *= np.sqrt(2) * np.sin(ell * g2d["phi"])

        coef_ref = np.zeros(self.basis.count, dtype=self.dtype)
        coef_ref[(ells == ell) & (sgns == sgn) & (ks == k)] = 1

        im_ref = self.basis.evaluate(coef_ref).asnumpy()[0]

        coef = self.basis.expand(im)

        # NOTE: These tolerances are expected to be rather loose since the
        # above expression for `im` is derived from the analytical formulation
        # (eq. 6 in Zhao and Singer, 2013) and does not take into account
        # discretization and other approximations.
        self.assertTrue(np.allclose(im, im_ref, atol=1e-1))
        self.assertTrue(np.allclose(coef, coef_ref, atol=1e-1))

    def testElements(self):
        ells = [1, 1, 1, 1]
        ks = [1, 2, 1, 2]
        sgns = [-1, -1, 1, 1]

        for ell, k, sgn in zip(ells, ks, sgns):
            self._testElement(ell, k, sgn)

    def testRotate(self):
        # Convergence issues for double precision.
        if np.dtype(self.dtype) is np.dtype(np.float64):
            raise SkipTest

        # Now low res (8x8) had problems;
        #  better with odd (7x7), but still not good.
        # We'll use a higher res test image.
        # fh = np.load(os.path.join(DATA_DIR, 'ffbbasis2d_xcoeff_in_8_8.npy'))[:7,:7]
        # Use a real data volume to generate a clean test image.
        v = Volume(
            np.load(os.path.join(DATA_DIR, "clean70SRibosome_vol.npy")).astype(
                np.float64
            )
        )
        src = Simulation(L=v.resolution, n=1, vols=v, dtype=v.dtype)
        # Extract, this is the original image to transform.
        x1 = src.images[0]

        # Rotate 90 degrees in cartesian coordinates.
        x2 = Image(np.rot90(x1.asnumpy(), axes=(1, 2)))

        # Express in an FB basis
        basis = FFBBasis2D((x1.res,) * 2, dtype=x1.dtype)
        v1 = basis.evaluate_t(x1)
        v2 = basis.evaluate_t(x2)
        v3 = basis.evaluate_t(x1)
        v4 = basis.evaluate_t(x1)

        # Reflect in the FB basis space
        v4 = basis.rotate(v1, 0, refl=[True])

        # Rotate in the FB basis space
        v3 = basis.rotate(v1, 2 * np.pi)
        v1 = basis.rotate(v1, -np.pi / 2)

        # Evaluate back into cartesian
        y1 = basis.evaluate(v1)
        y2 = basis.evaluate(v2)
        y3 = basis.evaluate(v3)
        y4 = basis.evaluate(v4)

        # Rotate 90
        self.assertTrue(np.allclose(y1[0], y2[0], atol=1e-4))

        # 2*pi Identity
        self.assertTrue(np.allclose(x1[0], y3[0], atol=utest_tolerance(self.dtype)))

        # Refl (flipped using flipud)
        self.assertTrue(np.allclose(np.flipud(x1[0]), y4[0], atol=1e-4))

    def testShift(self):
        """
        Compare shifting using Image with shifting provided by the Basis.

        Note the Basis shift method converts from FB to Image space and back.
        """

        n_img = 3
        test_shift = np.array([10, 2])

        # Construct some synthetic data
        v = Volume(
            np.load(os.path.join(DATA_DIR, "clean70SRibosome_vol.npy")).astype(
                self.dtype
            )
        ).downsample(self.L)

        src = Simulation(L=self.L, n=n_img, vols=v, dtype=self.dtype)

        # Shift images using the Image method directly
        shifted_imgs = src.images[:n_img].shift(test_shift)

        # Convert original images to basis coefficients
        f_imgs = self.basis.evaluate_t(src.images[:n_img])

        # Use the basis shift method
        f_shifted_imgs = self.basis.shift(f_imgs, test_shift)

        # Compute diff between the shifted image sets
        diff = shifted_imgs.asnumpy() - self.basis.evaluate(f_shifted_imgs).asnumpy()

        # Compute mask to compare only the core of the shifted images
        g = grid_2d(self.L, indexing="yx", normalized=False)
        mask = g["r"] > self.L / 2
        # Masking values outside radius to 0
        diff = np.where(mask, 0, diff)

        # Compute and check error
        rmse = np.sqrt(np.mean(np.square(diff), axis=(1, 2)))
        logger.info(f"RMSE shifted image diffs {rmse}")
        self.assertTrue(np.allclose(rmse, 0, atol=1e-5))
