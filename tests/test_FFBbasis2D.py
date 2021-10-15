import logging
import os.path
from unittest import TestCase

import numpy as np

from aspire.basis import FFBBasis2D
from aspire.image import Image
from aspire.source import Simulation
from aspire.utils import utest_tolerance
from aspire.utils.misc import grid_2d
from aspire.volume import Volume

logger = logging.getLogger(__name__)
DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class FFBBasis2DTestCase(TestCase):
    def setUp(self):
        self.dtype = np.float32  # Required for convergence of this test
        self.L = 8
        self.basis = FFBBasis2D((self.L, self.L), dtype=self.dtype)

    def tearDown(self):
        pass

    def testFFBBasis2DIndices(self):
        indices = self.basis.indices()

        self.assertTrue(
            np.allclose(
                indices["ells"],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    4.0,
                    4.0,
                    4.0,
                    4.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    6.0,
                    6.0,
                    7.0,
                    7.0,
                    8.0,
                    8.0,
                ],
            )
        )

        self.assertTrue(
            np.allclose(
                indices["ks"],
                [
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                    0.0,
                    1.0,
                    2.0,
                    0.0,
                    1.0,
                    2.0,
                    0.0,
                    1.0,
                    2.0,
                    0.0,
                    1.0,
                    2.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            )
        )

        self.assertTrue(
            np.allclose(
                indices["sgns"],
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    1.0,
                    1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    -1.0,
                    1.0,
                    -1.0,
                    1.0,
                    -1.0,
                ],
            )
        )

    def testFFBBasis2DNorms(self):
        radial_norms, angular_norms = self.basis.norms()
        self.assertTrue(
            np.allclose(
                radial_norms * angular_norms,
                [
                    3.68065992303471,
                    2.41241466684800,
                    1.92454669738088,
                    1.64809729313301,
                    2.01913617828263,
                    1.50455726188833,
                    1.25183461029289,
                    1.70284654929000,
                    1.36051054373844,
                    1.16529703804363,
                    1.49532071137207,
                    1.25039038364830,
                    1.34537533748304,
                    1.16245357319190,
                    1.23042467443861,
                    1.09002083501080,
                    1.13867113286781,
                    1.06324777330476,
                    0.999841586390824,
                ],
            )
        )

    def testFFBBasis2DEvaluate(self):
        v = np.array(
            [
                1.07338590e-01,
                1.23690941e-01,
                6.44482039e-03,
                -5.40484306e-02,
                -4.85304586e-02,
                1.09852144e-02,
                3.87838396e-02,
                3.43796455e-02,
                -6.43284705e-03,
                -2.86677145e-02,
                -1.42313328e-02,
                -2.25684091e-03,
                -3.31840727e-02,
                -2.59706174e-03,
                -5.91919887e-04,
                -9.97433028e-03,
                9.19123928e-04,
                1.19891589e-03,
                7.49154982e-03,
                6.18865229e-03,
                -8.13265715e-04,
                -1.30715655e-02,
                -1.44160603e-02,
                2.90379956e-03,
                2.37066082e-02,
                4.88805735e-03,
                1.47870707e-03,
                7.63376018e-03,
                -5.60619559e-03,
                1.05165081e-02,
                3.30510143e-03,
                -3.48652120e-03,
                -4.23228797e-04,
                1.40484061e-02,
            ],
            dtype=self.dtype,
        )
        result = self.basis.evaluate(v)

        self.assertTrue(
            np.allclose(
                result.asnumpy(),  # Result of evaluate is an Image after RCOPT
                np.load(
                    os.path.join(DATA_DIR, "ffbbasis2d_xcoeff_out_8_8.npy")
                ).T,  # RCOPT
                atol=utest_tolerance(self.dtype),
            )
        )

    def testFFBBasis2DEvaluate_t(self):
        x = np.load(os.path.join(DATA_DIR, "ffbbasis2d_xcoeff_in_8_8.npy")).T  # RCOPT
        result = self.basis.evaluate_t(x.astype(self.dtype))

        self.assertTrue(
            np.allclose(
                result,
                np.load(os.path.join(DATA_DIR, "ffbbasis2d_vcoeff_out_8_8.npy"))[
                    ..., 0
                ],
            )
        )

    def testFFBBasis2DExpand(self):
        x = np.load(os.path.join(DATA_DIR, "ffbbasis2d_xcoeff_in_8_8.npy")).T  # RCOPT
        result = self.basis.expand(x.astype(self.dtype))
        self.assertTrue(
            np.allclose(
                result,
                np.load(os.path.join(DATA_DIR, "ffbbasis2d_vcoeff_out_exp_8_8.npy"))[
                    ..., 0
                ],
                atol=utest_tolerance(self.dtype),
            )
        )

    def testRotate(self):
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
        x1 = src.images(0, 1)

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
                np.float64
            )
        ).downsample(self.L)

        src = Simulation(L=self.L, n=n_img, vols=v, dtype=np.float64)

        # Shift images using the Image method directly
        shifted_imgs = src.images(0, n_img).shift(test_shift)

        # Convert original images to basis coefficients
        f_imgs = self.basis.evaluate_t(src.images(0, n_img))

        # Use the basis shift method
        f_shifted_imgs = self.basis.shift(f_imgs, test_shift)

        # Compute diff between the shifted image sets
        diff = shifted_imgs.asnumpy() - self.basis.evaluate(f_shifted_imgs).asnumpy()

        # Compute mask to compare only the core of the shifted images
        g = grid_2d(self.L, normalized=False)
        mask = g["r"] > self.L / 2
        # Masking values outside radius to 0
        diff = np.where(mask, 0, diff)

        # Compute and check error
        rmse = np.sqrt(np.mean(np.square(diff), axis=(1, 2)))
        logger.info(f"RMSE shifted image diffs {rmse}")
        self.assertTrue(np.allclose(rmse, 0, atol=1e-5))
