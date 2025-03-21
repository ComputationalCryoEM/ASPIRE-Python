import glob
import logging
import os.path
import tempfile
from unittest import TestCase

import numpy as np
import pytest
from pytest import raises

from aspire.basis import FBBasis3D
from aspire.image import Image
from aspire.operators import IdentityFilter
from aspire.reconstruction import MeanEstimator
from aspire.source import ArrayImageSource, RelionSource, Simulation
from aspire.utils import Rotation, utest_tolerance

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")

logger = logging.getLogger(__name__)


class ImageTestCase(TestCase):
    def setUp(self):
        self.dtype = np.float32
        self.resolution = 8

        self.n = 1024

        # Generate a stack of images
        self.sim = sim = Simulation(
            n=self.n,
            L=self.resolution,
            unique_filters=[IdentityFilter()],
            seed=0,
            dtype=self.dtype,
            # We'll use random angles
            offsets=np.zeros((self.n, 2)),  # No offsets
            amplitudes=np.ones((self.n)),  # Constant amplitudes
        )

        # Expose images as numpy array.
        self.ims_np = sim.images[:].asnumpy()
        self.im = Image(self.ims_np)

        # Vol estimation requires a 3D basis
        self.basis = FBBasis3D((self.resolution,) * 3, dtype=self.dtype)

    def testArrayImageSource(self):
        """
        An Image can be wrapped in an ArrayImageSource when we need to deal with ImageSource objects.

        This checks round trip conversion does not crash and returns identity.
        """

        src = ArrayImageSource(self.im)
        im = src.images[:]  # returns Image instance
        self.assertTrue(np.allclose(im.asnumpy(), self.ims_np))

    def testArrayImageSourceFromNumpy(self):
        """
        An Array can be wrapped in an ArrayImageSource when we need to deal with ImageSource objects.

        This checks round trip conversion does not crash and returns identity.
        """

        # Create an ArrayImageSource directly from Numpy array
        src = ArrayImageSource(self.ims_np)

        # Ask the Source for all images in the stack as a Numpy array
        ims_np = src.images[:].asnumpy()

        # Comparison should be yield identity
        self.assertTrue(np.allclose(ims_np, self.ims_np))

    def testArrayImageSourceNumpyError(self):
        """
        Test that ArrayImageSource when instantiated with incorrect input
        gives appropriate error.
        """

        # Test we raise with expected message from getter.
        with raises(RuntimeError, match=r"Creating Image object from Numpy.*"):
            _ = ArrayImageSource(np.empty((3, 2, 1)))

    def testArrayImageSourceAngGetterError(self):
        """
        Test that ArrayImageSource when instantiated without required
        rotations/angles gives an appropriate error.
        """

        # Construct the source for testing.
        #   Rotations (via angles) are required,
        #   but we intentionally do not pass
        #   to instantiater here.
        src = ArrayImageSource(self.im)

        # Test we raise with expected message
        with raises(RuntimeError, match=r"Consumer of ArrayImageSource.*"):
            _ = src.angles

        # We also test that a source consumer generates same error,
        #   by instantiating a volume estimator.
        estimator = MeanEstimator(src, basis=self.basis, preconditioner="none")

        # Test we raise with expected message
        with raises(RuntimeError, match=r"Consumer of ArrayImageSource.*"):
            _ = estimator.estimate()

    def testArrayImageSourceRotGetterError(self):
        """
        Test that ArrayImageSource when instantiated without required
        rotations/angles gives an appropriate error.
        Here we specifically test `rotations`.
        """

        # Construct the source for testing.
        src = ArrayImageSource(self.im)

        # Test we raise with expected message from getter.
        with raises(RuntimeError, match=r"Consumer of ArrayImageSource.*"):
            _ = src.rotations

    def testArrayImageSourceMeanVol(self):
        """
        Test that ArrayImageSource can be consumed by mean/volume codes.
        Checks that the estimate is consistent with Simulation source.
        """

        # Run estimator with a Simulation source as a reference.
        sim_estimator = MeanEstimator(self.sim, basis=self.basis, preconditioner="none")
        sim_est = sim_estimator.estimate()
        logger.info("Simulation source checkpoint")

        # Construct the source for testing
        src = ArrayImageSource(self.im, angles=self.sim.angles)

        # Instantiate a volume estimator using ArrayImageSource
        estimator = MeanEstimator(src, basis=self.basis, preconditioner="none")

        # Get estimate consuming ArrayImageSource
        est = estimator.estimate()

        # Compute RMS error and log it for debugging.
        delta = np.sqrt(np.mean(np.square(est - sim_est)))
        logger.info(f"Simulation vs ArrayImageSource estimates MRSE: {delta}")

        # Estimate RMSE should be small.
        # Loosened by factor of two to accomodate fast math on OSX CI.
        self.assertTrue(delta <= 2 * utest_tolerance(self.dtype))
        # And the estimate themselves should be close (virtually same inputs).
        #  We should be within same neighborhood as generating sim_est multiple times...
        self.assertTrue(
            np.allclose(est, sim_est, atol=10 * utest_tolerance(self.dtype))
        )

    def testArrayImageSourceRotsSetGet(self):
        """
        Test ArrayImageSource `rotations` property, setter and getter function.
        """

        # Construct the source for testing
        src = ArrayImageSource(self.im, angles=self.sim.angles)

        # Get some random angles, can use from sim.
        angles = self.sim.angles
        self.assertTrue(angles.shape == (src.n, 3))

        # Convert to rotation matrix (n,3,3)
        rotations = Rotation.from_euler(angles).matrices
        self.assertTrue(rotations.shape == (src.n, 3, 3))

        # Excercise the setter
        src.rotations = rotations

        # Test Rotations Getter
        self.assertTrue(
            np.allclose(rotations, src.rotations, atol=utest_tolerance(self.dtype))
        )

        # Test Angles Getter
        self.assertTrue(
            np.allclose(angles, src.angles, atol=utest_tolerance(self.dtype))
        )

    def testArrayImageSourceAnglesShape(self):
        """
        Test ArrayImageSource `angles` argument shapes.
        """

        # Construct the source with correct shape.
        _ = ArrayImageSource(self.im, angles=self.sim.angles)

        # Should match this error message.
        msg = r"Angles should be shape.*"

        # Construct the source with wrong shape.
        wrong_width = np.random.randn(self.n, 2)
        with raises(ValueError, match=msg):
            _ = ArrayImageSource(self.im, angles=wrong_width)

        wrong_dim = np.random.randn(self.n, 3, 3)
        with raises(ValueError, match=msg):
            _ = ArrayImageSource(self.im, angles=wrong_dim)

    def test_save_mrc(self):
        """
        Test saving single batch mode (.mrc).
        """

        src = ArrayImageSource(self.im)

        with tempfile.TemporaryDirectory() as tmp_dir:
            star_path = os.path.join(tmp_dir, "test.star")
            src.save(star_path, batch_size=1)

            # Test the filenames are as expected
            self.assertTrue(len(glob.glob(os.path.join(tmp_dir, "*.mrc"))) == self.n)
            self.assertTrue(len(glob.glob(os.path.join(tmp_dir, "*.mrcs"))) == 0)

            # Test the content matched when loaded
            src2 = RelionSource(star_path)
            np.testing.assert_allclose(src.images[:], src2.images[:])

    def test_save_mrcs(self):
        """
        Test saving stack batch mode (.mrcs).
        """
        src = ArrayImageSource(self.im)
        batch_size = 256

        with tempfile.TemporaryDirectory() as tmp_dir:
            star_path = os.path.join(tmp_dir, "test.star")
            src.save(star_path, batch_size=256)

            # Test the filenames are as expected
            self.assertTrue(len(glob.glob(os.path.join(tmp_dir, "*.mrc"))) == 0)
            self.assertTrue(
                len(glob.glob(os.path.join(tmp_dir, "*.mrcs")))
                == (self.n + batch_size - 1) // batch_size
            )

            # Test the content matched when loaded
            src2 = RelionSource(star_path)
            np.testing.assert_allclose(src.images[:], src2.images[:])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dtype_passthrough(dtype):
    """
    Test dtypes are passed appropriately to metadata.
    """
    n_ims = 10
    res = 32
    ims = np.ones((n_ims, res, res), dtype=dtype)

    src = ArrayImageSource(ims)

    # Check dtypes
    np.testing.assert_equal(src.dtype, dtype)
    np.testing.assert_equal(src.images[:].dtype, dtype)
    np.testing.assert_equal(src.amplitudes.dtype, dtype)
    np.testing.assert_equal(src.offsets.dtype, dtype)
