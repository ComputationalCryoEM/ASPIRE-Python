import logging
import os.path
from unittest import TestCase

import numpy as np
from pytest import raises

from aspire.basis import FBBasis3D
from aspire.image import Image
from aspire.operators import IdentityFilter
from aspire.reconstruction import MeanEstimator
from aspire.source import ArrayImageSource, Simulation
from aspire.utils import utest_tolerance

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")

logger = logging.getLogger(__name__)


class ImageTestCase(TestCase):
    def setUp(self):
        self.dtype = np.float32
        self.resolution = 8

        n = 1024

        # Generate a stack of images
        self.sim = sim = Simulation(
            n=n,
            L=self.resolution,
            unique_filters=[IdentityFilter()],
            seed=0,
            dtype=self.dtype,
            # We'll use random angles
            offsets=np.zeros((n, 2)),  # No offsets
            amplitudes=np.ones((n)),  # Constant amplitudes
        )

        # Expose images as numpy array.
        self.ims_np = sim.images(0, sim.n).asnumpy()
        self.im = Image(self.ims_np)

        # Vol estimation requires a 3D basis
        self.basis = FBBasis3D((self.resolution,) * 3, dtype=self.dtype)

    def testArrayImageSource(self):
        """
        An Image can be wrapped in an ArrayImageSource when we need to deal with ImageSource objects.

        This checks round trip conversion does not crash and returns identity.
        """

        src = ArrayImageSource(self.im)
        im = src.images(start=0, num=np.inf)  # returns Image instance
        self.assertTrue(np.allclose(im.asnumpy(), self.ims_np))

    def testArrayImageSourceMeanVolError1(self):
        """
        Test that ArrayImageSource when instantiated without required rotations/angles gives an appropriate error.
        """

        # Construct the source for testing
        # Rotations(angles) are required, but we intentionally do not pass to instantiater here.
        src = ArrayImageSource(self.im)

        # Instatiate a volume estimator
        estimator = MeanEstimator(src, self.basis, preconditioner="none")

        # Test we raise with expected message
        with raises(RuntimeError, match=r"Consumer of ArrayImageSource.*"):
            _ = estimator.estimate()

    def testArrayImageSourceMeanVol(self):
        """
        Test that ArrayImageSource can be consumed by mean/volume codes.
        Checks that the estimate is consistent with Simulation source.
        """

        # Run estimator with a Simulation source as a reference.
        sim_estimator = MeanEstimator(self.sim, self.basis, preconditioner="none")
        sim_est = sim_estimator.estimate()
        logger.info("Simulation source checkpoint")

        # Construct the source for testing
        src = ArrayImageSource(self.im, angles=self.sim.angles)

        # Instatiate a volume estimator using ArrayImageSource
        estimator = MeanEstimator(src, self.basis, preconditioner="none")

        # Get estimate consuming ArrayImageSource
        est = estimator.estimate()

        delta = np.sqrt(np.mean(np.square(est - sim_est)))
        logger.info(f"Simulation vs ArrayImageSource estimates MRSE: {delta}")

        self.assertTrue(delta <= utest_tolerance(self.dtype))
        self.assertTrue(np.allclose(est, sim_est, atol=utest_tolerance(self.dtype)))
