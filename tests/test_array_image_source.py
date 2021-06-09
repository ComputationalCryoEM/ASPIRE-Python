import os.path
from unittest import TestCase

import numpy as np

from aspire.basis import FBBasis3D
from aspire.image import Image
from aspire.reconstruction import MeanEstimator
from aspire.source import ArrayImageSource, Simulation

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class ImageTestCase(TestCase):
    def setUp(self):
        self.dtype = np.float32

        # Generate a stack of images
        sim = Simulation(
            n=1024,
            L=64,
            seed=0,
            dtype=self.dtype,
        )

        # Expose images as numpy array.
        self.ims_np = sim.images(0, sim.n).asnumpy()
        self.im = Image(self.ims_np)

    def testArrayImageSource(self):
        """
        An Image can be wrapped in an ArrayImageSource when we need to deal with ImageSource objects.

        This checks round trip conversion does not crash and returns identity.
        """

        src = ArrayImageSource(self.im)
        im = src.images(start=0, num=np.inf)  # returns Image instance
        self.assertTrue(np.allclose(im.asnumpy(), self.ims_np))

    def testArrayImageSourceMeanVol(self):
        """
        Test that ArrayImageSource can be consumed by mean/volume codes.
        """

        # Construct the source for testing
        src = ArrayImageSource(self.im)

        # Vol estimation requires a 3D basis
        basis = FBBasis3D((8, 8, 8), dtype=self.dtype)

        # Instatiate a volume estimator
        estimator = MeanEstimator(src, basis, preconditioner="none")

        # Test we do not crash.
        _ = estimator.estimate()
