from unittest import TestCase
import numpy as np

from aspyre.source.star import Starfile
from aspyre.imaging.filters import ScalarFilter

import os.path
DATA_DIR = os.path.join(os.path.dirname(__file__), 'saved_test_data')


class StarfileTestCase(TestCase):
    def setUp(self):
        self.src = Starfile(os.path.join(DATA_DIR, 'starfile.star'), pixel_size=1.338, ignore_missing_files=True)

    def tearDown(self):
        pass

    def testImageShape(self):
        # Note that the test folder only includes the .mrcs file for the very first image
        # so we restrict attention to just the first image
        # Load 1 image starting at index 0
        first_image = self.src.images(0, 1)
        # We get a 3d array back, regardless of how many images we're reading (1 in this case)
        self.assertEqual(first_image.shape, (200, 200, 1))

    def testImage(self):
        first_image = self.src.images(0, 1)
        first_image = np.squeeze(first_image)  # Convert to 2D
        self.assertTrue(np.allclose(
            first_image,
            np.load(os.path.join(DATA_DIR, 'starfile_image_0.npy'))
        ))

    def testImageDownsample(self):
        self.src.set_max_resolution(16)
        first_image = self.src.images(0, 1).squeeze()
        self.assertEqual(first_image.shape, (16, 16))

    def testImageDownsampleAndWhiten(self):
        self.src.set_max_resolution(16)
        self.src.whiten(whiten_filter=ScalarFilter(dim=2, value=0.02450909546680349, power=-0.5))
        first_whitened_image = self.src.images(
            start=0,
            num=1
        ).squeeze()

        self.assertTrue(np.allclose(
            first_whitened_image,
            np.load(os.path.join(DATA_DIR, 'starfile_image_0_whitened.npy')),
            atol=1e-6
        ))
