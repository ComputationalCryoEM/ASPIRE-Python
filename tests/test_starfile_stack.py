from unittest import TestCase
import numpy as np
import importlib_resources
import aspire.data
from aspire.source.relion import RelionStarfileStack
from aspire.image import Image
from aspire.utils.filters import ScalarFilter

import os.path
DATA_DIR = os.path.join(os.path.dirname(__file__), 'saved_test_data')


class StarfileTestCase(TestCase):
    def run(self, result=None):
        """Overridden run method to use context manager provided by importlib_resources"""
        with importlib_resources.path(aspire.data, 'sample.star') as path:

            # TODO: This test only works because the mrcs files referenced by the starfile are actually
            # found at the same location as the starfile (due to zip_safe=False in setup.py)
            # A more advanced approach with mock/patch is needed for this to work in a general case.
            # Once this test is fixed, zip_safe can be set to True in setup.py

            self.src = RelionStarfileStack(path, ignore_missing_files=True)
            super(StarfileTestCase, self).run(result)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testImageStackType(self):
        # Since src is an ImageSource, we can call images() on it to get an ImageStack
        image_stack = self.src.images()
        self.assertIsInstance(image_stack, Image)

    def testImageStackShape(self):
        # Note that sample data only includes a single .mrcs file for the first 17 images
        # Load 10 images starting at index 0
        images = self.src.images(0, 10)
        self.assertEqual(images.shape, (200, 200, 10))

    def testImage0(self):
        image_stack = self.src.images(0, 1)
        first_image = image_stack[:, :, 0]
        self.assertTrue(np.allclose(
            first_image,
            np.load(os.path.join(DATA_DIR, 'starfile_image_0.npy'))
        ))

    def testMetadata(self):
        # The 'df' attribute of the StarfileStack object is a Pandas Dataframe
        # that contains relevant metadata for the individual .mrcs file(s)
        self.assertAlmostEqual(3073.912046, self.src.df.iloc[0].rlnCoordinateY)

    def testImageDownsample(self):
        self.src.set_max_resolution(16)
        first_image = self.src.images(0, 1)[:, :, 0]
        self.assertEqual(first_image.shape, (16, 16))

    def testImageDownsampleAndWhiten(self):
        self.src.set_max_resolution(16)
        self.src.whiten(whiten_filter=ScalarFilter(dim=2, value=0.02450909546680349, power=-0.5))
        first_whitened_image = self.src.images(0, 1)[:, :, 0]

        self.assertTrue(np.allclose(
            first_whitened_image,
            np.load(os.path.join(DATA_DIR, 'starfile_image_0_whitened.npy')),
            atol=1e-6
        ))
