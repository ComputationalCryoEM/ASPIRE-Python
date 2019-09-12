from unittest import TestCase
import numpy as np
import importlib_resources
import os

import aspire.data
from aspire.source.relion import RelionSource
from aspire.image import Image
from aspire.utils.filters import ScalarFilter

import os.path
DATA_DIR = os.path.join(os.path.dirname(__file__), 'saved_test_data')


class StarFileTestCase(TestCase):
    def run(self, result=None):
        """Overridden run method to use context manager provided by importlib_resources"""
        with importlib_resources.path(aspire.data, 'sample_relion_data.star') as path:
            # Create a temporary file with the contents of the sample.mrcs file at the same location as the starfile,
            # to allow our classes to do their job
            temp_file_path = os.path.join(path.parent.absolute(), 'sample.mrcs')

            should_delete = False
            if not os.path.exists(temp_file_path):
                with open(temp_file_path, 'wb') as f:
                    f.write(importlib_resources.read_binary(aspire.data, 'sample.mrcs'))
                should_delete = True

            self.src = RelionSource(path, max_rows=12)
            super(StarFileTestCase, self).run(result)

            if should_delete:
                os.remove(temp_file_path)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testImageStackType(self):
        # Since src is an ImageSource, we can call images() on it to get an Image
        image_stack = self.src.images()
        self.assertIsInstance(image_stack, Image)

    def testImageStackShape(self):
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
        # The 'get_metadata' method of the StarFileStack object can be used to get metadata information
        # for a particular image index. Here we get the '_rlnCoordinateY' attribute of the first image.
        self.assertAlmostEqual(3073.912046, self.src.get_metadata('_rlnCoordinateY', [0])[0])

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
