from unittest import TestCase
from aspyre.image import ImageStack
from aspyre.source.mrcstack import MrcStack

import os.path
MRCS_FILE = os.path.join(os.path.dirname(__file__), 'saved_test_data', 'mrc_files', 'stack_0500_cor_DW.mrcs')


class MicrographTestCase(TestCase):
    def setUp(self):
        self.mrc_stack = MrcStack(MRCS_FILE)

    def tearDown(self):
        pass

    def testImageStackType(self):
        # Since mrc_stack is an ImageSource, we can call images() on it to get an ImageStack
        image_stack = self.mrc_stack.images()
        self.assertIsInstance(image_stack, ImageStack)

    def testImageStackShape(self):
        # Try to get a total of 10 images from our ImageSource
        image_stack = self.mrc_stack.images(num=10)
        # The shape of the resulting ImageStack is 200 (height) x 200 (width) x 10 (n_images)
        self.assertEqual(image_stack.shape, (200, 200, 10))
