from unittest import TestCase

import importlib_resources
import numpy as np

import tests.saved_test_data
from aspire.image import Image
from aspire.source.mrcstack import MrcStack


class MicrographTestCase(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testImageStackType(self):
        with importlib_resources.path(tests.saved_test_data, "sample.mrcs") as path:
            mrc_stack = MrcStack(path)
            # Since mrc_stack is an ImageSource, we can call images() on it to get an ImageStack
            image_stack = mrc_stack.images(start=0, num=np.inf)
            self.assertIsInstance(image_stack, Image)

    def testImageStackShape(self):
        with importlib_resources.path(tests.saved_test_data, "sample.mrcs") as path:
            mrc_stack = MrcStack(path)
            # Try to get a total of 5 images from our ImageSource
            image_stack = mrc_stack.images(start=0, num=5)
            # The shape of the resulting ImageStack is 200 (height) x 200 (width) x 5 (n_images)
            self.assertEqual(image_stack.shape, (5, 200, 200))
