from unittest import TestCase
import importlib_resources
import aspire.data
from aspire.io.micrograph import Micrograph


class MicrographTestCase(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testShape1(self):
        # Load a single micrograph and check it's shape
        with importlib_resources.path(aspire.data, 'sample.mrc') as path:
            micrograph = Micrograph(path, margin=100, shrink_factor=2)

        # Original Image = 4096 x 4096 -> remove 100px margins -> 3896 x 3896 -> shrink by 2 -> 1948 x 1948
        self.assertEqual(micrograph.im.shape, (1948, 1948))

    def testShape2(self):
        # Load a MRCS stack and check it's shape
        with importlib_resources.path(aspire.data, 'sample.mrcs') as path:
            micrograph = Micrograph(path)

        # The first 2 dimensions are the shape of each image, the last dimension the no. of images
        self.assertEqual(micrograph.im.shape, (200, 200, 267))
