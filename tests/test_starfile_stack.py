import os
import os.path
from unittest import TestCase

import importlib_resources
import mrcfile
import numpy as np

import tests.saved_test_data
from aspire.image import Image
from aspire.operators import ScalarFilter
from aspire.source.relion import RelionSource

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class StarFileTestCase(TestCase):
    def setUpStarFile(self, starfile_name):
        # set up RelionSource object for tests
        with importlib_resources.path(tests.saved_test_data, starfile_name) as starfile:
            self.src = RelionSource(starfile, data_folder=DATA_DIR, max_rows=12)

    def setUp(self):
        # this method is used by StarFileMainCase but overridden by StarFileOneImage
        self.setUpStarFile("sample_relion_data.star")

    def tearDown(self):
        pass


class StarFileMainCase(StarFileTestCase):
    def testImageStackType(self):
        # Since src is an ImageSource, we can call images() on it to get an Image
        image_stack = self.src.images(start=0, num=np.inf)
        self.assertIsInstance(image_stack, Image)

    def testImageStackShape(self):
        # Load 10 images starting at index 0
        images = self.src.images(0, 10)
        self.assertEqual(images.shape, (10, 200, 200))

    def testImage0(self):
        image_stack = self.src.images(0, 1)
        first_image = image_stack[0]
        self.assertTrue(
            np.allclose(
                first_image,
                np.load(os.path.join(DATA_DIR, "starfile_image_0.npy")).T,  # RCOPT
            )
        )

    def testMetadata(self):
        # The 'get_metadata' method of the StarFileStack object can be used to get metadata information
        # for a particular image index. Here we get the '_rlnCoordinateY' attribute of the first image.
        self.assertAlmostEqual(
            3073.912046, self.src.get_metadata("_rlnCoordinateY", [0])
        )

    def testImageDownsample(self):
        self.src.downsample(16)
        first_image = self.src.images(0, 1)[0]
        self.assertEqual(first_image.shape, (16, 16))

    def testImageDownsampleAndWhiten(self):
        self.src.downsample(16)
        self.src.whiten(noise_filter=ScalarFilter(dim=2, value=0.02450909546680349))
        first_whitened_image = self.src.images(0, 1)[0]
        self.assertTrue(
            np.allclose(
                first_whitened_image,
                np.load(
                    os.path.join(DATA_DIR, "starfile_image_0_whitened.npy")
                ).T,  # RCOPT
                atol=1e-6,
            )
        )


class StarFileSingleImage(StarFileTestCase):
    def setUp(self):
        # create new mrcs containing only one particle image
        with importlib_resources.path(tests.saved_test_data, "sample.mrcs") as path:
            stack_path = str(path)
            new_mrcs_path = os.path.join(
                os.path.dirname(stack_path), "sample_one_image.mrcs"
            )
            mrcs_data = mrcfile.open(stack_path, "r").data[0]
            with mrcfile.new(new_mrcs_path) as new_mrcs:
                new_mrcs.set_data(mrcs_data)
        self.setUpStarFile("sample_relion_one_image.star")

    def tearDown(self):
        with importlib_resources.path(
            tests.saved_test_data, "sample_one_image.mrcs"
        ) as path:
            os.remove(str(path))

    def testMRCSWithOneParticle(self):
        # tests conversion of 2D numpy arrays into 3D stacks in the case
        # where there is only one image in the mrcs
        single_image = self.src.images(0, 1)[0]
        self.assertEqual(single_image.shape, (200, 200))
