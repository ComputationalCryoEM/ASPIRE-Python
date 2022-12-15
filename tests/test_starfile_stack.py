import os
import os.path
from unittest import TestCase

import mrcfile
import numpy as np

import tests.saved_test_data
from aspire.image import Image
from aspire.source.relion import RelionSource
from aspire.utils import importlib_path

from . import _copy_util

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class StarFileTestCase(TestCase):
    def setUpStarFile(self, starfile_name):
        # set up RelionSource object for tests
        with importlib_path(tests.saved_test_data, starfile_name) as starfile:
            self.src = RelionSource(starfile, data_folder=DATA_DIR, max_rows=12)

    def setUp(self):
        # this method is used by StarFileMainCase but overridden by StarFileOneImage
        self.setUpStarFile("sample_relion_data.star")

    def tearDown(self):
        pass


class StarFileMainCase(StarFileTestCase):
    def testImageStackType(self):
        # Since src is an ImageSource, we can call images() on it to get an Image
        image_stack = self.src.images[:]
        self.assertIsInstance(image_stack, Image)

    def testImageStackShape(self):
        # Load 10 images starting at index 0
        images = self.src.images[:10]
        self.assertEqual(images.shape, (10, 200, 200))

    def testImage0(self):
        image_stack = self.src.images[0]
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
        first_image = self.src.images[0][0]
        self.assertEqual(first_image.shape, (16, 16))

    def testRelionSourceCopy(self):
        src_copy = self.src.copy()
        # sanity check that ASPIRE objects that are attributes of the source
        # were deepcopied
        for var in _copy_util.source_vars:
            if hasattr(self.src, var):
                self.assertTrue(
                    (
                        _copy_util.source_vars_deepcopied(
                            getattr(self.src, var), getattr(src_copy, var), var
                        )
                    )
                )
        # make sure we can perform operations on both sources separately
        src_copy.downsample(8)
        img = self.src.images[:1]
        img_copy = src_copy.images[:1]
        self.assertEqual(img.resolution, 200)
        self.assertEqual(img_copy.resolution, 8)
        # copy should have an updated xform pipeline
        self.assertTrue(len(self.src.generation_pipeline.xforms) == 0)
        self.assertTrue(len(src_copy.generation_pipeline.xforms) == 1)
        # make sure metadata can be modified separately
        src_copy.set_metadata("test_col", 0)
        self.assertFalse(self.src.has_metadata("test_col"))


class StarFileSingleImage(StarFileTestCase):
    def setUp(self):
        # create new mrcs containing only one particle image
        with importlib_path(tests.saved_test_data, "sample.mrcs") as path:
            stack_path = str(path)
            new_mrcs_path = os.path.join(
                os.path.dirname(stack_path), "sample_one_image.mrcs"
            )
            mrcs_data = mrcfile.open(stack_path, "r").data[0]
            with mrcfile.new(new_mrcs_path) as new_mrcs:
                new_mrcs.set_data(mrcs_data)
        self.setUpStarFile("sample_relion_one_image.star")

    def tearDown(self):
        with importlib_path(tests.saved_test_data, "sample_one_image.mrcs") as path:
            os.remove(str(path))

    def testMRCSWithOneParticle(self):
        # tests conversion of 2D numpy arrays into 3D stacks in the case
        # where there is only one image in the mrcs
        single_image = self.src.images[0][0]
        self.assertEqual(single_image.shape, (200, 200))
