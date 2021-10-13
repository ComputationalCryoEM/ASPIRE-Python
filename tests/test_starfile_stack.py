import os
import os.path
from unittest import TestCase



import importlib_resources
import numpy as np

import tests.saved_test_data
from aspire.image import Image
from aspire.operators import ScalarFilter
from aspire.source.relion import RelionSource

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class StarFileTestCase(TestCase):

    def setUpStarFile(self, starfile_name, mrcs_name):
        """Overridden run method to use context manager provided by importlib_resources"""
        with importlib_resources.path(
            tests.saved_test_data, starfile_name
        ) as path:

            # Create a temporary file with the contents of the sample.mrcs file in a subfolder at the same location
            # as the starfile, to allow our classes to do their job
            temp_folder_path = os.path.join(path.parent.absolute(), "_temp")

            should_delete_folder = False
            if not os.path.exists(temp_folder_path):
                os.mkdir(temp_folder_path)
                should_delete_folder = True

            temp_file_path = os.path.join(temp_folder_path, mrcs_name)

            should_delete_file = False
            if not os.path.exists(temp_file_path):
                with open(temp_file_path, "wb") as f:
                    f.write(
                        importlib_resources.read_binary(
                            tests.saved_test_data, mrcs_name
                        )
                    )
                    should_delete_file = True
            should_delete_file_one_img = False
    
            self.src = RelionSource(path, data_folder=temp_folder_path, max_rows=12)

            #super(StarFileTestCase, self).run(starfile_name, mrcs_name, result)

            #if should_delete_file:
             #   os.remove(temp_file_path)
            #if should_delete_folder:
             #   os.removedirs(temp_folder_path)
    def setUp(self):
        starfile_name = 'sample_relion_data.star'
        mrcs_name = 'sample.mrcs'
        self.setUpStarFile(starfile_name, mrcs_name)
    def tearDown(self):
        pass

class StarFileMainCase(StarFileTestCase):
    #def setUp(self):
       # self.run('sample_relion_data.star', 'sample.mrcs', None)

    #def tearDown(self):
     #   pass

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
        self.setUpStarFile('sample_relion_one_image.star', 'sample_one_image.mrcs')
    
    def testMRCSWithOneParticle(self):
        # tests conversion of 2D numpy arrays into 3D stacks in the case
        # where there is only one image in the mrcs
        single_image = self.src.images(0, 1)[0]
        self.assertEqual(single_image.shape, (200, 200))
