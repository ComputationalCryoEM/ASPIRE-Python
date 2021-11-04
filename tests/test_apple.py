import os
import pickle
import tempfile
from shutil import copyfile
from unittest import TestCase

import importlib_resources
import mrcfile

import tests.saved_test_data
from aspire.apple.apple import Apple


class ApplePickerTestCase(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testPickCenters(self):
        # 440 particles with the following centers
        with importlib_resources.path(
            tests.saved_test_data, "apple_centers.p"
        ) as centers_path:
            centers = set(pickle.load(open(str(centers_path), "rb")))
        with tempfile.TemporaryDirectory() as tmpdir_name:
            apple_picker = Apple(output_dir=tmpdir_name)
            with importlib_resources.path(
                tests.saved_test_data, "sample.mrc"
            ) as mrc_path:
                centers_found = apple_picker.process_micrograph(
                    mrc_path, create_jpg=True
                )
                for center_found in centers_found:
                    _x, _y = tuple(center_found)
                    if (_x, _y) not in centers:
                        self.fail("({}, {}) not an expected center.".format(_x, _y))
                    else:
                        centers.remove((_x, _y))

                if centers:
                    self.fail("Not all expected centers were found!")

    def testFileCorruption(self):
        """
        Test that corrupt mrcfiles are logged as expected.
        """

        # Create a tmp dir for this test output
        with tempfile.TemporaryDirectory() as tmpdir_name:

            # Instantiate an Apple instance
            apple_picker = Apple(output_dir=tmpdir_name)

            # Get the path of an input mrcfile
            with importlib_resources.path(
                tests.saved_test_data, "sample.mrc"
            ) as good_mrc_path:
                # Store bad mrc in tmp test dir so it gets cleaned up
                bad_mrc_path = os.path.join(tmpdir_name, "bad.mrc")

                # Copy mrc file
                copyfile(good_mrc_path, bad_mrc_path)

            # Open mrc file and soft corrupt it
            with mrcfile.open(bad_mrc_path, "r+") as fh:
                fh.header.map = -1

            # Check that we get a WARNING
            with self.assertLogs(level="WARNING") as logs:
                _ = apple_picker.process_micrograph(bad_mrc_path, create_jpg=False)

            # Check the message prefix
            self.assertTrue(
                "APPLE.picking mrcfile reporting 1 corruptions" in logs.output[0]
            )

            # Check the message contains the file path
            self.assertTrue(bad_mrc_path in logs.output[0])
