import logging
import os
import tempfile
from shutil import copyfile
from unittest import TestCase

import mrcfile
import numpy as np
from parameterized import parameterized

from aspire.ctf import estimate_ctf

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class CtfEstimatorTestCase(TestCase):
    def setUp(self):
        self.test_input_fn = "sample.mrc"
        self.test_output = {
            "defocus_u": 1.137359876e03,
            "defocus_v": 9.617226108e02,
            "defocus_ang": 1.5706205116381249,
            "cs": 2.0,
            "voltage": 300.0,
            "pixel_size": 1,
            "amplitude_contrast": 0.07,
        }

    def tearDown(self):
        pass

    def testEstimateCTF(self):
        with tempfile.TemporaryDirectory() as tmp_input_dir:
            # Copy input file
            copyfile(
                os.path.join(DATA_DIR, self.test_input_fn),
                os.path.join(tmp_input_dir, self.test_input_fn),
            )

            with tempfile.TemporaryDirectory() as tmp_output_dir:
                # Returns results in output_dir
                results = estimate_ctf(
                    data_folder=tmp_input_dir,
                    pixel_size=1,
                    cs=2.0,
                    amplitude_contrast=0.07,
                    voltage=300.0,
                    num_tapers=2,
                    psd_size=512,
                    g_min=30.0,
                    g_max=5.0,
                    output_dir=tmp_output_dir,
                    dtype=np.float64,
                    save_ctf_images=True,
                    save_noise_images=True,
                )

                logger.debug(f"results: {results}")

                for result in results.values():
                    # the following parameters have higher tolerances

                    # defocusU
                    self.assertTrue(
                        np.allclose(
                            result["defocus_u"],
                            self.test_output["defocus_u"],
                            atol=5e-2,
                        )
                    )
                    # defocusV
                    self.assertTrue(
                        np.allclose(
                            result["defocus_u"],
                            self.test_output["defocus_u"],
                            atol=5e-2,
                        )
                    )
                    # defocusAngle
                    self.assertTrue(
                        np.allclose(
                            result["defocus_ang"],
                            self.test_output["defocus_ang"],
                            atol=5e-2,
                        )
                    )

                    for param in ["cs", "amplitude_contrast", "voltage", "pixel_size"]:
                        self.assertTrue(
                            np.allclose(result[param], self.test_output[param])
                        )

    # we are chopping the micrograph into a vertical and a horizontal rectangle
    # as small as possible to save testing duration
    @parameterized.expand(
        [[(slice(0, 128), slice(0, 64))], [(slice(0, 64), slice(0, 128))]]
    )
    def testRectangularMicrograph(self, slice_range):
        with tempfile.TemporaryDirectory() as tmp_input_dir:
            # copy input file
            copyfile(
                os.path.join(DATA_DIR, self.test_input_fn),
                os.path.join(tmp_input_dir, "rect_" + self.test_input_fn),
            )
            # trim the file into a rectangle
            with mrcfile.open(
                os.path.join(tmp_input_dir, "rect_" + self.test_input_fn), "r+"
            ) as mrc_in:
                data = mrc_in.data[slice_range]
                mrc_in.set_data(data)
            # make sure we can estimate with no errors
            _ = estimate_ctf(data_folder=tmp_input_dir, psd_size=64)
