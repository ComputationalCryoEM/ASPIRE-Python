import logging
import os
import tempfile
from shutil import copyfile
from unittest import TestCase

import numpy as np

from aspire.ctf import estimate_ctf

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class CtfEstimatorTestCase(TestCase):
    def setUp(self):
        self.test_input_fn = "sample.mrc"
        self.test_output = [
            [
                1.1142363760e03,
                1.0920983202e03,
                -8.3521800000e-03,
                2.0,
                300.0,
                1,
                0.07,
                self.test_input_fn,
            ]
        ]

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
                )

                logger.debug(f"results: {results}")

                for i, result in enumerate(results):
                    diffs = np.subtract(result[:-1], self.test_output[i][:-1])
                    logger.debug(f"diffs: {diffs}")

                    # defocusU
                    np.allclose(result[0], self.test_output[i][0], atol=5e-2)
                    # defocusV
                    np.allclose(result[1], self.test_output[i][1], atol=5e-2)
                    # defocusAngle
                    np.allclose(result[2], self.test_output[i][2], atol=5e-5)

                    self.assertTrue(
                        np.allclose(result[3:-1], self.test_output[i][3:-1])
                    )
                    self.assertTrue(result[-1] == self.test_output[i][-1])
