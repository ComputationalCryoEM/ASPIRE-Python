import os
import tempfile
from shutil import copyfile
from unittest import TestCase

import numpy as np

from aspire.ctf import estimate_ctf

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class CtfEstimatorTestCase(TestCase):
    def setUp(self):
        self.test_input_fn = "sample.mrc"
        self.test_output = [
            [
                1114.249878341215,
                1092.097776049334,
                -0.4798126410114381,
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
                    corr=1.0,
                    output_dir=tmp_output_dir,
                    dtype=np.float64,
                )

                print(results)

                for i, result in enumerate(results):
                    diffs = np.subtract(result[:-1], self.test_output[i][:-1])
                    print(f"diffs:", diffs)

                    self.assertTrue(np.allclose(result[:-1], self.test_output[i][:-1]))
                    self.assertEqual(result[-1], self.test_output[i][-1])
