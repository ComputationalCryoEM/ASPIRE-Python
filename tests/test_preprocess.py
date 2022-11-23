import os.path
from unittest import TestCase

import numpy as np

from aspire.volume import Volume

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class PreprocessTestCase(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testVol2img(self):
        results = np.load(os.path.join(DATA_DIR, "clean70SRibosome_down8_imgs32.npy"))
        vols = Volume(np.load(os.path.join(DATA_DIR, "clean70SRibosome_vol_down8.npy")))
        rots = np.load(os.path.join(DATA_DIR, "rand_rot_matrices32.npy"))
        rots = np.moveaxis(rots, 2, 0)
        imgs_clean = vols.project(0, rots).asnumpy()
        self.assertTrue(np.allclose(results, imgs_clean, atol=1e-7))
