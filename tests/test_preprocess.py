import os.path
from unittest import TestCase

import numpy as np
from scipy.fftpack import fftn, fftshift

from aspire.image import crop_pad, downsample, fuzzy_mask
from aspire.volume import Volume

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class PreprocessTestCase(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test01CropPad(self):
        results = np.load(os.path.join(DATA_DIR, "clean70SRibosome_vol_crop8.npy"))
        vols = np.load(os.path.join(DATA_DIR, "clean70SRibosome_vol.npy"))
        vols = vols[..., np.newaxis]
        vols_f = crop_pad(fftshift(fftn(vols[:, :, :, 0])), 8)
        self.assertTrue(np.allclose(results, vols_f, atol=1e-7))

    def test02Downsample(self):
        results = np.load(os.path.join(DATA_DIR, "clean70SRibosome_vol_down8.npy"))
        results = results[np.newaxis, ...]
        vols = np.load(os.path.join(DATA_DIR, "clean70SRibosome_vol.npy"))
        vols = vols[np.newaxis, ...]
        vols = downsample(vols, (8, 8, 8))
        self.assertTrue(np.allclose(results, vols, atol=1e-7))

    def test03Vol2img(self):
        results = np.load(os.path.join(DATA_DIR, "clean70SRibosome_down8_imgs32.npy"))
        vols = Volume(np.load(os.path.join(DATA_DIR, "clean70SRibosome_vol_down8.npy")))
        rots = np.load(os.path.join(DATA_DIR, "rand_rot_matrices32.npy"))
        rots = np.moveaxis(rots, 2, 0)
        imgs_clean = vols.project(0, rots).asnumpy()
        self.assertTrue(np.allclose(results, imgs_clean, atol=1e-7))

    def test04FuzzyMask(self):
        results = np.array(
            [
                [
                    2.03406033e-06,
                    7.83534653e-05,
                    9.19567967e-04,
                    3.73368194e-03,
                    5.86559882e-03,
                    3.73368194e-03,
                    9.19567967e-04,
                    7.83534653e-05,
                ],
                [
                    7.83534653e-05,
                    2.35760928e-03,
                    2.15315317e-02,
                    7.15226076e-02,
                    1.03823087e-01,
                    7.15226076e-02,
                    2.15315317e-02,
                    2.35760928e-03,
                ],
                [
                    9.19567967e-04,
                    2.15315317e-02,
                    1.48272439e-01,
                    3.83057355e-01,
                    5.00000000e-01,
                    3.83057355e-01,
                    1.48272439e-01,
                    2.15315317e-02,
                ],
                [
                    3.73368194e-03,
                    7.15226076e-02,
                    3.83057355e-01,
                    7.69781837e-01,
                    8.96176913e-01,
                    7.69781837e-01,
                    3.83057355e-01,
                    7.15226076e-02,
                ],
                [
                    5.86559882e-03,
                    1.03823087e-01,
                    5.00000000e-01,
                    8.96176913e-01,
                    9.94134401e-01,
                    8.96176913e-01,
                    5.00000000e-01,
                    1.03823087e-01,
                ],
                [
                    3.73368194e-03,
                    7.15226076e-02,
                    3.83057355e-01,
                    7.69781837e-01,
                    8.96176913e-01,
                    7.69781837e-01,
                    3.83057355e-01,
                    7.15226076e-02,
                ],
                [
                    9.19567967e-04,
                    2.15315317e-02,
                    1.48272439e-01,
                    3.83057355e-01,
                    5.00000000e-01,
                    3.83057355e-01,
                    1.48272439e-01,
                    2.15315317e-02,
                ],
                [
                    7.83534653e-05,
                    2.35760928e-03,
                    2.15315317e-02,
                    7.15226076e-02,
                    1.03823087e-01,
                    7.15226076e-02,
                    2.15315317e-02,
                    2.35760928e-03,
                ],
            ]
        )
        fmask = fuzzy_mask((8, 8), 2, 2)
        self.assertTrue(np.allclose(results, fmask, atol=1e-7))
