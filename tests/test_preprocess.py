import numpy as np
from unittest import TestCase

from scipy.fftpack import fftshift, fftn

from aspire.utils.preprocess import crop_pad, downsample, vol2img
from aspire.utils.coor_trans import qrand_rots
import os.path
DATA_DIR = os.path.join(os.path.dirname(__file__), 'saved_test_data')


class PreprocessTestCase(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test01CropPad(self):
        results = np.load(os.path.join(DATA_DIR, 'clean70SRibosome_vol_crop8.npy'))
        vols = np.load(os.path.join(DATA_DIR, 'clean70SRibosome_vol.npy'))
        vols = vols[..., np.newaxis]
        vols_f = crop_pad(fftshift(fftn(vols[:, :, :, 0])), 8)
        self.assertTrue(np.allclose(results, vols_f, atol=1e-7))

    def test02Downsample(self):
        results = np.load(os.path.join(DATA_DIR, 'clean70SRibosome_vol_down8.npy'))
        results = results[..., np.newaxis]
        vols = np.load(os.path.join(DATA_DIR, 'clean70SRibosome_vol.npy'))
        vols = vols[..., np.newaxis]
        vols = downsample(vols, (8, 8, 8))
        self.assertTrue(np.allclose(results, vols, atol=1e-7))

    def test03Vol2img(self):
        results = np.load(os.path.join(DATA_DIR, 'clean70SRibosome_down8_imgs32.npy'))
        vols = np.load(os.path.join(DATA_DIR, 'clean70SRibosome_vol_down8.npy'))
        vols = vols[..., np.newaxis]
        rots = qrand_rots(32, seed=0)
        imgs_clean = vol2img(vols[..., 0], rots)
        self.assertTrue(np.allclose(results, imgs_clean, atol=1e-7))
