import os.path
from unittest import TestCase

import numpy as np
from scipy import misc

from aspire.image import Image, _im_translate2
from aspire.source import ArrayImageSource

DATA_DIR = os.path.join(os.path.dirname(__file__), 'saved_test_data')


class ImageTestCase(TestCase):
    def setUp(self):
        # numpy array for top-level functions that directly expect it
        self.im_np = misc.face(gray=True).astype('float64')[np.newaxis, :768, :768]
        # Independent Image object for testing Image methods
        self.im = Image(misc.face(gray=True).astype('float64')[:768, :768])

    def tearDown(self):
        pass

    def testImShift(self):
        # Ensure that the two separate im_translate functions we have return the same thing

        # A single shift applied to all images
        shifts = np.array([100, 200])

        im = self.im.shift(shifts)

        im1 = self.im._im_translate(shifts)
        # Note the difference in the concept of shifts for _im_translate2 - negative sign
        im2 = _im_translate2(self.im_np, -shifts)

        # Pure numpy 'shifting'
        # 'Shifting' an Image corresponds to a 'roll' of a numpy array - again, note the negated signs and the axes
        im3 = np.roll(self.im.asnumpy()[0], -shifts, axis=(0,1))

        self.assertTrue(np.allclose(im.asnumpy(), im1))
        self.assertTrue(np.allclose(im1, im2))
        self.assertTrue(np.allclose(im1[0, :, :], im3))


    def testArrayImageSource(self):
        # An Image can be wrapped in an ArrayImageSource when we need to deal with ImageSource objects.
        src = ArrayImageSource(self.im)
        im = src.images(start=0, num=np.inf)
        self.assertTrue(np.allclose(im.asnumpy(), self.im_np))
