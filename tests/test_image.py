import os.path
from unittest import TestCase

import numpy as np
from scipy import misc

from aspire.image import Image, _im_translate2

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class ImageTestCase(TestCase):
    def setUp(self):
        self.dtype = np.float64
        # numpy array for top-level functions that directly expect it
        self.im_np = misc.face(gray=True).astype(self.dtype)[np.newaxis, :768, :768]
        # Independent Image object for testing Image methods
        self.im = Image(misc.face(gray=True).astype(self.dtype)[:768, :768])
        # Construct a simple stack of Images
        self.n = 3
        self.ims_np = np.empty((3, *self.im_np.shape[1:]), dtype=self.dtype)
        for i in range(self.n):
            self.ims_np[i] = self.im_np * (i + 1) / float(self.n)
        # Independent Image stack object for testing Image methods
        self.ims = Image(self.ims_np)

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
        im3 = np.roll(self.im.asnumpy()[0], -shifts, axis=(0, 1))

        self.assertTrue(np.allclose(im.asnumpy(), im1.asnumpy()))
        self.assertTrue(np.allclose(im1.asnumpy(), im2.asnumpy()))
        self.assertTrue(np.allclose(im1.asnumpy()[0, :, :], im3))

    def testImageSqrt(self):
        self.assertTrue(np.allclose(self.im.sqrt().asnumpy(), np.sqrt(self.im_np)))

        self.assertTrue(np.allclose(self.ims.sqrt().asnumpy(), np.sqrt(self.ims_np)))

    def testImageTranspose(self):
        self.assertTrue(
            np.allclose(
                self.im.flip_axes().asnumpy(), np.transpose(self.im_np, (0, 2, 1))
            )
        )

        # This is equivalent to checking np.tranpose(..., (0, 2, 1))
        for i in range(self.ims_np.shape[0]):

            self.assertTrue(np.allclose(self.ims.flip_axes()[i], self.ims_np[i].T))

            # Check against the contruction.
            self.assertTrue(
                np.allclose(
                    self.ims.flip_axes()[i], self.im_np[0].T * (i + 1) / float(self.n)
                )
            )
