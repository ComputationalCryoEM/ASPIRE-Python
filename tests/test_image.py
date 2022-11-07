import os.path
from unittest import TestCase

import numpy as np
from scipy import misc

from aspire.image import Image, _im_translate2
from aspire.utils import powerset

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
        self.ims_np = np.empty((self.n, *self.im_np.shape[1:]), dtype=self.dtype)
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
        # test method and abbreviation
        self.assertTrue(
            np.allclose(self.im.T.asnumpy(), np.transpose(self.im_np, (0, 2, 1)))
        )
        self.assertTrue(
            np.allclose(
                self.im.transpose().asnumpy(), np.transpose(self.im_np, (0, 2, 1))
            )
        )

        # Check individual imgs in a stack
        for i in range(self.ims_np.shape[0]):
            self.assertTrue(np.allclose(self.ims.T[i], self.ims_np[i].T))
            self.assertTrue(np.allclose(self.ims.transpose()[i], self.ims_np[i].T))

    def testImageFlip(self):
        for axis in powerset(range(1, 3)):
            if not axis:
                # test default
                result_single = self.im.flip().asnumpy()
                result_stack = self.ims.flip().asnumpy()
                axis = 1
            else:
                result_single = self.im.flip(axis).asnumpy()
                result_stack = self.ims.flip(axis).asnumpy()
            # single image
            self.assertTrue(np.allclose(result_single, np.flip(self.im_np, axis)))
            # stack
            self.assertTrue(
                np.allclose(
                    result_stack,
                    np.flip(self.ims_np, axis),
                )
            )

        # test error for axis 0
        axes = [0, (0, 1)]
        for axis in axes:
            with self.assertRaisesRegex(ValueError, "stack axis"):
                _ = self.im.flip(axis)
