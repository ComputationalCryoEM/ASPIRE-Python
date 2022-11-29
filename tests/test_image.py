import os.path
from unittest import TestCase

import numpy as np
from parameterized import parameterized
from scipy import misc

from aspire.image import Image
from aspire.utils import powerset

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class ImageTestCase(TestCase):
    def setUp(self):
        self.dtype = np.float64
        self.size = 768
        self.n = 3
        # numpy array for top-level functions that directly expect it
        self.im_np_e = misc.face(gray=True).astype(self.dtype)[
            np.newaxis, : self.size, : self.size
        ]
        self.im_np_o = self.im_np_e[:, : self.size - 1, : self.size - 1]
        # Independent Image object for testing Image methods
        self.im_e = Image(
            misc.face(gray=True).astype(self.dtype)[: self.size, : self.size]
        )
        self.im_o = Image(self.im_e.asnumpy()[:, : self.size - 1, : self.size - 1])
        # Construct a simple stack of Images
        self.ims_np_e = np.empty((self.n, *self.im_np_e.shape[1:]), dtype=self.dtype)
        for i in range(self.n):
            self.ims_np_e[i] = self.im_np_e * (i + 1) / float(self.n)
        self.ims_np_o = self.ims_np_e[:, : self.size - 1, : self.size - 1]
        # Independent Image stack object for testing Image methods
        self.ims_e = Image(self.ims_np_e)
        self.ims_o = Image(self.ims_np_o)

        self.parities = {
            "even": (
                self.im_np_e,
                self.im_e,
                self.ims_np_e,
                self.ims_e,
            ),
            "odd": (
                self.im_np_o,
                self.im_o,
                self.ims_np_o,
                self.ims_o,
            ),
        }

    def tearDown(self):
        pass

    @parameterized.expand([("even",), ("odd",)])
    def testImShift(self, parity):
        # Note that the _im_translate method can handle float input shifts, as it
        # computes the shifts in Fourier space, rather than performing a roll
        # However, NumPy's roll() only accepts integer inputs
        shifts = np.array([100, 200])

        im_np, im, _, _ = self.parities[parity]
        # test built-in
        im0 = im.shift(shifts)
        # test explicit call
        im1 = im._im_translate(shifts)
        # test that float input returns the same thing
        im2 = im.shift(shifts.astype(np.float64))
        # ground truth numpy roll
        im3 = np.roll(im_np[0, :, :], -shifts, axis=(0, 1))

        self.assertTrue(np.allclose(im0.asnumpy(), im1.asnumpy()))
        self.assertTrue(np.allclose(im1.asnumpy(), im2.asnumpy()))
        self.assertTrue(np.allclose(im0.asnumpy()[0, :, :], im3))

    @parameterized.expand([("even",), ("odd",)])
    def testImShiftStack(self, parity):
        # test stack of shifts (same number as Image.num_img)
        # mix of odd and even
        shifts = np.array([[100, 200], [203, 150], [55, 307]])

        _, _, ims_np, ims = self.parities[parity]

        # test built-in
        im0 = ims.shift(shifts)
        # test explicit call
        im1 = ims._im_translate(shifts)
        # test that float input returns the same thing
        im2 = ims.shift(shifts.astype(np.float64))
        # ground truth numpy roll
        im3 = np.array(
            [np.roll(ims_np[i, :, :], -shifts[i], axis=(0, 1)) for i in range(self.n)]
        )
        self.assertTrue(np.allclose(im0.asnumpy(), im1.asnumpy()))
        self.assertTrue(np.allclose(im1.asnumpy(), im2.asnumpy()))
        self.assertTrue(np.allclose(im0.asnumpy(), im3))

    @parameterized.expand([("even",), ("odd",)])
    def testImageShiftErrors(self, parity):
        _, im, _, _ = self.parities[parity]
        # test bad shift shape
        with self.assertRaisesRegex(ValueError, "Input shifts must be of shape"):
            _ = im.shift(np.array([100, 100, 100]))
        # test bad number of shifts
        with self.assertRaisesRegex(ValueError, "The number of shifts"):
            _ = im.shift(np.array([[100, 200], [100, 200]]))

    @parameterized.expand([("even",), ("odd",)])
    def testImageSqrt(self, parity):
        im_np, im, ims_np, ims = self.parities[parity]

        self.assertTrue(np.allclose(im.sqrt().asnumpy(), np.sqrt(im_np)))
        self.assertTrue(np.allclose(ims.sqrt().asnumpy(), np.sqrt(ims_np)))

    @parameterized.expand([("even",), ("odd",)])
    def testImageTranspose(self, parity):
        im_np, im, ims_np, ims = self.parities[parity]

        # test method and abbreviation
        self.assertTrue(np.allclose(im.T.asnumpy(), np.transpose(im_np, (0, 2, 1))))
        self.assertTrue(
            np.allclose(im.transpose().asnumpy(), np.transpose(im_np, (0, 2, 1)))
        )

        # Check individual imgs in a stack
        for i in range(ims_np.shape[0]):
            self.assertTrue(np.allclose(ims.T[i], ims_np[i].T))
            self.assertTrue(np.allclose(ims.transpose()[i], ims_np[i].T))

    @parameterized.expand([("even",), ("odd",)])
    def testImageFlip(self, parity):
        im_np, im, ims_np, ims = self.parities[parity]
        for axis in powerset(range(1, 3)):
            if not axis:
                # test default
                result_single = im.flip().asnumpy()
                result_stack = ims.flip().asnumpy()
                axis = 1
            else:
                result_single = im.flip(axis).asnumpy()
                result_stack = ims.flip(axis).asnumpy()
            # single image
            self.assertTrue(np.allclose(result_single, np.flip(im_np, axis)))
            # stack
            self.assertTrue(
                np.allclose(
                    result_stack,
                    np.flip(ims_np, axis),
                )
            )

        # test error for axis 0
        axes = [0, (0, 1)]
        for axis in axes:
            with self.assertRaisesRegex(ValueError, "stack axis"):
                _ = im.flip(axis)
