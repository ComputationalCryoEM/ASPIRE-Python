import os.path
from unittest import TestCase

import numpy as np
from parameterized import parameterized_class
from scipy import misc

from aspire.image import Image
from aspire.utils import powerset

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


@parameterized_class(("parity",), [(0,), (1,)])
class ImageTestCase(TestCase):
    def setUp(self):
        self.dtype = np.float64
        self.n = 3
        self.size = 768 - self.parity
        # numpy array for top-level functions that directly expect it
        self.im_np = misc.face(gray=True).astype(self.dtype)[
            np.newaxis, : self.size, : self.size
        ]
        # Independent Image object for testing Image methods
        self.im = Image(
            misc.face(gray=True).astype(self.dtype)[: self.size, : self.size]
        )

        # Construct a simple stack of Images
        self.ims_np = np.empty((self.n, *self.im_np.shape[1:]), dtype=self.dtype)
        for i in range(self.n):
            self.ims_np[i] = self.im_np * (i + 1) / float(self.n)

        # Independent Image stack object for testing Image methods
        self.ims = Image(self.ims_np)

        # Multi dimensional stack Image object
        self.mdim = 2
        self.mdim_ims_np = np.concatenate([self.ims_np] * self.mdim).reshape(
            self.mdim, *self.ims_np.shape
        )
        self.mdim_ims = Image(self.mdim_ims_np)

    def tearDown(self):
        pass

    def testImShift(self):
        # Note that the _im_translate method can handle float input shifts, as it
        # computes the shifts in Fourier space, rather than performing a roll
        # However, NumPy's roll() only accepts integer inputs
        shifts = np.array([100, 200])

        # test built-in
        im0 = self.im.shift(shifts)
        # test explicit call
        im1 = self.im._im_translate(shifts)
        # test that float input returns the same thing
        im2 = self.im.shift(shifts.astype(np.float64))
        # ground truth numpy roll
        im3 = np.roll(self.im_np[0, :, :], -shifts, axis=(0, 1))

        self.assertTrue(np.allclose(im0.asnumpy(), im1.asnumpy()))
        self.assertTrue(np.allclose(im1.asnumpy(), im2.asnumpy()))
        self.assertTrue(np.allclose(im0.asnumpy()[0, :, :], im3))

    def testImShiftStack(self):
        # test stack of shifts (same number as Image.num_img)
        # mix of odd and even
        shifts = np.array([[100, 200], [203, 150], [55, 307]])

        # test built-in
        im0 = self.ims.shift(shifts)
        # test explicit call
        im1 = self.ims._im_translate(shifts)
        # test that float input returns the same thing
        im2 = self.ims.shift(shifts.astype(np.float64))
        # ground truth numpy roll
        im3 = np.array(
            [
                np.roll(self.ims_np[i, :, :], -shifts[i], axis=(0, 1))
                for i in range(self.n)
            ]
        )
        self.assertTrue(np.allclose(im0.asnumpy(), im1.asnumpy()))
        self.assertTrue(np.allclose(im1.asnumpy(), im2.asnumpy()))
        self.assertTrue(np.allclose(im0.asnumpy(), im3))

    def testImageShiftErrors(self):
        # test bad shift shape
        with self.assertRaisesRegex(ValueError, "Input shifts must be of shape"):
            _ = self.im.shift(np.array([100, 100, 100]))
        # test bad number of shifts
        with self.assertRaisesRegex(ValueError, "The number of shifts"):
            _ = self.im.shift(np.array([[100, 200], [100, 200]]))

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

    def testShape(self):
        self.assertEqual(self.ims.shape, self.ims_np.shape)
        self.assertEqual(self.ims.stack_shape, self.ims_np.shape[:-2])
        self.assertEqual(self.ims.stack_ndim, 1)

    def testMultiDimShape(self):
        self.assertEqual(self.mdim_ims.shape, self.mdim_ims_np.shape)
        self.assertEqual(self.mdim_ims.stack_shape, self.mdim_ims_np.shape[:-2])
        self.assertEqual(self.mdim_ims.stack_ndim, self.mdim)
        self.assertEqual(self.mdim_ims.n_images, self.mdim * self.ims.n_images)

    def testBadKey(self):
        with self.assertRaisesRegex(ValueError, "slice length must be"):
            _ = self.mdim_ims[tuple(range(self.mdim_ims.ndim + 1))]

    def testMultiDimGets(self):
        for X in self.mdim_ims:
            self.assertTrue(np.allclose(self.ims_np, X))

        # Test a slice
        self.assertTrue(np.allclose(self.mdim_ims[:, 1:], self.ims[1:]))

    def testMultiDimSets(self):
        self.mdim_ims[0, 1] = 123
        # Check the values changed
        self.assertTrue(np.allclose(self.mdim_ims[0, 1], 123))
        # and only those values changed
        self.assertTrue(np.allclose(self.mdim_ims[0, 0], self.ims_np[0]))
        self.assertTrue(np.allclose(self.mdim_ims[0, 2:], self.ims_np[2:]))
        self.assertTrue(np.allclose(self.mdim_ims[1, :], self.ims_np))

    def testMultiDimSetsSlice(self):
        # Test setting a slice
        self.mdim_ims[0, 1:] = 456
        # Check the values changed
        self.assertTrue(np.allclose(self.mdim_ims[0, 1:], 456))
        # and only those values changed
        self.assertTrue(np.allclose(self.mdim_ims[0, 0], self.ims_np[0]))
        self.assertTrue(np.allclose(self.mdim_ims[1, :], self.ims_np))

    def testMultiDimReshape(self):
        # Try mdim reshape
        X = self.mdim_ims.stack_reshape(*self.mdim_ims.stack_shape[::-1])
        self.assertEqual(X.stack_shape, self.mdim_ims.stack_shape[::-1])
        # Compare with direct np.reshape of axes of ndarray
        shape = (*self.mdim_ims_np.shape[:-2][::-1], *self.mdim_ims_np.shape[-2:])
        self.assertTrue(np.allclose(X.asnumpy(), self.mdim_ims_np.reshape(shape)))

    def testMultiDimFlattens(self):
        # Try flattening
        X = self.mdim_ims.stack_reshape(self.mdim_ims.n_images)
        self.assertEqual(X.stack_shape, (self.mdim_ims.n_images,))

    def testMultiDimFlattensTrick(self):
        # Try flattening with -1
        X = self.mdim_ims.stack_reshape(-1)
        self.assertEqual(X.stack_shape, (self.mdim_ims.n_images,))

    def testMultiDimReshapeTuples(self):
        # Try flattening with (-1,)
        X = self.mdim_ims.stack_reshape((-1,))
        self.assertEqual(X.stack_shape, (self.mdim_ims.n_images,))

        # Try mdim reshape
        X = self.mdim_ims.stack_reshape(self.mdim_ims.stack_shape[::-1])
        self.assertEqual(X.stack_shape, self.mdim_ims.stack_shape[::-1])

    def testMultiDimBadReshape(self):
        # Incorrect flat shape
        with self.assertRaisesRegex(ValueError, "Number of images"):
            _ = self.mdim_ims.stack_reshape(8675309)

        # Incorrect mdin shape
        with self.assertRaisesRegex(ValueError, "Number of images"):
            _ = self.mdim_ims.stack_reshape(42, 8675309)

    def testMultiDimBroadcast(self):
        X = self.mdim_ims + self.ims
        self.assertTrue(np.allclose(X[0], 2 * self.ims.asnumpy()))
