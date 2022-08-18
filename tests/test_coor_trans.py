import os.path
from unittest import TestCase

import numpy as np

from aspire.utils import (
    Rotation,
    crop_pad_2d,
    crop_pad_3d,
    get_aligned_rotations,
    grid_2d,
    grid_3d,
    register_rotations,
    uniform_random_angles,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class UtilsTestCase(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testGrid2d(self):
        # Note these reference files were created using Matlab compat grid indexing.
        grid2d = grid_2d(8, indexing="xy")
        self.assertTrue(
            np.allclose(grid2d["x"], np.load(os.path.join(DATA_DIR, "grid2d_8_x.npy")))
        )
        self.assertTrue(
            np.allclose(grid2d["y"], np.load(os.path.join(DATA_DIR, "grid2d_8_y.npy")))
        )
        self.assertTrue(
            np.allclose(grid2d["r"], np.load(os.path.join(DATA_DIR, "grid2d_8_r.npy")))
        )
        self.assertTrue(
            np.allclose(
                grid2d["phi"], np.load(os.path.join(DATA_DIR, "grid2d_8_phi.npy"))
            )
        )

    def testGrid3d(self):
        # Note these reference files were created using Matlab compat grid indexing.
        grid3d = grid_3d(8, indexing="xyz")
        self.assertTrue(
            np.allclose(grid3d["x"], np.load(os.path.join(DATA_DIR, "grid3d_8_x.npy")))
        )
        self.assertTrue(
            np.allclose(grid3d["y"], np.load(os.path.join(DATA_DIR, "grid3d_8_y.npy")))
        )
        self.assertTrue(
            np.allclose(grid3d["z"], np.load(os.path.join(DATA_DIR, "grid3d_8_z.npy")))
        )
        self.assertTrue(
            np.allclose(grid3d["r"], np.load(os.path.join(DATA_DIR, "grid3d_8_r.npy")))
        )
        self.assertTrue(
            np.allclose(
                grid3d["phi"], np.load(os.path.join(DATA_DIR, "grid3d_8_phi.npy"))
            )
        )
        self.assertTrue(
            np.allclose(
                grid3d["theta"], np.load(os.path.join(DATA_DIR, "grid3d_8_theta.npy"))
            )
        )

    def testRegisterRots(self):
        angles = uniform_random_angles(32, seed=0)
        rots_ref = Rotation.from_euler(angles).matrices

        q_ang = [[np.pi / 4, np.pi / 4, np.pi / 4]]
        q_mat = Rotation.from_euler(q_ang).matrices[0]
        flag = 0
        regrots_ref = get_aligned_rotations(rots_ref, q_mat, flag)
        q_mat_est, flag_est = register_rotations(rots_ref, regrots_ref)

        self.assertTrue(np.allclose(flag_est, flag) and np.allclose(q_mat_est, q_mat))

    def testSquareCrop2D(self):
        # Test even/odd cases based on the convention that the center of a sequence of length n
        # is (n+1)/2 if n is odd and n/2 + 1 if even.
        # Cropping is done to keep the center of the sequence the same value before and after.
        # Therefore the following apply:
        # Cropping even to odd will result in the 0-index (beginning)
        # of the sequence being chopped off (x marks the center, ~ marks deleted data):
        # ---x-- => ~--x--
        # Cropping odd to even will result in the -1-index (end)
        # of the sequence being chopped off:
        # ---x--- => ---x--~

        # even to even
        a = np.diag(np.arange(8))
        test_a = np.diag(np.arange(1, 7))
        self.assertTrue(np.array_equal(test_a, crop_pad_2d(a, 6)))

        # even to odd
        # the extra row/column cut off are the top and left
        # due to the centering convention
        a = np.diag(np.arange(8))
        test_a = np.diag(np.arange(1, 8))
        self.assertTrue(np.array_equal(test_a, crop_pad_2d(a, 7)))

        # odd to odd
        a = np.diag(np.arange(9))
        test_a = np.diag(np.arange(1, 8))
        self.assertTrue(np.array_equal(test_a, crop_pad_2d(a, 7)))

        # odd to even
        # the extra row/column cut off are the bottom and right
        # due to the centering convention
        a = np.diag(np.arange(9))
        test_a = np.diag(np.arange(8))
        self.assertTrue(np.array_equal(test_a, crop_pad_2d(a, 8)))

    def testSquareCrop3D(self):
        # even to even
        a = np.zeros((8, 8, 8))
        # pad it with the parts that will be cropped off from a 10x10x10
        a = np.pad(a, ((1, 1), (1, 1), (1, 1)), "constant", constant_values=1)
        # after cropping
        test_a = np.zeros((8, 8, 8))
        self.assertTrue(np.array_equal(crop_pad_3d(a, 8), test_a))

        # even to odd
        a = np.zeros((7, 7, 7))
        # pad it with the parts that will be cropped off from a 10x10x10
        a = np.pad(a, ((2, 1), (2, 1), (2, 1)), "constant", constant_values=1)
        test_a = np.zeros((7, 7, 7))
        self.assertTrue(np.array_equal(crop_pad_3d(a, 7), test_a))

        # odd to odd
        a = np.zeros((7, 7, 7))
        # pad it with the parts that will be cropped off from a 9x9x9
        a = np.pad(a, ((1, 1), (1, 1), (1, 1)), "constant", constant_values=1)
        test_a = np.zeros((7, 7, 7))
        self.assertTrue(np.array_equal(crop_pad_3d(a, 7), test_a))

        # odd to even
        a = np.zeros((8, 8, 8))
        # pad it with the parts that will be cropped off from 11x11x11
        a = np.pad(a, ((1, 2), (1, 2), (1, 2)), "constant", constant_values=1)
        test_a = np.zeros((8, 8, 8))
        self.assertTrue(np.array_equal(crop_pad_3d(a, 8), test_a))

    def testSquarePad2D(self):
        # Test even/odd cases based on the convention that the center of a sequence of length n
        # is (n+1)/2 if n is odd and n/2 + 1 if even.
        # Padding is done to keep the center of the sequence the same value before and after.
        # Therefore the following apply:
        # Padding from even to odd results in the spare padding being added to the -1-index (end)
        # of the sequence (x represents the center, + represents padding):
        # ---x-- => ---x--+
        # Padding from odd to even results in the spare padding being added to the 0-index (beginning)
        # of the sequence:
        # --x-- => +--x--

        # even to even
        a = np.diag(np.arange(1, 9))
        test_a = np.diag([0, 1, 2, 3, 4, 5, 6, 7, 8, 0])
        self.assertTrue(np.array_equal(test_a, crop_pad_2d(a, 10)))

        # even to odd
        # the extra padding is to the bottom and right
        # due to the centering convention
        a = np.diag(np.arange(1, 9))
        test_a = np.diag([0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0])
        self.assertTrue(np.array_equal(test_a, crop_pad_2d(a, 11)))

        # odd to odd
        a = np.diag(np.arange(1, 10))
        test_a = np.diag([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
        self.assertTrue(np.array_equal(test_a, crop_pad_2d(a, 11)))

        # odd to even
        # the extra padding is to the top and left
        # due to the centering convention
        a = np.diag(np.arange(1, 10))
        test_a = np.diag([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertTrue(np.array_equal(test_a, crop_pad_2d(a, 10)))

    def testSquarePad3D(self):
        # even to even
        a = np.zeros((8, 8, 8))
        # after padding to 10x10x10
        test_a = np.pad(a, ((1, 1), (1, 1), (1, 1)), "constant", constant_values=1)
        self.assertTrue(np.array_equal(crop_pad_3d(a, 10, fill_value=1), test_a))

        # even to odd
        a = np.zeros((8, 8, 8))
        # after padding to 11x11x11
        test_a = np.pad(a, ((1, 2), (1, 2), (1, 2)), "constant", constant_values=1)
        self.assertTrue(np.array_equal(crop_pad_3d(a, 11, fill_value=1), test_a))

        # odd to odd
        a = np.zeros((7, 7, 7))
        # after padding to 9x9x9
        test_a = np.pad(a, ((1, 1), (1, 1), (1, 1)), "constant", constant_values=1)
        self.assertTrue(np.array_equal(crop_pad_3d(a, 9, fill_value=1), test_a))

        # odd to even
        a = np.zeros((7, 7, 7))
        # after padding to 10x10x10
        test_a = np.pad(a, ((2, 1), (2, 1), (2, 1)), "constant", constant_values=1)
        self.assertTrue(np.array_equal(crop_pad_3d(a, 10, fill_value=1), test_a))

    def testRectCrop2D(self):
        # Additional sanity checks for rectangular cropping case

        # 12x10 -> 10x10
        a = np.diag(np.arange(1, 11))
        # augment to 12 rows
        aug = np.vstack([a, np.zeros(10)])
        aug = np.vstack([np.zeros(10), aug])
        # make sure the top and bottom rows are stripped
        self.assertTrue(np.array_equal(a, crop_pad_2d(aug, 10)))

        # 10x12 -> 10x10
        a = np.diag(np.arange(1, 11))
        # augment to 12 columns
        aug = np.column_stack([a, np.zeros(10)])
        aug = np.column_stack([np.zeros(10), aug])
        # make sure the left and right columns are stripped
        self.assertTrue(np.array_equal(a, crop_pad_2d(aug, 10)))

        # 9x7 -> 7x7
        a = np.diag(np.arange(1, 8))
        # augment to 9 rows
        aug = np.vstack([a, np.zeros(7)])
        aug = np.vstack([np.zeros(7), aug])
        # make sure the top and bottom rows are stripped
        self.assertTrue(np.array_equal(a, crop_pad_2d(aug, 7)))

        # 7x9 -> 7x7
        a = np.diag(np.arange(1, 8))
        # augment to 9 columns
        aug = np.column_stack([a, np.zeros(7)])
        aug = np.column_stack([np.zeros(7), aug])
        # make sure the left and right columns are stripped
        self.assertTrue(np.array_equal(a, crop_pad_2d(aug, 7)))

    def testRectPad2D(self):
        # Additional sanity checks for rectangular padding case

        # 12x10 -> 12x12
        a = np.diag(np.arange(1, 11))
        # augment to 12 rows
        aug = np.vstack([a, np.zeros(10)])
        aug = np.vstack([np.zeros(10), aug])
        # expected result
        padded = np.column_stack([aug, np.zeros(12)])
        padded = np.column_stack([np.zeros(12), padded])
        # make sure columns of fill value (0) are added to the
        # left and right
        self.assertTrue(np.array_equal(padded, crop_pad_2d(aug, 12)))

        # 10x12 -> 12x12
        a = np.diag(np.arange(1, 11))
        # augment to 12 columns
        aug = np.column_stack([a, np.zeros(10)])
        aug = np.column_stack([np.zeros(10), aug])
        # expected result
        padded = np.vstack([aug, np.zeros(12)])
        padded = np.vstack([np.zeros(12), padded])
        # make sure rows of fill value (0) are added to the
        # top and bottom
        self.assertTrue(np.array_equal(padded, crop_pad_2d(aug, 12)))

        # 9x7 -> 9x9
        a = np.diag(np.arange(1, 8))
        # augment to 9 rows
        aug = np.vstack([a, np.zeros(7)])
        aug = np.vstack([np.zeros(7), aug])
        # expected result
        padded = np.column_stack([aug, np.zeros(9)])
        padded = np.column_stack([np.zeros(9), padded])
        # make sure columns of fill value (0) are added to the
        # left and right
        self.assertTrue(np.array_equal(padded, crop_pad_2d(aug, 9)))

        # 7x9 -> 9x9
        a = np.diag(np.arange(1, 8))
        # augment to 9 columns
        aug = np.column_stack([a, np.zeros(7)])
        aug = np.column_stack([np.zeros(7), aug])
        # expected result
        padded = np.vstack([aug, np.zeros(9)])
        padded = np.vstack([np.zeros(9), padded])
        # make sure rows of fill value (0) are added to the
        # top and bottom
        self.assertTrue(np.array_equal(padded, crop_pad_2d(aug, 9)))

    def testCropPad2DError(self):
        with self.assertRaises(ValueError) as e:
            _ = crop_pad_2d(np.zeros((6, 10)), 8)
            self.assertEqual(
                "Cannot crop and pad an image at the same time.", str(e.exception)
            )

    def testCropPad3DError(self):
        with self.assertRaises(ValueError) as e:
            _ = crop_pad_3d(np.zeros((6, 8, 10)), 8)
            self.assertEqual(
                "Cannot crop and pad a volume at the same time.", str(e.exception)
            )

    def testCrop2DDtype(self):
        # crop_pad_2d must return an array of the same dtype it was given
        # in particular, because the method is used for Fourier downsampling
        # methods involving cropping complex arrays
        self.assertEqual(
            crop_pad_2d(np.eye(10).astype("complex"), 5).dtype, np.dtype("complex128")
        )

    def testCrop3DDtype(self):
        self.assertEqual(
            crop_pad_3d(np.ones((8, 8, 8)).astype("complex"), 5).dtype,
            np.dtype("complex128"),
        )

    def testCrop2DFillValue(self):
        # make sure the fill value is as expected
        # we are padding from an odd to an even dimension
        # so the padded column is added to the left
        a = np.ones((4, 3))
        b = crop_pad_2d(a, 4, fill_value=-1)
        self.assertTrue(np.array_equal(b[:, 0], np.array([-1, -1, -1, -1])))

    def testCrop3DFillValue(self):
        # make sure the fill value is expected. Since we are padding from odd to even
        # the padded side is added to the 0-end of dimension 3
        a = np.ones((4, 4, 3))
        b = crop_pad_3d(a, 4, fill_value=-1)
        self.assertTrue(np.array_equal(b[:, :, 0], -1 * np.ones((4, 4))))
