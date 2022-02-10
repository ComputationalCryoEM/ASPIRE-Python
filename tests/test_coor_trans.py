import os.path
from unittest import TestCase

import numpy as np

from aspire.utils import (
    Rotation,
    crop_2d,
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
        # test even/odd cases
        # based on the choice that the center of a sequence of length n is (n+1)/2
        # if n is odd and n/2 + 1 if even.

        # even to even
        # the center is preserved
        a = np.zeros((8, 8))
        np.fill_diagonal(a, np.arange(8))
        test_a = np.zeros((6, 6))
        np.fill_diagonal(test_a, np.arange(1, 7))
        self.assertTrue(np.array_equal(test_a, crop_2d(a, 6)))

        # even to odd
        # the crop gives us a[1:,1:] since we shift towards
        # higher x and y values due to the centering convention
        a = np.zeros((8, 8))
        np.fill_diagonal(a, np.arange(8))
        test_a = np.zeros((7, 7))
        np.fill_diagonal(test_a, np.arange(1, 8))
        self.assertTrue(np.array_equal(test_a, crop_2d(a, 7)))

        # odd to odd
        # the center is preserved
        a = np.zeros((9, 9))
        np.fill_diagonal(a, np.arange(9))
        test_a = np.zeros((7, 7))
        np.fill_diagonal(test_a, np.arange(1, 8))
        self.assertTrue(np.array_equal(test_a, crop_2d(a, 7)))

        # odd to even
        # the crop gives us a[:8, :8] since we shift towards
        # lower x and y values due to the centering convention
        a = np.zeros((9, 9))
        np.fill_diagonal(a, np.arange(9))
        test_a = np.zeros((8, 8))
        np.fill_diagonal(test_a, np.arange(8))
        self.assertTrue(np.array_equal(test_a, crop_2d(a, 8)))

    def testSquarePad2D(self):
        # test even/odd cases of padding operation of crop_2d

        # even to even
        # the center is preserved
        a = np.zeros((8, 8))
        np.fill_diagonal(a, np.arange(1, 9))
        test_a = np.zeros((10, 10))
        np.fill_diagonal(test_a, [0, 1, 2, 3, 4, 5, 6, 7, 8, 0])
        self.assertTrue(np.array_equal(test_a, crop_2d(a, 10)))

        # even to odd
        # the shift is towards lower x and y values
        # due to the centering convention
        a = np.zeros((8, 8))
        np.fill_diagonal(a, np.arange(1, 9))
        test_a = np.zeros((11, 11))
        np.fill_diagonal(test_a, [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0])
        self.assertTrue(np.array_equal(test_a, crop_2d(a, 11)))

        # odd to odd
        # the center is preserved
        a = np.zeros((9, 9))
        np.fill_diagonal(a, np.arange(1, 10))
        test_a = np.zeros((11, 11))
        np.fill_diagonal(test_a, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
        self.assertTrue(np.array_equal(test_a, crop_2d(a, 11)))

        # odd to even
        # the shift is towards higher x and y values
        # due to the centering convention
        a = np.zeros((9, 9))
        np.fill_diagonal(a, np.arange(1, 10))
        test_a = np.zeros((10, 10))
        np.fill_diagonal(test_a, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertTrue(np.array_equal(test_a, crop_2d(a, 10)))

    def testCrop2DDtype(self):
        # crop_2d must return an array of the same dtype it was given
        # in particular, because the method is used for Fourier downsampling
        # methods involving cropping complex arrays
        self.assertEqual(
            crop_2d(np.eye(10).astype("complex"), 5).dtype, np.dtype("complex128")
        )
