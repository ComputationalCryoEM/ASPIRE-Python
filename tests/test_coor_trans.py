import os.path
from unittest import TestCase

import numpy as np
from scipy.spatial.transform import Rotation

from aspire.utils.coor_trans import (
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
        grid2d = grid_2d(8)
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
        grid3d = grid_3d(8)
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
        rots_ref = Rotation.from_euler("ZYZ", angles).as_matrix()

        q_ang = [[45, 45, 45]]
        q_mat = Rotation.from_euler("ZYZ", q_ang, degrees=True).as_matrix()[0]
        flag = 0
        regrots_ref = get_aligned_rotations(rots_ref, q_mat, flag)
        q_mat_est, flag_est = register_rotations(rots_ref, regrots_ref)

        self.assertTrue(np.allclose(flag_est, flag) and np.allclose(q_mat_est, q_mat))
