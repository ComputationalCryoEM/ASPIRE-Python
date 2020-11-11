from unittest import TestCase

import numpy as np
from scipy.spatial.transform import Rotation as sp_rot

from aspire.utils.rotation import Rotation
from aspire.utils.types import utest_tolerance


class UtilsTestCase(TestCase):
    def setUp(self):
        self.rot_obj = Rotation(32, seed=0, dtype=np.float32)

    def testRotMatrices(self):
        rots_ref = (
            sp_rot.from_euler("ZYZ", self.rot_obj.angles, degrees=True)
            .as_matrix()
            .astype(self.rot_obj.dtype)
        )
        self.assertTrue(np.allclose(self.rot_obj.rot_matrices, rots_ref))

    def testRotAngles(self):
        rot = sp_rot.from_matrix(self.rot_obj.rot_matrices)
        angles = rot.as_euler(self.rot_obj.rot_seq, degrees=True).astype(
            self.rot_obj.dtype
        )
        self.assertTrue(np.allclose(self.rot_obj.angles, angles))

    def testTranspose(self):
        rot_mat = self.rot_obj.rot_matrices
        rot_mat_t = self.rot_obj.T
        self.assertTrue(np.allclose(rot_mat_t, np.transpose(rot_mat, (0, 2, 1))))

    def testMultiplication(self):
        rot_obj_t = Rotation(32, seed=0, seq="ZYZ")
        rot_obj_t.rot_matrices = self.rot_obj.T
        result = self.rot_obj * rot_obj_t
        for i in range(self.rot_obj.num_rots):
            self.assertTrue(
                np.allclose(
                    np.eye(3), result[i], atol=utest_tolerance(self.rot_obj.dtype)
                )
            )

    def testRegisterRots(self):
        q_ang = [[45, 45, 45]]
        q_mat = sp_rot.from_euler("ZYZ", q_ang, degrees=True).as_matrix()[0]
        flag = 0
        regrots_ref = self.rot_obj.get_aligned_rotations(q_mat, flag)
        q_mat_est, flag_est = self.rot_obj.register_rotations(regrots_ref)

        self.assertTrue(np.allclose(flag_est, flag) and np.allclose(q_mat_est, q_mat))
