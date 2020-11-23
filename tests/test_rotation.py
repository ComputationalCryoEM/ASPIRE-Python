from unittest import TestCase

import numpy as np
from scipy.spatial.transform import Rotation as sp_rot

from aspire.utils.rotation import Rotation
from aspire.utils.types import utest_tolerance


class UtilsTestCase(TestCase):
    def setUp(self):
        self.dtype = np.float32
        self.num_rots = 32
        self.rot_obj = Rotation(self.num_rots, seed=0, dtype=self.dtype)
        self.angles = self.rot_obj.angles
        self.matrices = self.rot_obj.matrices

    def testRotMatrices(self):
        rot_ref = sp_rot.from_matrix(self.matrices)
        matrices = rot_ref.as_matrix().astype(self.dtype)
        self.assertTrue(np.allclose(self.matrices, matrices))

    def testRotAngles(self):
        rot_ref = sp_rot.from_euler("ZYZ", self.angles, degrees=False)
        angles = rot_ref.as_euler("ZYZ", degrees=False).astype(self.dtype)
        self.assertTrue(np.allclose(self.angles, angles))

    def testFromMatrix(self):
        rot_ref = sp_rot.from_matrix(self.matrices)
        angles = rot_ref.as_euler("ZYZ", degrees=False).astype(self.dtype)
        rot = Rotation(self.num_rots, matrices=self.matrices, dtype=self.dtype)
        self.assertTrue(np.allclose(rot.angles, angles))

    def testFromEuler(self):
        rot_ref = sp_rot.from_euler("ZYZ", self.angles, degrees=False)
        matrices = rot_ref.as_matrix().astype(self.dtype)
        rot = Rotation(self.num_rots, angles=self.angles, dtype=self.dtype)
        self.assertTrue(np.allclose(rot.matrices, matrices))

    def testTranspose(self):
        rot_mat = self.rot_obj.matrices
        rot_mat_t = self.rot_obj.T
        self.assertTrue(np.allclose(rot_mat_t, np.transpose(rot_mat, (0, 2, 1))))

    def testMultiplication(self):
        rot_obj_t = Rotation(self.num_rots, matrices=self.rot_obj.T)
        result = self.rot_obj * rot_obj_t
        for i in range(self.rot_obj.num_rots):
            self.assertTrue(
                np.allclose(np.eye(3), result[i], atol=utest_tolerance(self.dtype))
            )

    def testRegisterRots(self):
        q_ang = [np.random.random(3)]
        q_mat = sp_rot.from_euler("ZYZ", q_ang, degrees=False).as_matrix()[0]
        for flag in [0, 1]:
            regrots_ref = self.rot_obj.apply_registration(q_mat, flag)
            q_mat_est, flag_est = self.rot_obj.find_registration(regrots_ref)
            self.assertTrue(
                np.allclose(flag_est, flag)
                and np.allclose(q_mat_est, q_mat, atol=utest_tolerance(self.dtype))
            )

    def testMSE(self):
        q_ang = [np.random.random(3)]
        q_mat = sp_rot.from_euler("ZYZ", q_ang, degrees=False).as_matrix()[0]
        for flag in [0, 1]:
            regrots_ref = self.rot_obj.apply_registration(q_mat, flag)
            mse = self.rot_obj.mse(regrots_ref)
            self.assertTrue(mse < utest_tolerance(self.dtype))

    def testCommonLines(self):
        ell_ij, ell_ji = self.rot_obj.common_lines(8, 11, 360)
        self.assertTrue(ell_ij == 235 and ell_ji == 284)
