from unittest import TestCase
import numpy as np

from scipy.spatial.transform import Rotation as sp_rot
from aspire.utils.rotation import Rotation
from aspire.utils.types import utest_tolerance


class UtilsTestCase(TestCase):
    def setUp(self):
        self.dtype = np.float32
        self.rot_obj = Rotation(32, seed=0, seq='ZYZ', dtype=self.dtype)
        self.angles = self.rot_obj.angles

    def testRotMatrices(self):
        rots_ref = sp_rot.from_euler('ZYZ', self.angles, degrees=True).as_matrix()
        self.assertTrue(np.allclose(self.rot_obj.rot_matrices, rots_ref))

    def testRotAngles(self):
        rot = sp_rot.from_matrix(self.rot_obj.rot_matrices)
        angles = rot.as_euler(self.rot_obj.rot_seq, degrees=True)
        self.assertTrue(np.allclose(self.rot_obj.angles, angles))

    def testTranspose(self):
        rot_mat = self.rot_obj.rot_matrices
        rot_mat_t = self.rot_obj.T
        for i in range(self.rot_obj.num_rots):
            self.assertTrue(np.allclose(rot_mat_t[i], rot_mat[i].T))

    def testMultiplication(self):
        rot_obj_t = Rotation(32, seed=0, seq='ZYZ')
        rot_obj_t.rot_matrices = self.rot_obj.T
        result = self.rot_obj*rot_obj_t
        for i in range(self.rot_obj.num_rots):
            self.assertTrue(
                np.allclose(np.eye(3), result[i],
                            atol=utest_tolerance(self.dtype)))

    def testRegisterRots(self):
        rots_ref = self.rot_obj.rot_matrices
        q_ang = [[45, 45, 45]]
        q_mat = sp_rot.from_euler('ZYZ', q_ang, degrees=True).as_matrix()[0]
        flag = 0
        regrots_ref = Rotation.get_aligned_rotations(rots_ref, q_mat, flag)
        q_mat_est, flag_est = Rotation.register_rotations(rots_ref, regrots_ref)

        self.assertTrue(np.allclose(flag_est, flag)
                        and np.allclose(q_mat_est, q_mat))
