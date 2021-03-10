import logging
from unittest import TestCase

import numpy as np
from scipy.spatial.transform import Rotation as sp_rot

from aspire.utils import Rotation, utest_tolerance

logger = logging.getLogger(__name__)


class UtilsTestCase(TestCase):
    def setUp(self):
        self.dtype = np.float32
        self.num_rots = 32
        self.rot_obj = Rotation.generate_random_rotations(
            self.num_rots, seed=0, dtype=self.dtype
        )
        self.angles = self.rot_obj.angles
        self.matrices = self.rot_obj.matrices

    def testRotMatrices(self):
        rot_ref = sp_rot.from_matrix(self.matrices)
        matrices = rot_ref.as_matrix().astype(self.dtype)
        self.assertTrue(
            np.allclose(self.matrices, matrices, atol=utest_tolerance(self.dtype))
        )

    def testRotAngles(self):
        rot_ref = sp_rot.from_euler("ZYZ", self.angles, degrees=False)
        angles = rot_ref.as_euler("ZYZ", degrees=False).astype(self.dtype)
        self.assertTrue(np.allclose(self.angles, angles))

    def testFromMatrix(self):
        rot_ref = sp_rot.from_matrix(self.matrices)
        angles = rot_ref.as_euler("ZYZ", degrees=False).astype(self.dtype)
        rot = Rotation.from_matrix(self.matrices, dtype=self.dtype)
        self.assertTrue(np.allclose(rot.angles, angles))

    def testFromEuler(self):
        rot_ref = sp_rot.from_euler("ZYZ", self.angles, degrees=False)
        matrices = rot_ref.as_matrix().astype(self.dtype)
        rot = Rotation.from_euler(self.angles, dtype=self.dtype)
        self.assertTrue(np.allclose(rot._matrices, matrices))

    def testInvert(self):
        rot_mat = self.rot_obj.matrices
        rot_mat_t = self.rot_obj.invert()
        self.assertTrue(np.allclose(rot_mat_t, np.transpose(rot_mat, (0, 2, 1))))

    def testMultiplication(self):
        result = (self.rot_obj * self.rot_obj.invert()).matrices
        for i in range(len(self.rot_obj)):
            self.assertTrue(
                np.allclose(np.eye(3), result[i], atol=utest_tolerance(self.dtype))
            )

    def testRegisterRots(self):
        q_mat = Rotation.generate_random_rotations(1, dtype=self.dtype)[0]
        for flag in [0, 1]:
            regrots_ref = self.rot_obj.apply_registration(q_mat, flag)
            q_mat_est, flag_est = self.rot_obj.find_registration(regrots_ref)
            self.assertTrue(
                np.allclose(flag_est, flag)
                and np.allclose(q_mat_est, q_mat, atol=utest_tolerance(self.dtype))
            )

    def testRegister(self):
        # These will yield two more distinct sets of random rotations wrt self.rot_obj
        set1 = Rotation.generate_random_rotations(self.num_rots, dtype=self.dtype)
        set2 = Rotation.generate_random_rotations(
            self.num_rots, dtype=self.dtype, seed=7
        )
        # Align both sets of random rotations to rot_obj
        aligned_rots1 = self.rot_obj.register(set1)
        aligned_rots2 = self.rot_obj.register(set2)
        self.assertTrue(aligned_rots1.mse(aligned_rots2) < utest_tolerance(self.dtype))
        self.assertTrue(aligned_rots2.mse(aligned_rots1) < utest_tolerance(self.dtype))

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

    def testString(self):
        logger.debug(str(self.rot_obj))

    def testRepr(self):
        logger.debug(repr(self.rot_obj))

    def testLen(self):
        self.assertTrue(len(self.rot_obj) == self.num_rots)

    def testSetterGetter(self):
        # Excute set
        tmp = np.arange(9).reshape((3, 3))
        self.rot_obj[13] = tmp
        # Execute get
        self.assertTrue(np.all(self.rot_obj[13] == tmp))

    def testDtype(self):
        self.assertTrue(self.dtype == self.rot_obj.dtype)
