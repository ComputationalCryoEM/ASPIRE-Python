from unittest import TestCase
import numpy as np

from aspire.utils.coor_trans import grid_2d, grid_3d, qrand, q_to_rot, qrand_rots
from aspire.utils.coor_trans import register_rotations

import os.path
DATA_DIR = os.path.join(os.path.dirname(__file__), 'saved_test_data')


class UtilsTestCase(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testGrid2d(self):
        grid2d = grid_2d(8)
        self.assertTrue(np.allclose(grid2d['x'], np.load(os.path.join(DATA_DIR, 'grid2d_8_x.npy'))))
        self.assertTrue(np.allclose(grid2d['y'], np.load(os.path.join(DATA_DIR, 'grid2d_8_y.npy'))))
        self.assertTrue(np.allclose(grid2d['r'], np.load(os.path.join(DATA_DIR, 'grid2d_8_r.npy'))))
        self.assertTrue(np.allclose(grid2d['phi'], np.load(os.path.join(DATA_DIR, 'grid2d_8_phi.npy'))))

    def testGrid3d(self):
        grid3d = grid_3d(8)
        self.assertTrue(np.allclose(grid3d['x'], np.load(os.path.join(DATA_DIR, 'grid3d_8_x.npy'))))
        self.assertTrue(np.allclose(grid3d['y'], np.load(os.path.join(DATA_DIR, 'grid3d_8_y.npy'))))
        self.assertTrue(np.allclose(grid3d['z'], np.load(os.path.join(DATA_DIR, 'grid3d_8_z.npy'))))
        self.assertTrue(np.allclose(grid3d['r'], np.load(os.path.join(DATA_DIR, 'grid3d_8_r.npy'))))
        self.assertTrue(np.allclose(grid3d['phi'], np.load(os.path.join(DATA_DIR, 'grid3d_8_phi.npy'))))
        self.assertTrue(np.allclose(grid3d['theta'], np.load(os.path.join(DATA_DIR, 'grid3d_8_theta.npy'))))

    def testQrand(self):
        results = np.load(os.path.join(DATA_DIR, 'rand_quaternions32.npy'))
        quaternions32 = qrand(32, seed=0)
        self.assertTrue(np.allclose(results, quaternions32, atol=1e-7))

    def testQ2Rot(self):
        results = np.load(os.path.join(DATA_DIR, 'rand_rot_matrices32.npy'))
        quaternions32 = qrand(32, seed=0)
        rot_matrices32 = q_to_rot(quaternions32)
        self.assertTrue(np.allclose(np.moveaxis(results, 2, 0), rot_matrices32, atol=1e-7))

    def testQrandRots(self):
        results = np.load(os.path.join(DATA_DIR, 'rand_rot_matrices32.npy'))
        rot_matrices32 = qrand_rots(32, seed=0)
        self.assertTrue(np.allclose(np.moveaxis(results, 2, 0), rot_matrices32, atol=1e-7))

    def testRegisterRots(self):
        rots = qrand_rots(32, seed=0).T
        regrots, mse, diff, o_mat, flag = register_rotations(rots, rots)

        o_result = np.array(
            [[ 1.00000000e+00, -3.37174190e-18,  2.60886371e-18],
             [ 3.37174190e-18,  1.00000000e+00, -5.19757560e-17],
             [-2.60886371e-18,  1.39415739e-16,  1.00000000e+00]]
        )
        self.assertTrue(np.allclose(regrots, rots) and np.allclose(o_mat, o_result))
