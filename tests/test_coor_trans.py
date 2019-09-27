from unittest import TestCase
import numpy as np

from aspire.utils.coor_trans import grid_2d, grid_3d, qrand, q_to_rot, qrand_rots, angles_to_rots, rots_to_angles

import os.path
DATA_DIR = os.path.join(os.path.dirname(__file__), 'saved_test_data')


class UtilsTestCase(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testAnglesToRots(self):
        # An n x 3 array of angles
        angles = np.array([[1.234, 1.541, -0.3124], [0.421, 1.133, 0.511]])
        rots = angles_to_rots(angles)

        self.assertTrue(np.allclose(
            rots[0, :, :],
            [
                [0.29944493,  -0.89511032,  0.33031842],
                [-0.07480906,  0.32311209,  0.94339927],
                [-0.95117629, -0.30720693,  0.02979192]
            ]
        ))

        self.assertTrue(np.allclose(
            rots[1, :, :],
            [
                [ 0.13763699,  -0.54569385,  0.82660407],
                [ 0.59746887,   0.71136089,  0.37013058],
                [-0.78999178,   0.44292653,  0.42394466]
            ]
        ))

    def testRotsToAngles(self):
        angles = np.array([[1.234, 1.541, -0.3124], [0.421, 1.133, 0.511]])
        rots = angles_to_rots(angles)
        angles_redux = rots_to_angles(rots)
        self.assertTrue(np.allclose(angles, angles_redux))

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
