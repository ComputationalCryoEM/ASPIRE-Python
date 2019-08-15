from unittest import TestCase
import numpy as np

from aspire.utils.coor_trans import grid_2d, grid_3d

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