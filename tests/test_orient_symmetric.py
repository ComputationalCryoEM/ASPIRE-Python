from unittest import TestCase

import numpy as np
from numpy import linalg, random

from aspire.abinitio import CLSymmetryC3C4


class OrientSymmTestCase(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testEstimateThirdRows(self):
        n_ims = 25

        # Build random third rows, ground truth vis (unit vectors)
        gt_vis = np.zeros((n_ims, 3), dtype=np.float32)
        for i in range(n_ims):
            random.seed(i)
            v = random.randn(3)
            gt_vis[i] = v / linalg.norm(v)

        # Find outer products viis and vijs for i<j
        vijs = np.zeros((int(n_ims * (n_ims - 1) / 2), 3, 3))
        viis = np.zeros((n_ims, 3, 3))

        # All pairs (i,j) where i<j
        indices = np.arange(n_ims)
        pairs = [(i, j) for idx, i in enumerate(indices) for j in indices[idx + 1 :]]

        for k, (i, j) in enumerate(pairs):
            vijs[k] = np.outer(gt_vis[i], gt_vis[j])

        for i in range(n_ims):
            viis[i] = np.outer(gt_vis[i], gt_vis[i])

        # Estimate third rows from outer products
        vis = CLSymmetryC3C4.estimate_third_rows(vijs, viis)

        # Check if all-close up to difference of sign
        ground_truth = np.sign(gt_vis) * gt_vis
        estimate = np.sign(vis) * vis
        np.allclose(ground_truth, estimate)
