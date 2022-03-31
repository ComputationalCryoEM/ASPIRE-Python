from unittest import TestCase

import numpy as np
from numpy import linalg, random

from aspire.abinitio import CLSymmetryC3C4
from aspire.source import Simulation


class OrientSymmTestCase(TestCase):
    def setUp(self):
        self.L = 32
        self.symm = "C4"
        self.n_ims = 32
        src = Simulation(L=self.L, n=self.n_ims, symmetry_type=self.symm)
        self.cl_class = CLSymmetryC3C4(src, n_symm=4, n_theta=360)
        self.seed = 8679305

    def tearDown(self):
        pass

    def testGlobalJSync(self):
        n_ims = self.n_ims

        # Build a set of outer products of random third rows.
        vijs, viis, _ = self.buildOuterProducts(n_ims)

        # J-conjugate some of these outer products (every other element).
        J = np.diag((-1, -1, 1))
        vijs_conj, viis_conj = vijs.copy(), viis.copy()
        vijs_conj[::2] = J @ vijs_conj[::2] @ J
        viis_conj[::2] = J @ viis_conj[::2] @ J

        # Synchronize vijs_conj and viis_conj.
        vijs_sync, viis_sync = self.cl_class._global_J_sync(vijs_conj, viis_conj)

        # Check that synchronized outer products equal original
        # up to J-conjugation of the entire set.

        if vijs[0].all() == vijs_sync[0].all():
            self.assertTrue(np.allclose(vijs, vijs_sync))
            self.assertTrue(np.allclose(viis, viis_sync))
        else:
            self.assertTrue(np.allclose(vijs, J @ vijs_sync @ J))
            self.assertTrue(np.allclose(viis, J @ viis_sync @ J))

    def testEstimateThirdRows(self):
        n_ims = self.n_ims

        # Build outer products vijs, viis, and get ground truth third rows.
        vijs, viis, gt_vis = self.buildOuterProducts(n_ims)

        # Estimate third rows from outer products
        vis = self.cl_class._estimate_third_rows(vijs, viis)

        # Check if all-close up to difference of sign
        ground_truth = np.sign(gt_vis) * gt_vis
        estimate = np.sign(vis) * vis
        self.assertTrue(np.allclose(ground_truth, estimate))

    def buildOuterProducts(self, n_ims):
        # Build random third rows, ground truth vis (unit vectors)
        gt_vis = np.zeros((n_ims, 3), dtype=np.float32)
        random.seed(self.seed)
        for i in range(n_ims):
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

        return vijs, viis, gt_vis
