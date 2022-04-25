from unittest import TestCase

import numpy as np
from numpy import linalg

from aspire.abinitio import CLSymmetryC3C4
from aspire.source import Simulation
from aspire.utils import J_conjugate, all_pairs
from aspire.utils.random import randn


class OrientSymmTestCase(TestCase):
    def setUp(self):
        self.L = 32
        self.symm = "C4"
        self.n_img = 32
        self.seed = 8675309
        src = Simulation(L=self.L, n=self.n_img, symmetry=self.symm)
        self.cl_class = CLSymmetryC3C4(
            src, symmetry=self.symm, n_theta=360, seed=self.seed
        )

    def tearDown(self):
        pass

    def testGlobalJSync(self):
        n_img = self.n_img

        # Build a set of outer products of random third rows.
        vijs, viis, _ = self.buildOuterProducts(n_img)

        # J-conjugate some of these outer products (every other element).
        vijs_conj, viis_conj = vijs.copy(), viis.copy()
        vijs_conj[::2] = J_conjugate(vijs_conj[::2])
        viis_conj[::2] = J_conjugate(viis_conj[::2])

        # Synchronize vijs_conj and viis_conj.
        vijs_sync, viis_sync = self.cl_class._global_J_sync(vijs_conj, viis_conj)

        # Check that synchronized outer products equal original
        # up to J-conjugation of the entire set.
        if (vijs[0] == vijs_sync[0]).all():
            self.assertTrue(np.allclose(vijs, vijs_sync))
            self.assertTrue(np.allclose(viis, viis_sync))
        else:
            self.assertTrue(np.allclose(vijs, J_conjugate(vijs_sync)))
            self.assertTrue(np.allclose(viis, J_conjugate(viis_sync)))

    def testEstimateThirdRows(self):
        n_img = self.n_img

        # Build outer products vijs, viis, and get ground truth third rows.
        vijs, viis, gt_vis = self.buildOuterProducts(n_img)

        # Estimate third rows from outer products.
        # Due to factorization of V, these might be negated third rows.
        vis = self.cl_class._estimate_third_rows(vijs, viis)

        # Check if all-close up to difference of sign
        ground_truth = np.sign(gt_vis[0, 0]) * gt_vis
        estimate = np.sign(vis[0, 0]) * vis
        self.assertTrue(np.allclose(ground_truth, estimate))

    def buildOuterProducts(self, n_img):
        # Build random third rows, ground truth vis (unit vectors)
        gt_vis = np.zeros((n_img, 3), dtype=np.float32)
        for i in range(n_img):
            v = randn(3, seed=self.seed)
            gt_vis[i] = v / linalg.norm(v)

        # Find outer products viis and vijs for i<j
        nchoose2 = int(n_img * (n_img - 1) / 2)
        vijs = np.zeros((nchoose2, 3, 3))
        viis = np.zeros((n_img, 3, 3))

        # All pairs (i,j) where i<j
        pairs = all_pairs(n_img)

        for k, (i, j) in enumerate(pairs):
            vijs[k] = np.outer(gt_vis[i], gt_vis[j])

        for i in range(n_img):
            viis[i] = np.outer(gt_vis[i], gt_vis[i])

        return vijs, viis, gt_vis
