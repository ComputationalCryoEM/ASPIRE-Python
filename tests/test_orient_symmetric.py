from unittest import TestCase

import numpy as np
from numpy import linalg, random

from aspire.abinitio import CLSymmetryC3C4
from aspire.source import Simulation
from aspire.utils import Rotation
from aspire.utils.misc import all_pairs, gaussian_3d
from aspire.volume import Volume


class OrientSymmTestCase(TestCase):
    def setUp(self):
        self.L = 32
        self.symm = "C4"
        self.n_ims = 32
        self.dtype = np.float32
        src = Simulation(L=self.L, n=self.n_ims, symmetry_type=self.symm)
        self.cl_class = CLSymmetryC3C4(src, n_symm=4, n_theta=360)

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

    def testCommonLines(self):
        n_ims = 32
        res = 64
        order = 4

        # Build symmetric volume and associated Simulation object.
        # For the Simulation object we use clean, non-shifted projection images.
        volume, rots_symm = self.buildSimpleSymmetricVolume(res, order)
        offsets = np.zeros((n_ims, 2))
        src = Simulation(
            L=res, n=n_ims, offsets=offsets, dtype=self.dtype, vols=volume, C=1
        )

        # Initialize the common-lines class and build common-lines matrix
        cl_symm = CLSymmetryC3C4(src, n_symm=order, n_theta=360)
        cl_symm.build_clmatrix()
        cl = cl_symm.clmatrix

        # Compare common-line indices with ground truth angles.
        rots = src.rots  # ground truth rotations
        pairs = all_pairs(n_ims)
        within_1_degree = 0
        within_5_degrees = 0
        for _, (i, j) in enumerate(pairs):
            a_ij_s = np.zeros(order)
            a_ji_s = np.zeros(order)
            cl_ij = cl[i, j] % 180
            cl_ji = cl[j, i] % 180

            # The common-line estimates cl_ij, cl_ji should match the
            # true common-line angles a_ij_s, a_ji_s for some value s,
            # where s is the number of common-lines induced by the symmetric order.
            for s in range(order):
                rel_rot = (rots[i]).T @ rots_symm[s] @ rots[j]
                a_ij_s[s] = np.rad2deg(np.arctan(-rel_rot[0, 2] / rel_rot[1, 2])) % 180
                a_ji_s[s] = np.rad2deg(np.arctan(-rel_rot[2, 0] / rel_rot[2, 1])) % 180
            best_s = np.argmin(abs(cl_ij - a_ij_s) + abs(cl_ji - a_ji_s))
            diff_ij = abs(cl_ij - a_ij_s[best_s])
            diff_ji = abs(cl_ji - a_ji_s[best_s])

            # Count the number of good estimates.
            if diff_ij < 1:
                within_1_degree += 1
                within_5_degrees += 1
            elif diff_ij < 5:
                within_5_degrees += 1

            if diff_ji < 1:
                within_1_degree += 1
                within_5_degrees += 1
            elif diff_ji < 5:
                within_5_degrees += 1

        # Assert that at least 99% of estimates are within 5 degrees and
        # at least 95% of estimates are within 1 degree.
        n_estimates = 2 * len(pairs)
        within_5 = within_5_degrees / n_estimates
        within_1 = within_1_degree / n_estimates
        self.assertTrue(within_5 > 0.99)
        self.assertTrue(within_1 > 0.95)

    def buildOuterProducts(self, n_ims):
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

        return vijs, viis, gt_vis

    def buildSimpleSymmetricVolume(self, res, order):
        # Construct rotatation matrices associated with cyclic order.
        angles = np.zeros((order, 3), dtype=self.dtype)
        angles[:, 2] = 2 * np.pi * np.arange(order) / order
        rots_symm = Rotation.from_euler(angles).matrices

        # Assign centers and sigmas of Gaussian blobs
        centers = np.zeros((3, order, 3), dtype=self.dtype)
        centers[0, 0, :] = np.array([res // 12, res // 12, 0])
        centers[1, 0, :] = np.array([res // 6, res // 6, res // 20])
        centers[2, 0, :] = np.array([res // 4, res // 7, res // 16])
        for o in range(order):
            centers[0, o, :] = rots_symm[o] @ centers[0, 0, :]
            centers[1, o, :] = rots_symm[o] @ centers[1, 0, :]
            centers[2, o, :] = rots_symm[o] @ centers[2, 0, :]
        sigmas = np.zeros((3, 3), dtype=self.dtype)
        sigmas[0] = [res / 15, res / 15, res / 15]
        sigmas[1] = [res / 20, res / 20, res / 20]
        sigmas[2] = [res / 30, res / 30, res / 30]

        # Build volume
        vol = np.zeros((res, res, res), dtype=self.dtype)
        n_blobs = centers.shape[0]
        for o in range(order):
            for i in range(n_blobs):
                vol += gaussian_3d(res, centers[i, o, :], sigmas[i])

        volume = Volume(vol)

        return volume, rots_symm
