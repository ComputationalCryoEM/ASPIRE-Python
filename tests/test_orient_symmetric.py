from unittest import TestCase

import numpy as np
from numpy import linalg, pi, random
from parameterized import parameterized

from aspire.abinitio import CLSymmetryC3C4
from aspire.source import Simulation
from aspire.utils import Rotation
from aspire.utils.misc import all_pairs, gaussian_3d
from aspire.volume import Volume


class OrientSymmTestCase(TestCase):
    # `order` is at this scope to be picked up by parameterized in testCommonLines.
    order = 3

    def setUp(self):
        self.L = 64
        self.n_ims = 32
        self.dtype = np.float32
        self.n_theta = 360

        # Build symmetric volume and associated Simulation object and common-lines class.
        # For the Simulation object we use clean, non-shifted projection images.
        orders = [3, 4]
        self.vols = {}
        self.srcs = {}
        self.cl_classes = {}

        for o in orders:
            self.vols[o] = self.buildSimpleSymmetricVolume(self.L, o)

            self.srcs[o] = Simulation(
                L=self.L,
                n=self.n_ims,
                offsets=np.zeros((self.n_ims, 2)),
                dtype=self.dtype,
                vols=self.vols[o],
                C=1,
            )

            self.cl_classes[o] = CLSymmetryC3C4(
                self.srcs[o], n_symm=o, n_theta=self.n_theta
            )

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
        vijs_sync, viis_sync = self.cl_classes[3]._global_J_sync(vijs_conj, viis_conj)

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
        vis = self.cl_classes[3]._estimate_third_rows(vijs, viis)

        # Check if all-close up to difference of sign
        ground_truth = np.sign(gt_vis) * gt_vis
        estimate = np.sign(vis) * vis
        self.assertTrue(np.allclose(ground_truth, estimate))

    @parameterized.expand([(order,), (order + 1,)])
    def testSelfCommonLines(self, order):
        n_ims = self.n_ims
        n_theta = self.n_theta
        src = self.srcs[order]
        cl_symm = self.cl_classes[order]

        # Initialize common-lines class and compute self-common-lines matrix.
        scl, _, _ = cl_symm._self_clmatrix_c3_c4()

        # Compute ground truth self-common-lines matrix.
        rots = src.rots
        scl_gt = self.buildSelfCommonLinesMatrix(rots, order)

        # Get angle difference between scl_gt and scl.
        scl_diff1 = scl_gt - scl
        scl_diff2 = scl_gt - np.flip(scl, 1)  # Order of indices might be switched.
        scl_diff1_angle = scl_diff1 * 2 * pi / n_theta
        scl_diff2_angle = scl_diff2 * 2 * pi / n_theta

        # cosine is invariant to 2pi, and abs is invariant to +-pi due to J-conjugation.
        # We take the mean deviation wrt to the two lines in each image.
        scl_diff1_angle_mean = np.mean(np.arccos(abs(np.cos(scl_diff1_angle))), axis=1)
        scl_diff2_angle_mean = np.mean(np.arccos(abs(np.cos(scl_diff2_angle))), axis=1)

        scl_diff_angle_mean = np.vstack((scl_diff1_angle_mean, scl_diff2_angle_mean))
        scl_idx = np.argmin(scl_diff_angle_mean, axis=0)
        min_mean_angle_diff = scl_idx.choose(scl_diff_angle_mean)

        # Assert scl detection rate is greater than 90% with 1 degree tolerance for order=3,
        # and 5 degree tolerance for order=4.
        if order == 3:
            angle_tol_err = 1 * pi / 180
        else:
            angle_tol_err = 5 * pi / 180

        detection_rate = np.count_nonzero(min_mean_angle_diff < angle_tol_err) / n_ims
        self.assertTrue(detection_rate > 0.90)

    @parameterized.expand([(order,), (order + 1,)])
    def testCommonLines(self, order):
        n_ims = self.n_ims
        src = self.srcs[order]
        cl_symm = self.cl_classes[order]

        # Build common-lines matrix.
        cl_symm.build_clmatrix()
        cl = cl_symm.clmatrix

        # Compare common-line indices with ground truth angles.
        rots = src.rots  # ground truth rotations
        rots_symm = self.buildCyclicRotations(order)
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
        rots_symm = self.buildCyclicRotations(order)

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

        return volume

    def buildSelfCommonLinesMatrix(self, rots, order):
        # Construct rotatation matrices associated with cyclic order.
        rots_symm = self.buildCyclicRotations(order)

        # Build ground truth self-common-lines matrix.
        scl_gt = np.zeros((self.n_ims, 2), dtype=self.dtype)
        n_theta = self.n_theta
        g = rots_symm[1]
        g_n = rots_symm[order - 1]
        for i in range(self.n_ims):
            Ri = rots[i]

            U1 = Ri.T @ g @ Ri
            U2 = Ri.T @ g_n @ Ri

            c1 = np.array([-U1[1, 2], U1[0, 2]])
            c2 = np.array([-U2[1, 2], U2[0, 2]])

            theta_g = np.arctan2(c1[1], c1[0]) % (2 * np.pi)
            theta_gn = np.arctan2(c2[1], c2[0]) % (2 * np.pi)

            scl_gt[i, 0] = np.round(theta_g * n_theta / (2 * np.pi)) % n_theta
            scl_gt[i, 1] = np.round(theta_gn * n_theta / (2 * np.pi)) % n_theta

        return scl_gt

    def buildCyclicRotations(self, order):
        # Construct rotatation matrices associated with cyclic order.
        angles = np.zeros((order, 3), dtype=self.dtype)
        angles[:, 2] = 2 * np.pi * np.arange(order) / order
        rots_symm = Rotation.from_euler(angles).matrices

        return rots_symm
