from unittest import TestCase

import numpy as np
from numpy import pi, random
from numpy.linalg import norm
from parameterized import parameterized

from aspire.abinitio import CLSymmetryC3C4
from aspire.source import Simulation
from aspire.utils import Rotation
from aspire.utils.misc import J_conjugate, all_pairs, gaussian_3d
from aspire.volume import Volume


class OrientSymmTestCase(TestCase):
    # `order` is at this scope to be picked up by parameterized in testCommonLines.
    order = 3

    def setUp(self):
        self.L = 64
        self.n_img = 32
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
                n=self.n_img,
                offsets=np.zeros((self.n_img, 2)),
                dtype=self.dtype,
                vols=self.vols[o],
                C=1,
            )

            self.cl_classes[o] = CLSymmetryC3C4(
                self.srcs[o], symmetry=f"C{o}", n_theta=self.n_theta
            )

    def tearDown(self):
        pass

    @parameterized.expand([(order,), (order + 1,)])
    def testRelativeRotations(self, order):
        n_img = self.n_img

        # Simulation source and common lines Class corresponding to
        # volume with C3 or C4 symmetry.
        src = self.srcs[order]
        cl_symm = self.cl_classes[order]

        # Estimate relative viewing directions.
        cl_symm.build_clmatrix()
        cl = cl_symm.clmatrix
        Rijs = cl_symm._estimate_all_Rijs_c3_c4(cl)

        # Each Rij belongs to the set {Ri.Tg_n^sRj, JRi.Tg_n^sRjJ},
        # s = 1, 2, ..., order. We find the mean squared error over
        # the minimum error between Rij and the above set.
        rots_symm = self.buildCyclicRotations(order)
        gs = rots_symm
        J = np.diag([-1, -1, 1])
        rots_gt = src.rots

        nchoose2 = int(n_img * (n_img - 1) / 2)
        min_idx = np.zeros(nchoose2, dtype=int)
        errs = np.zeros(nchoose2)
        diffs = np.zeros(order)
        pairs = all_pairs(n_img)
        for idx, (i, j) in enumerate(pairs):
            Rij = Rijs[idx]
            Ri_gt = rots_gt[i]
            Rj_gt = rots_gt[j]
            for s in range(order):
                Rij_s_gt = Ri_gt.T @ gs[s] @ Rj_gt
                diffs[s] = np.minimum(
                    norm(Rij - Rij_s_gt), norm(J @ Rij @ J - Rij_s_gt)
                )
            min_idx[idx] = np.argmin(diffs)
            errs[idx] = diffs[min_idx[idx]]

        mse = np.mean(errs**2)

        # Mean-squared-error is better for C3 than for C4.
        if order == 3:
            self.assertTrue(mse < 0.005)
        else:
            self.assertTrue(mse < 0.03)

    @parameterized.expand([(order,), (order + 1,)])
    def testSelfRelativeRotations(self, order):
        n_img = self.n_img

        # Simulation source and common lines Class corresponding to
        # volume with C3 or C4 symmetry.
        src = self.srcs[order]
        cl_symm = self.cl_classes[order]

        # Estimate self-relative viewing directions, Riis.
        scl, _, _ = cl_symm._self_clmatrix_c3_c4()
        Riis = cl_symm._estimate_all_Riis_c3_c4(scl)

        # Each estimated Rii belongs to the set
        # {Ri.Tg_nRi, Ri.Tg_n^{n-1}Ri, JRi.Tg_nRiJ, JRi.Tg_n^{n-1}RiJ}
        # We find the minimum mean-squared-error over the 4 possibilities.
        rots_symm = self.buildCyclicRotations(order)
        g = rots_symm[1]
        J = np.diag([-1, -1, 1])
        rots_gt = src.rots

        min_idx = np.zeros(n_img, dtype=int)
        errs = np.zeros(n_img)
        for i, rot_gt in enumerate(rots_gt):
            Rii_gt = rot_gt.T @ g @ rot_gt
            Rii = Riis[i]

            diff0 = norm(Rii - Rii_gt)
            diff1 = norm(Rii.T - Rii_gt)
            diff2 = norm((J @ Rii @ J) - Rii_gt)
            diff3 = norm((J @ Rii.T @ J) - Rii_gt)
            diffs = [diff0, diff1, diff2, diff3]
            min_idx[i] = np.argmin(diffs)
            errs[i] = diffs[min_idx[i]]

        mse = np.mean(errs**2)

        # Mean-squared-error is better for C3 than for C4.
        if order == 3:
            self.assertTrue(mse < 0.0035)
        else:
            self.assertTrue(mse < 0.025)

    @parameterized.expand([(order,), (order + 1,)])
    def testRelativeViewingDirections(self, order):
        n_img = self.n_img

        # Simulation source and common lines Class corresponding to
        # volume with C3 or C4 symmetry.
        src = self.srcs[order]
        cl_symm = self.cl_classes[order]

        # Calculate ground truth relative viewing directions, viis and vijs.
        rots_gt = src.rots

        viis_gt = np.zeros((n_img, 3, 3))
        for i in range(n_img):
            vi = rots_gt[i, 2]
            viis_gt[i] = np.outer(vi, vi)

        pairs = all_pairs(n_img)
        n_pairs = len(pairs)
        vijs_gt = np.zeros((n_pairs, 3, 3))
        for idx, (i, j) in enumerate(pairs):
            vi = rots_gt[i, 2]
            vj = rots_gt[j, 2]
            vijs_gt[idx] = np.outer(vi, vj)

        # Estimate relative viewing directions.
        vijs, viis = cl_symm._estimate_relative_viewing_directions_c3_c4()

        # Calculate the mean squared error for vijs.
        errs_vijs = np.zeros(n_pairs)
        diffs_vijs = np.zeros(2)
        for idx, (vij_gt, vij) in enumerate(zip(vijs_gt, vijs)):
            diffs_vijs[0] = norm(vij - vij_gt)
            diffs_vijs[1] = norm(J_conjugate(vij) - vij_gt)
            errs_vijs[idx] = diffs_vijs[np.argmin(diffs_vijs)]
        mse_vijs = np.mean(errs_vijs**2)

        # Calculate the mean squared error for viis.
        errs_viis = np.zeros(n_img)
        diffs_viis = np.zeros(2)
        for idx, (vii_gt, vii) in enumerate(zip(viis_gt, viis)):
            diffs_viis[0] = norm(vii - vii_gt)
            diffs_viis[1] = norm(J_conjugate(vii) - vii_gt)
            errs_viis[idx] = diffs_viis[np.argmin(diffs_viis)]
        mse_viis = np.mean(errs_viis**2)

        # Check that MSE is small.
        # MSE is better for C3 than C4.
        if order == 3:
            self.assertTrue(mse_vijs < 0.0016)
            self.assertTrue(mse_viis < 0.0011)
        else:
            self.assertTrue(mse_vijs < 0.021)
            self.assertTrue(mse_viis < 0.012)

    def testGlobalJSync(self):
        n_img = self.n_img

        # Build a set of outer products of random third rows.
        vijs, viis, _ = self.buildOuterProducts(n_img)

        # J-conjugate some of these outer products (every other element).
        vijs_conj, viis_conj = vijs.copy(), viis.copy()
        vijs_conj[::2] = J_conjugate(vijs_conj[::2])
        viis_conj[::2] = J_conjugate(viis_conj[::2])

        # Synchronize vijs_conj and viis_conj.
        vijs_sync, viis_sync = self.cl_classes[3]._global_J_sync(vijs_conj, viis_conj)

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
        vis = self.cl_classes[3]._estimate_third_rows(vijs, viis)

        # Check if all-close up to difference of sign
        ground_truth = np.sign(gt_vis[0, 0]) * gt_vis
        estimate = np.sign(vis[0, 0]) * vis
        self.assertTrue(np.allclose(ground_truth, estimate))

    @parameterized.expand([(order,), (order + 1,)])
    def testSelfCommonLines(self, order):
        n_img = self.n_img
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

        detection_rate = np.count_nonzero(min_mean_angle_diff < angle_tol_err) / n_img
        self.assertTrue(detection_rate > 0.90)

    @parameterized.expand([(order,), (order + 1,)])
    def testCommonLines(self, order):
        n_img = self.n_img
        src = self.srcs[order]
        cl_symm = self.cl_classes[order]

        # Build common-lines matrix.
        cl_symm.build_clmatrix()
        cl = cl_symm.clmatrix

        # Compare common-line indices with ground truth angles.
        rots = src.rots  # ground truth rotations
        rots_symm = self.buildCyclicRotations(order)
        pairs = all_pairs(n_img)
        within_1_degree = 0
        within_5_degrees = 0
        for (i, j) in pairs:
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

    def buildOuterProducts(self, n_img):
        # Build random third rows, ground truth vis (unit vectors)
        gt_vis = np.zeros((n_img, 3), dtype=np.float32)
        for i in range(n_img):
            random.seed(i)
            v = random.randn(3)
            gt_vis[i] = v / norm(v)

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
        scl_gt = np.zeros((self.n_img, 2), dtype=self.dtype)
        n_theta = self.n_theta
        g = rots_symm[1]
        g_n = rots_symm[order - 1]
        for i in range(self.n_img):
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
