from unittest import TestCase

import numpy as np
from numpy import pi, random
from numpy.linalg import det, norm
from parameterized import parameterized

from aspire.abinitio import CLSymmetryC3C4
from aspire.source import Simulation
from aspire.utils import Rotation
from aspire.utils.coor_trans import (
    get_aligned_rotations,
    get_rots_mse,
    register_rotations,
)
from aspire.utils.misc import J_conjugate, all_pairs, cyclic_rotations, gaussian_3d
from aspire.utils.random import randn
from aspire.volume import Volume


class OrientSymmTestCase(TestCase):
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
        self.cl_orient_ests = {}

        for order in orders:
            self.vols[order] = self.buildSimpleSymmetricVolume(self.L, order)

            self.srcs[order] = Simulation(
                L=self.L,
                n=self.n_img,
                offsets=np.zeros((self.n_img, 2)),
                dtype=self.dtype,
                vols=self.vols[order],
                C=1,
                seed=123,
            )

            self.cl_orient_ests[order] = CLSymmetryC3C4(
                self.srcs[order], symmetry=f"C{order}", n_theta=self.n_theta
            )

    def tearDown(self):
        pass

    @parameterized.expand([(3,), (4,)])
    def testEstimateRotations(self, order):
        src = self.srcs[order]
        cl_symm = self.cl_orient_ests[order]

        # Estimate rotations.
        cl_symm.estimate_rotations()
        rots_est = cl_symm.rotations

        # g-synchronize ground truth rotations.
        rots_gt = src.rotations
        rots_gt_sync = cl_symm.g_sync(rots_est, order, rots_gt)

        # Register estimates to ground truth rotations and compute MSE.
        Q_mat, flag = register_rotations(rots_est, rots_gt_sync)
        regrot = get_aligned_rotations(rots_est, Q_mat, flag)
        mse_reg = get_rots_mse(regrot, rots_gt_sync)

        # Assert mse is small.
        self.assertTrue(mse_reg < 0.005)

    @parameterized.expand([(3,), (4,)])
    def testRelativeRotations(self, order):
        n_img = self.n_img

        # Simulation source and common lines estimation instance
        # corresponding to volume with C3 or C4 symmetry.
        src = self.srcs[order]
        cl_symm = self.cl_orient_ests[order]

        # Estimate relative viewing directions.
        cl_symm.build_clmatrix()
        cl = cl_symm.clmatrix
        Rijs = cl_symm._estimate_all_Rijs_c3_c4(cl)

        # Each Rij belongs to the set {Ri.Tg_n^sRj, JRi.Tg_n^sRjJ},
        # s = 1, 2, ..., order. We find the mean squared error over
        # the minimum error between Rij and the above set.
        gs = cyclic_rotations(order, self.dtype).matrices
        rots_gt = src.rotations

        # Find the angular distance between each Rij and the ground truth.
        pairs = all_pairs(n_img)
        angular_distance = np.zeros(len(pairs))
        for idx, (i, j) in enumerate(pairs):
            Rij = Rijs[idx]
            Rij_J = J_conjugate(Rij)
            Ri_gt = rots_gt[i]
            Rj_gt = rots_gt[j]
            dist = np.zeros(order)
            for s in range(order):
                Rij_s_gt = Ri_gt.T @ gs[s] @ Rj_gt
                dist[s] = np.minimum(
                    Rotation.angle_dist(Rij, Rij_s_gt),
                    Rotation.angle_dist(Rij_J, Rij_s_gt),
                )
            angular_distance[idx] = np.min(dist)
        mean_angular_distance = np.mean(angular_distance)

        # Assert that the mean_angular_distance is less than 5 degrees.
        self.assertTrue(mean_angular_distance < 5)

    @parameterized.expand([(3,), (4,)])
    def testSelfRelativeRotations(self, order):
        n_img = self.n_img

        # Simulation source and common lines Class corresponding to
        # volume with C3 or C4 symmetry.
        src = self.srcs[order]
        cl_symm = self.cl_orient_ests[order]

        # Estimate self-relative viewing directions, Riis.
        scl = cl_symm._self_clmatrix_c3_c4()
        Riis = cl_symm._estimate_all_Riis_c3_c4(scl)

        # Each estimated Rii belongs to the set
        # {Ri.Tg_nRi, Ri.Tg_n^{n-1}Ri, JRi.Tg_nRiJ, JRi.Tg_n^{n-1}RiJ}.
        # We find the minimum angular distance between the estimate Rii
        # and the 4 possible ground truths.
        rots_symm = cyclic_rotations(order, self.dtype).matrices
        g = rots_symm[1]
        rots_gt = src.rotations

        # Find angular distance between estimate and ground truth.
        dist = np.zeros(4)
        angular_distance = np.zeros(n_img)
        for i, rot_gt in enumerate(rots_gt):
            Rii_gt = rot_gt.T @ g @ rot_gt
            Rii = Riis[i]
            cases = np.array([Rii, Rii.T, J_conjugate(Rii), J_conjugate(Rii.T)])
            for i, estimate in enumerate(cases):
                dist[i] = Rotation.angle_dist(estimate, Rii_gt)
            angular_distance[i] = dist[np.argmin(dist)]
        mean_angular_distance = np.mean(angular_distance)

        # Check that mean_angular_distance is less than 5 degrees.
        self.assertTrue(mean_angular_distance < 5)

    @parameterized.expand([(3,), (4,)])
    def testRelativeViewingDirections(self, order):
        n_img = self.n_img

        # Simulation source and common lines Class corresponding to
        # volume with C3 or C4 symmetry.
        src = self.srcs[order]
        cl_symm = self.cl_orient_ests[order]

        # Calculate ground truth relative viewing directions, viis and vijs.
        rots_gt = src.rotations

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

        # Since ground truth vijs and viis are rank 1 matrices they span a 1D subspace.
        # We use SVD to find this subspace for our estimates and the ground truth relative viewing directions.
        # We then calculate the angular distance between these subspaces (and take the mean).
        # SVD's:
        uij_gt, _, _ = np.linalg.svd(vijs_gt)
        uii_gt, _, _ = np.linalg.svd(viis_gt)
        uij_est, sij, _ = np.linalg.svd(vijs)
        uii_est, sii, _ = np.linalg.svd(viis)
        uij_J_est, _, _ = np.linalg.svd(J_conjugate(vijs))
        uii_J_est, _, _ = np.linalg.svd(J_conjugate(viis))

        # Ground truth 1D supbspaces.
        uij_gt = uij_gt[:, :, 0]
        uii_gt = uii_gt[:, :, 0]

        # 1D subspace of estimates.
        uij_est = uij_est[:, :, 0]
        uii_est = uii_est[:, :, 0]
        uij_J_est = uij_J_est[:, :, 0]
        uii_J_est = uii_J_est[:, :, 0]

        # Calculate angular distance between subspaces.
        theta_vij = np.arccos(np.sum(uij_gt * uij_est, axis=1))
        theta_vij_J = np.arccos(np.sum(uij_gt * uij_J_est, axis=1))
        theta_vii = np.arccos(np.sum(uii_gt * uii_est, axis=1))
        theta_vii_J = np.arccos(np.sum(uii_gt * uii_J_est, axis=1))

        # Minimum angle between subspaces.
        min_theta_vij = np.min(
            (theta_vij, theta_vij_J, np.pi - theta_vij, np.pi - theta_vij_J), axis=0
        )
        min_theta_vii = np.min(
            (theta_vii, theta_vii_J, np.pi - theta_vii, np.pi - theta_vii_J), axis=0
        )

        # Calculate the mean minimum angular distance.
        angular_dist_vijs = np.mean(min_theta_vij)
        angular_dist_viis = np.mean(min_theta_vii)

        # Check that estimates are indeed approximately rank 1.
        # ie. check that the two smaller singular values are close to zero.
        tol = 5e-2
        self.assertTrue(np.max(sij[:, 1:]) < tol)
        self.assertTrue(np.max(sii[:, 1:]) < tol)

        # Check that the mean angular difference is within 2 degrees.
        angle_tol = 2 * np.pi / 180
        self.assertTrue(angular_dist_vijs < angle_tol)
        self.assertTrue(angular_dist_viis < angle_tol)

    def testGlobalJSync(self):
        n_img = self.n_img

        # Build a set of outer products of random third rows.
        vijs, viis, _ = self.buildOuterProducts(n_img)

        # J-conjugate some of these outer products (every other element).
        vijs_conj, viis_conj = vijs.copy(), viis.copy()
        vijs_conj[::2] = J_conjugate(vijs_conj[::2])
        viis_conj[::2] = J_conjugate(viis_conj[::2])

        # Synchronize vijs_conj and viis_conj.
        # Note: `_global_J_sync()` does not depend on cyclic order, so we can use
        # either cl_orient_ests[3] or cl_orient_ests[4] to access the method.
        vijs_sync, viis_sync = self.cl_orient_ests[3]._global_J_sync(
            vijs_conj, viis_conj
        )

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
        vis = self.cl_orient_ests[3]._estimate_third_rows(vijs, viis)

        # Check if all-close up to difference of sign
        ground_truth = np.sign(gt_vis[0, 0]) * gt_vis
        estimate = np.sign(vis[0, 0]) * vis
        self.assertTrue(np.allclose(ground_truth, estimate))

    @parameterized.expand([(3,), (4,)])
    def testSelfCommonLines(self, order):
        n_theta = self.n_theta
        src = self.srcs[order]
        L = src.L
        cl_symm = self.cl_orient_ests[order]

        # Initialize common-lines orientation estimation object and compute self-common-lines matrix.
        scl = cl_symm._self_clmatrix_c3_c4()

        # Compute ground truth self-common-lines matrix.
        rots = src.rotations
        scl_gt = self.buildSelfCommonLinesMatrix(rots, order)

        # Since we search for self common lines whose angle differences fall
        # outside of 180 degrees by a tolerance of 2 * (360 // L), we must exclude
        # indices whose ground truth self common lines fall within that tolerance.
        gt_diffs = abs(scl_gt[:, 0] - scl_gt[:, 1])
        res = 2 * (360 // L)
        good_indices = (gt_diffs < (180 - res)) | (gt_diffs > (180 + res))
        scl = scl[good_indices]
        scl_gt = scl_gt[good_indices]

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

        # Assert scl detection rate is 100% for 5 degree angle tolerance
        angle_tol_err = 5 * pi / 180
        detection_rate = np.count_nonzero(min_mean_angle_diff < angle_tol_err) / len(
            scl
        )
        self.assertTrue(np.allclose(detection_rate, 1.0))

    @parameterized.expand([(3,), (4,)])
    def testCommonLines(self, order):
        n_img = self.n_img
        src = self.srcs[order]
        cl_symm = self.cl_orient_ests[order]
        n_theta = self.n_theta

        # Build common-lines matrix.
        cl_symm.build_clmatrix()
        cl = cl_symm.clmatrix

        # Compare common-line indices with ground truth angles.
        rots = src.rotations  # ground truth rotations
        rots_symm = cyclic_rotations(order, self.dtype).matrices
        pairs = all_pairs(n_img)
        within_1_degree = 0
        within_5_degrees = 0
        for (i, j) in pairs:
            a_ij_s = np.zeros(order)
            a_ji_s = np.zeros(order)
            # Convert common-line indices to angles. Use angle of common line in [0, 180).
            cl_ij = (cl[i, j] * 360 / n_theta) % 180
            cl_ji = (cl[j, i] * 360 / n_theta) % 180

            # The common-line estimates cl_ij, cl_ji should match the
            # true common-line angles a_ij_s, a_ji_s for some value s,
            # where s is the number of common-lines induced by the symmetric order.
            for s in range(order):
                rel_rot = rots[i].T @ rots_symm[s] @ rots[j]
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

        # Assert that at least 98% of estimates are within 5 degrees and
        # at least 90% of estimates are within 1 degree.
        n_estimates = 2 * len(pairs)
        within_5 = within_5_degrees / n_estimates
        within_1 = within_1_degree / n_estimates
        self.assertTrue(within_5 > 0.98)
        self.assertTrue(within_1 > 0.90)

    def testCompleteThirdRow(self):
        # Complete third row that coincides with z-axis
        z = np.array([0, 0, 1])
        Rz = CLSymmetryC3C4._complete_third_row_to_rot(z)

        # Complete random third row.
        r3 = randn(3, seed=123)
        r3 /= norm(r3)
        R = CLSymmetryC3C4._complete_third_row_to_rot(r3)

        # Assert that Rz is the identity matrix.
        self.assertTrue(np.allclose(Rz, np.eye(3)))

        # Assert that R is orthogonal with determinant 1.
        self.assertTrue(np.allclose(R @ R.T, np.eye(3)))
        self.assertTrue(np.allclose(det(R), 1))

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
        rots_symm = cyclic_rotations(order, self.dtype).matrices

        # Assign centers and sigmas of Gaussian blobs
        centers = np.zeros((3, order, 3), dtype=self.dtype)
        centers[0, 0, :] = np.array([res // 12, res // 12, 0])
        centers[1, 0, :] = np.array([res // 6, res // 6, res // 20])
        centers[2, 0, :] = np.array([res // 4, res // 7, res // 16])
        for o in range(order):
            centers[0, o, :] = rots_symm[o] @ centers[0, 0, :]
            centers[1, o, :] = rots_symm[o] @ centers[1, 0, :]
            centers[2, o, :] = rots_symm[o] @ centers[2, 0, :]
        sigmas = [res / 15, res / 20, res / 30]

        # Build volume
        vol = np.zeros((res, res, res), dtype=self.dtype)
        n_blobs = centers.shape[0]
        for o in range(order):
            for i in range(n_blobs):
                vol += gaussian_3d(res, centers[i, o, :], sigmas[i], indexing="xyz")

        volume = Volume(vol)

        return volume

    def buildSelfCommonLinesMatrix(self, rots, order):
        # Construct rotatation matrices associated with cyclic order.
        rots_symm = cyclic_rotations(order, self.dtype).matrices

        # Build ground truth self-common-lines matrix.
        scl_gt = np.zeros((self.n_img, 2), dtype=self.dtype)
        n_theta = self.n_theta
        g = rots_symm[1]
        g_n = rots_symm[-1]
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
