import logging

import numpy as np
from numpy import pi
from numpy.linalg import eigh, norm, svd
from tqdm import tqdm

from aspire.abinitio import CLSyncVoting
from aspire.utils import J_conjugate, Rotation, all_pairs, all_triplets, anorm
from aspire.utils.random import randn

logger = logging.getLogger(__name__)


class CLSymmetryC3C4(CLSyncVoting):
    """
    Define a class to estimate 3D orientations using common lines methods for molecules with
    C3 and C4 cyclic symmetry.

    The related publications are listed below:
    G. Pragier and Y. Shkolnisky,
    A Common Lines Approach for Abinitio Modeling of Cyclically Symmetric Molecules,
    Inverse Problems, 35, 124005, (2019).

    Y. Shkolnisky, and A. Singer,
    Viewing Direction Estimation in Cryo-EM Using Synchronization,
    SIAM J. Imaging Sciences, 5, 1088-1110 (2012).

    A. Singer, R. R. Coifman, F. J. Sigworth, D. W. Chester, Y. Shkolnisky,
    Detecting Consistent Common Lines in Cryo-EM by Voting,
    Journal of Structural Biology, 169, 312-322 (2010).
    """

    def __init__(
        self,
        src,
        symmetry=None,
        n_rad=None,
        n_theta=None,
        epsilon=1e-3,
        max_iters=1000,
        seed=None,
    ):
        """
        Initialize object for estimating 3D orientations for molecules with C3 and C4 symmetry.

        :param src: The source object of 2D denoised or class-averaged images with metadata
        :param symmetry: A string, 'C3' or 'C4', indicating the symmetry type.
        :param n_rad: The number of points in the radial direction
        :param n_theta: The number of points in the theta direction
        :param epsilon: Tolerance for the power method.
        :param max_iter: Maximum iterations for the power method.
        :param seed: Optional seed for RNG.
        """

        super().__init__(src, n_rad=n_rad, n_theta=n_theta)

        if symmetry is None:
            raise NotImplementedError(
                "Symmetry type not supplied. Please indicate C3 or C4 symmetry."
            )
        else:
            symmetry = symmetry.upper()
            if symmetry not in ["C3", "C4"]:
                raise NotImplementedError(
                    f"Only C3 and C4 symmetry supported. {symmetry} was supplied."
                )
            self.order = int(symmetry[1])
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.seed = seed

    def estimate_rotations(self):
        """
        Estimate rotation matrices for symmetric molecules.
        """
        pass

    ###########################################
    # Primary Methods                         #
    ###########################################

    def _estimate_relative_viewing_directions_c3_c4(self):
        """
        Estimate the relative viewing directions vij = vi'vj, i<j, and vii = vi'vi, where
        vi is the third row of the i'th rotation matrix Ri.
        """

        # Step 1: Detect a single pair of common-lines between each pair of images
        if self.clmatrix is None:
            self.build_clmatrix()

        clmatrix = self.clmatrix

        # Step 2: Detect self-common-lines in each image
        sclmatrix, corrs_stats, shifts_stats = self._self_clmatrix_c3_c4()

        # Step 3: Calculate self-relative-rotations
        Riis = self._estimate_all_Riis_c3_c4(sclmatrix)

        # Step 4: Calculate relative rotations
        Rijs = self._estimate_all_Rijs_c3_c4(clmatrix)

        # Step 5: Inner J-synchronization
        vijs, viis = self._local_J_sync_c3_c4(Rijs, Riis)

        return vijs, viis

    def _global_J_sync(self, vijs, viis):
        """
        Global J-synchronization of all third row outer products. Given 3x3 matrices vijs and viis, each
        of which might contain a spurious J (ie. vij = J*vi*vj^T*J instead of vij = vi*vj^T),
        we return vijs and viis that all have either a spurious J or not.

        :param vijs: An (n-choose-2)x3x3 array where each 3x3 slice holds an estimate for the corresponding
        outer-product vi*vj^T between the third rows of the rotation matrices Ri and Rj. Each estimate
        might have a spurious J independently of other estimates.

        :param viis: An n_imgx3x3 array where the i'th slice holds an estimate for the outer product vi*vi^T
        between the third row of matrix Ri and itself. Each estimate might have a spurious J independently
        of other estimates.

        :return: vijs, viis all of which have a spurious J or not.
        """
        n_img = self.n_img

        # Determine relative handedness of vijs.
        sign_ij_J = self._J_sync_power_method(vijs)

        # Synchronize vijs
        for i, sign in enumerate(sign_ij_J):
            if sign == -1:
                vijs[i] = J_conjugate(vijs[i])

        # Synchronize viis
        # We use the fact that if v_ii and v_ij are of the same handedness, then v_ii @ v_ij = v_ij.
        # If they are opposite handed then Jv_iiJ @ v_ij = v_ij. We compare each v_ii against all
        # previously synchronized v_ij to get a consensus on the handedness of v_ii.

        # All pairs (i,j) where i<j
        pairs = all_pairs(n_img)
        for i in range(n_img):
            vii = viis[i]
            vii_J = J_conjugate(vii)
            J_consensus = 0
            for j in range(n_img):
                if j < i:
                    idx = pairs.index((j, i))
                    vji = vijs[idx]

                    err1 = norm(vji @ vii - vji)
                    err2 = norm(vji @ vii_J - vji)

                elif j > i:
                    idx = pairs.index((i, j))
                    vij = vijs[idx]

                    err1 = norm(vii @ vij - vij)
                    err2 = norm(vii_J @ vij - vij)

                else:
                    continue

                # Accumulate J consensus
                if err1 < err2:
                    J_consensus -= 1
                else:
                    J_consensus += 1

            if J_consensus > 0:
                viis[i] = vii_J
        return vijs, viis

    def _estimate_third_rows(self, vijs, viis):
        """
        Find the third row of each rotation matrix given a collection of matrices
        representing the outer products of the third rows from each rotation matrix.

        :param vijs: An (n-choose-2)x3x3 array where each 3x3 slice holds the third rows
        outer product of the rotation matrices Ri and Rj.

        :param viis: An n_imgx3x3 array where the i'th 3x3 slice holds the outer product of
        the third row of Ri with itself.

        :param order: The underlying molecular symmetry.

        :return: vis, An n_imgx3 matrix whose i'th row is the third row of the rotation matrix Ri.
        """

        n_img = self.n_img

        # Build matrix V whose (i,j)-th block of size 3x3 holds the outer product vij
        V = np.zeros((n_img, n_img, 3, 3), dtype=vijs.dtype)

        # All pairs (i,j) where i<j
        pairs = all_pairs(n_img)

        # Populate upper triangle of V with vijs and lower triangle with vjis, where vji = vij^T.
        for idx, (i, j) in enumerate(pairs):
            V[i, j] = vijs[idx]
            V[j, i] = vijs[idx].T

        # Populate diagonal of V with viis
        for i, vii in enumerate(viis):
            V[i, i] = vii

        # Permute axes and reshape to (3 * n_img, 3 * n_img).
        V = np.swapaxes(V, 1, 2).reshape(3 * n_img, 3 * n_img)

        # In a clean setting V is of rank 1 and its eigenvector is the concatenation
        # of the third rows of all rotation matrices.
        # In the noisy setting we use the eigenvector corresponding to the leading eigenvalue
        val, vec = eigh(V)
        lead_idx = np.argmax(val)
        lead_vec = vec[:, lead_idx]

        # We decompose the leading eigenvector and normalize to obtain the third rows, vis.
        vis = lead_vec.reshape((n_img, 3))
        vis /= anorm(vis, axes=(-1,))[:, np.newaxis]

        return vis

    def _estimate_inplane_rotations(self, vis):
        """
        Estimate the rotation matrices for each image by constructing arbitrary rotation matrices
        populated with the given third rows, vis, and then rotating by an appropriate in-plane rotation.

        :param vis: An n_imgx3 array where the i'th row holds the estimate for the third row of
        the i'th rotation matrix.

        :return: An n_imgx3x3 array holding the n_img estimated rotation matrices.
        """
        # n_img = self.n_img
        # n_theta = self.n_theta
        # max_shift_1d = np.ceil(2 * np.sqrt(2) * self.max_shift)
        # shift_step = self.shift_step
        # order = self.order

        # Step 1: Construct all rotation matrices Ri_tildes whose third rows are equal to
        # the corresponding third rows vis.

        pass

    #################################################
    # Secondary Methods for computing outer product #
    #################################################

    def _self_clmatrix_c3_c4(self):
        """
        Find the single pair of self-common-lines in each image assuming that the underlying
        symmetry is C3 or C4.
        """
        pf = self.pf.copy()
        n_img = self.n_img
        n_theta = self.n_theta
        max_shift_1d = np.ceil(2 * np.sqrt(2) * self.max_shift)
        shift_step = self.shift_step
        order = self.order
        assert order in [3, 4], f"order must be 3 or 4. Got order: {order}."

        # The angle between self-common-lines is in the range [60, 180] for C3 symmetry
        # and [90, 180] for C4 symmetry. Since antipodal lines are perfectly correlated
        # we search for common lines in a smaller window.
        # Note: matlab code used [60, 165] for order=3 and [90, 160] for order=4.
        # We've increased this window to improve the MSE of our estimates.
        if order == 3:
            min_angle_diff = 60 * pi / 180
            max_angle_diff = 175 * pi / 180
        else:
            min_angle_diff = 90 * pi / 180
            max_angle_diff = 175 * pi / 180

        # The self-common-lines matrix holds two indices per image that represent
        # the two self common-lines in the image.
        sclmatrix = np.zeros((n_img, 2))
        corrs_stats = np.zeros(n_img)
        shifts_stats = np.zeros(n_img)

        # We create a mask associated with angle differences that fall in the
        # range [min_angle_diff, max_angle_diff].
        X, Y = np.meshgrid(range(n_theta), range(n_theta // 2))
        diff = Y - X
        unsigned_angle_diff = np.arccos(np.cos(diff * 2 * pi / n_theta))
        good_diffs = np.logical_and(
            min_angle_diff < unsigned_angle_diff, unsigned_angle_diff < max_angle_diff
        )

        # Compute the correlation over all shifts.
        # Generate Shifts.
        r_max = pf.shape[0]
        shifts, shift_phases, _ = self._generate_shift_phase_and_filter(
            r_max, max_shift_1d, shift_step
        )
        n_shifts = len(shifts)
        all_shift_phases = shift_phases.T

        # Transpose pf and reconstruct the full polar Fourier for use in correlation.
        # self.pf only consists of rays in the range [180, 360).
        pf = pf.transpose((2, 1, 0))
        pf_full = np.concatenate((pf, np.conj(pf)), axis=1)

        for i in tqdm(range(n_img)):
            pf_i = pf[i]
            pf_full_i = pf_full[i]

            # Generate shifted versions of images.
            pf_i_shifted = np.array(
                [pf_i * shift_phase for shift_phase in all_shift_phases]
            )
            pf_i_shifted = np.reshape(pf_i_shifted, (-1, r_max))

            # Normalize each ray.
            for ray in pf_full_i:
                ray /= norm(ray)
            for ray in pf_i_shifted:
                ray /= norm(ray)

            # Compute correlation.
            corrs = np.dot(pf_i_shifted, pf_full_i.T)
            corrs = np.reshape(corrs, (n_shifts, n_theta // 2, n_theta))

            # Mask with allowed combinations.
            corrs = np.array([corr * good_diffs for corr in corrs])

            # Find maximum correlation.
            shift, scl1, scl2 = np.unravel_index(np.argmax(np.real(corrs)), corrs.shape)
            sclmatrix[i] = [scl1, scl2]
            corrs_stats[i] = np.real(corrs[(shift, scl1, scl2)])
            shifts_stats[i] = shift

        return sclmatrix, corrs_stats, shifts_stats

    def _estimate_all_Riis_c3_c4(self, sclmatrix):
        """
        Compute estimates for the self relative rotations Rii for every rotation matrix Ri.
        """

        order = self.order
        n_theta = self.n_theta

        # Calculate the cosine of angle between self-common-lines.
        cos_diff = np.cos((sclmatrix[:, 1] - sclmatrix[:, 0]) * 2 * np.pi / n_theta)

        # Calculate Euler angle gamma.
        if order == 3:
            # cos_diff should be <= 0.5, but due to discretization that might be violated.
            if np.max(cos_diff) > 0.5:
                bad_diffs = np.count_nonzero([cos_diff > 0.5])
                logger.warning(
                    "cos(angular_diff) should be < 0.5."
                    f"Found {bad_diffs} estimates exceeding 0.5, with maximum {np.max(cos_diff)}"
                )

                cos_diff[cos_diff > 0.5] = 0.5
            gammas = np.arccos(cos_diff / (1 - cos_diff))

        else:
            # cos_diff should be <= 0, but due to discretization that might be violated.
            if np.max(cos_diff) > 0:
                bad_diffs = np.count_nonzero([cos_diff > 0])
                logger.warning(
                    "cos(angular_diff) should be < 0."
                    f"Found {bad_diffs} estimates exceeding 0, with maximum {np.max(cos_diff)}"
                )

                cos_diff[cos_diff > 0.5] = 0.5
            gammas = np.arccos((1 + cos_diff) / (1 - cos_diff))

        # Calculate remaining Euler angles in ZYZ convention.
        # Note: Publication uses ZXZ convention.
        alphas = sclmatrix[:, 0] * 2 * np.pi / n_theta + np.pi / 2
        betas = sclmatrix[:, 1] * 2 * np.pi / n_theta - np.pi / 2

        # Compute Riis from Euler angles.
        angles = np.array((alphas, gammas, -betas), dtype=self.dtype).T
        Riis = Rotation.from_euler(angles, dtype=self.dtype).matrices

        return Riis

    def _estimate_all_Rijs_c3_c4(self, clmatrix):
        """
        Estimate Rijs using the voting method.
        """
        n_img = self.n_img
        n_theta = self.n_theta

        nchoose2 = int(n_img * (n_img - 1) / 2)
        Rijs = np.zeros((nchoose2, 3, 3))
        pairs = all_pairs(n_img)
        for idx, (i, j) in enumerate(pairs):
            Rijs[idx] = self._syncmatrix_ij_vote_3n(
                clmatrix, i, j, np.arange(n_img), n_theta
            )

        return Rijs

    def _syncmatrix_ij_vote_3n(self, clmatrix, i, j, k_list, n_theta):
        """
        Compute the (i,j) rotation block of the synchronization matrix using voting method

        Given the common lines matrix `clmatrix`, a list of images specified in k_list
        and the number of common lines n_theta, find the (i, j) rotation block Rij.
        :param clmatrix: The common lines matrix
        :param i: The i image
        :param j: The j image
        :param k_list: The list of images for the third image for voting algorithm
        :param n_theta: The number of points in the theta direction (common lines)
        :return: The (i,j) rotation block of the synchronization matrix
        """

        good_k = self._vote_ij(clmatrix, n_theta, i, j, k_list)

        rots = self._rotratio_eulerangle_vec(clmatrix, i, j, good_k, n_theta)

        if rots is not None:
            rot_mean = np.mean(rots, 0)

        else:
            # This for the case that images i and j correspond to the same
            # viewing direction and differ only by in-plane rotation.
            # Simply put to zero as Matlab code.
            rot_mean = np.zeros((3, 3))

        return rot_mean

    def _local_J_sync_c3_c4(self, Rijs, Riis):
        """
        Estimate viis and vijs. In order to estimate vij = vi @ vj.T, it is necessary for Rii, Rjj,
        and Rij to be of the same handedness. We perform a local handedness synchronization and
        set vij = 1/n ∑ Rii^s @ Rij @ Rjj^s.

        :param Rijs: An n-choose-2x3x3 array of estimates of relative rotations
            (each pair of images induces two estimates).
        :param Riis: A nx3x3 array of estimates of self-relative rotations.
        :return: vijs, viis
        """

        order = self.order
        n_img = self.n_img

        nchoose2 = int(n_img * (n_img - 1) / 2)
        assert (
            len(Riis) == n_img
        ), f"There must be one self-relative rotation per image. Got {len(Riis)} Riis."
        assert (
            len(Rijs) == nchoose2
        ), f"There must be n-choose-2 relative rotations. Got {len(Rijs)}."

        # Estimate viis from Riis. vii = 1/order * (∑ Rii ** s) for s = 0, 1, ..., order.
        viis = np.zeros((n_img, 3, 3))
        for i, Rii in enumerate(Riis):
            viis[i] = np.mean(
                [np.linalg.matrix_power(Rii, s) for s in np.arange(order)], axis=0
            )

        # Estimate vijs via local handedness synchronization.
        vijs = np.zeros((nchoose2, 3, 3))
        e1 = [1, 0, 0]
        J = np.diag((-1, -1, 1))
        opts = np.zeros((8, 3, 3))
        scores_rank1 = np.zeros(8)
        min_idxs = np.zeros((nchoose2, 3, 3))
        pairs = all_pairs(n_img)
        for idx, (i, j) in enumerate(pairs):
            Rii = Riis[i]
            Rjj = Riis[j]
            Rij = Rijs[idx]

            Rii_J = J @ Rii @ J
            Rjj_J = J @ Rjj @ J

            # vij should be a singular matrix.
            # We test 8 combinations of handedness and rotation by {g, g^n-1} for singularity to determine:
            # a. whether to transpose Rii
            # b. whether to J-conjugate Rii
            # c. whether to J-conjugate Rjj
            if order == 3:
                opts[0] = Rij + (Rii @ Rij @ Rjj) + (Rii.T @ Rij @ Rjj.T)
                opts[1] = Rij + (Rii_J @ Rij @ Rjj) + (Rii_J.T @ Rij @ Rjj.T)
                opts[2] = Rij + (Rii @ Rij @ Rjj_J) + (Rii.T @ Rij @ Rjj_J.T)
                opts[3] = Rij + (Rii_J @ Rij @ Rjj_J) + (Rii_J.T @ Rij @ Rjj_J.T)

                opts[4] = Rij + (Rii.T @ Rij @ Rjj) + (Rii @ Rij @ Rjj.T)
                opts[5] = Rij + (Rii_J.T @ Rij @ Rjj) + (Rii_J @ Rij @ Rjj.T)
                opts[6] = Rij + (Rii.T @ Rij @ Rjj_J) + (Rii @ Rij @ Rjj_J.T)
                opts[7] = Rij + (Rii_J.T @ Rij @ Rjj_J) + (Rii_J @ Rij @ Rjj_J.T)

                # Normalize
                opts = opts / 3

            else:
                opts[0] = Rij + (Rii @ Rij @ Rjj)
                opts[1] = Rij + (Rii_J @ Rij @ Rjj)
                opts[2] = Rij + (Rii @ Rij @ Rjj_J)
                opts[3] = Rij + (Rii_J @ Rij @ Rjj_J)

                opts[4] = Rij + (Rii.T @ Rij @ Rjj)
                opts[5] = Rij + (Rii_J.T @ Rij @ Rjj)
                opts[6] = Rij + (Rii.T @ Rij @ Rjj_J)
                opts[7] = Rij + (Rii_J.T @ Rij @ Rjj_J)

                # Normalize
                opts = opts / 2

            for k, opt in enumerate(opts):
                _, svals, _ = svd(opt)
                scores_rank1[k] = norm(svals - e1, 2)
            min_idx = np.argmin(scores_rank1)
            min_idxs[idx] = min_idx

            vijs[idx] = opts[min_idx]

        return vijs, viis

    #######################################
    # Secondary Methods for Global J Sync #
    #######################################

    def _J_sync_power_method(self, vijs):
        """
        Calculate the leading eigenvector of the J-synchronization matrix
        using the power method.

        As the J-synchronization matrix is of size (n-choose-2)x(n-choose-2), we
        use the power method to compute the eigenvalues and eigenvectors,
        while constructing the matrix on-the-fly.

        :param vijs: (n-choose-2)x3x3 array of estimates of relative orientation matrices.

        :return: An array of length n-choose-2 consisting of 1 or -1, where the sign of the
        i'th entry indicates whether the i'th relative orientation matrix will be J-conjugated.
        """

        # Set power method tolerance and maximum iterations.
        epsilon = self.epsilon
        max_iters = self.max_iters

        # Initialize candidate eigenvectors
        n_vijs = vijs.shape[0]
        vec = randn(n_vijs, seed=self.seed)
        vec = vec / norm(vec)
        residual = 1
        itr = 0

        # Power method iterations
        while itr < max_iters and residual > epsilon:
            itr += 1
            vec_new = self._signs_times_v(vijs, vec)
            vec_new = vec_new / norm(vec_new)
            residual = norm(vec_new - vec)
            vec = vec_new

        logger.info(
            f"Power method used {itr} iterations. Maximum iterations set to {max_iters}."
        )

        # We need only the signs of the eigenvector
        J_sync = np.sign(vec)

        return J_sync

    def _signs_times_v(self, vijs, vec):
        """
        We multiply the J-synchronization matrix by a candidate eigenvector. The J-synchronization matrix is of
        size (n-choose-2)x(n-choose-2), where each entry corresponds to the relative handedness of vij and vjk.
        The entry (ij, jk), where ij and jk are retrieved from the all_pairs indexing, is 1 if vij and vjk are
        of the same handedness and -1 if not. All other entries (ij, kl) hold a zero.

        Due to the large size of the J-synchronization matrix we construct it on the fly as follows.
        For each triplet of outer products vij, vjk, and vik, the associated elements of the J-synchronization
        matrix are populated with +1 or -1 and multiplied by the corresponding elements of
        the current candidate eigenvector supplied by the power method. The new candidate eigenvector
        is updated for each triplet.

        :param vijs: (n-choose-2)x3x3 array, where each 3x3 slice holds the outer product of vi and vj.

        :param vec: The current candidate eigenvector of length n-choose-2 from the power method.

        :return: New candidate eigenvector of length n-choose-2. The product of the J-sync matrix and vec.
        """

        # All pairs (i,j) and triplets (i,j,k) where i<j<k
        n_img = self.n_img
        pairs = all_pairs(n_img)
        triplets = all_triplets(n_img)

        # For each triplet of nodes (vij, vjk, vik) there are four possible configurations
        # for the corresponding entries of the J-synchronization matrix, dependent upon
        # whether any of the three nodes must be J-conjugated to achieve synchronization.
        signs = np.zeros((4, 3))
        signs[0] = [1, 1, 1]  # All nodes are synchronized.
        signs[1] = [-1, 1, -1]  # vij must be J-conjugated.
        signs[2] = [-1, -1, 1]  # vjk must be J-conjugated.
        signs[3] = [1, -1, -1]  # vik must be J-conjugated.

        v = vijs
        new_vec = np.zeros_like(vec)

        # For each triplet (vij, vjk, vik) we test the conditions for relative handedness, c,
        # and populate the entries of the J-synchronization matrix with corresponding signs.
        for (i, j, k) in triplets:
            ij = pairs.index((i, j))
            jk = pairs.index((j, k))
            ik = pairs.index((i, k))
            vij, vjk, vik = v[ij], v[jk], v[ik]
            vij_J = J_conjugate(vij)
            vjk_J = J_conjugate(vjk)
            vik_J = J_conjugate(vik)

            # Conditions for relative handedness. The minimum of these conditions determines
            # the relative handedness of the triplet of vijs.
            c = np.zeros(4)
            c[0] = norm(vij @ vjk - vik)
            c[1] = norm(vij_J @ vjk - vik)
            c[2] = norm(vij @ vjk_J - vik)
            c[3] = norm(vij @ vjk - vik_J)

            min_c = np.argmin(c)

            # Assign signs +-1 to edges between nodes vij, vik, vjk.
            s_ij_jk, s_ik_jk, s_ij_ik = signs[min_c]

            # Update multiplication of signs times vec
            new_vec[ij] += s_ij_jk * vec[jk] + s_ij_ik * vec[ik]
            new_vec[jk] += s_ij_jk * vec[ij] + s_ik_jk * vec[ik]
            new_vec[ik] += s_ij_jk * vec[ij] + s_ik_jk * vec[jk]

        return new_vec

    ###########################################
    # Secondary Methods fo In-Plane Rotations #
    ###########################################
    @staticmethod
    def complete_third_row_to_rot(r3):
        """
        Construct a rotation matrix whose thrid row is equal to the given row vector.

        :param r3: A 1x3 vector of norm 1.
        :return: A 3x3 rotation matrix whose third row is r3.
        """

        # If the third row coincides with the z-axis we return the identity matrix.
        if norm(r3 - [0, 0, 1]) < 1e-5:
            return np.eye(3)

        # tmp is non-zero since r3 does not coincide with the z-axis.
        tmp = np.sqrt(r3[0] ** 2 + r3[1] ** 2)

        # Construct an orthogonal row vector of norm 1.
        r1 = np.array([r3[1] / tmp, -r3[0] / tmp, 0])

        # Construct r2 so that r3 = r1xr2
        r2 = np.array([r3[0] * r3[2] / tmp, r3[1] * r3[2] / tmp, -tmp])

        return np.vstack((r1, r2, r3))
