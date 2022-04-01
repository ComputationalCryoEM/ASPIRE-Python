import logging

import numpy as np
from numpy.linalg import eig, norm

from aspire.abinitio import CLOrient3D

logger = logging.getLogger(__name__)


class CLSymmetryC3C4(CLOrient3D):
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
        self, src, n_symm=None, n_rad=None, n_theta=None, epsilon=1e-3, max_iters=1000
    ):
        """
        Initialize object for estimating 3D orientations for molecules with C3 and C4 symmetry.

        :param src: The source object of 2D denoised or class-averaged images with metadata
        :param n_symm: The symmetry order of the molecule. 3 or 4.
        :param n_rad: The number of points in the radial direction
        :param n_theta: The number of points in the theta direction
        :param epsilon: Tolerance for the power method.
        :param max_iter: Maximum iterations for the power method.
        """

        super().__init__(src, n_rad=n_rad, n_theta=n_theta)

        self.n_symm = n_symm
        self.epsilon = epsilon
        self.max_iters = max_iters

    def estimate_rotations(self):
        """
        Estimate rotation matrices for symmetric molecules.
        """
        pass

    ###########################################
    # Primary Methods                         #
    ###########################################

    def _compute_third_row_outer_prod_c34(self, max_shift_1d):
        """
        Compute the outer products of the third rows of the rotation matrices Rij and Rii.
        """

        pf = self.pf
        n_symm = self.n_symm
        max_shift = self.max_shift
        shift_step = self.shift_step
        n_theta = self.n_theta

        # Step 1: Detect a single pair of common-lines between each pair of images
        self.build_clmatrix()
        clmatrix = self.clmatrix

        # Step 2: Detect self-common-lines in each image
        sclmatrix = self.self_clmatrix_c3_c4(pf, n_symm, max_shift, shift_step)

        # Step 3: Calculate self-relative-rotations
        Riis = self._estimate_all_Riis_c3_c4(n_symm, sclmatrix, n_theta)

        # Step 4: Calculate relative rotations
        Rijs = self._estimate_all_Rijs_c3_c4(n_symm, clmatrix, n_theta)

        # Step 5: Inner J-synchronization
        vijs, viis = self._local_sync_J_c3_c4(n_symm, Rijs, Riis)

        return vijs, viis

    def _global_J_sync(self, vijs, viis):
        """
        Global J-synchronization of all third row outer products. Given 3x3 matrices vijs and viis, each
        of which might contain a spurious J, we return vijs and viis that all have either a spurious J
        or not.

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
        J = np.diag((-1, -1, 1))
        for i, sign in enumerate(sign_ij_J):
            if sign == -1:
                vijs[i] = J @ vijs[i] @ J

        # Synchronize viis
        # We use the fact that if v_ii and v_ij are of the same handedness, then v_ii @ v_ij = v_ij.
        # If they are opposite handed then Jv_iiJ @ v_ij = v_ij. We compare each v_ii against all
        # previously synchronized v_ij to get a consensus on the handedness of v_ii.

        # All pairs (i,j) where i<j
        pairs = [(i, j) for i in range(n_img) for j in range(n_img) if i < j]
        for i in range(n_img):
            vii = viis[i]
            J_consensus = 0
            for j in range(n_img):
                if j < i:
                    idx = pairs.index((j, i))
                    vji = vijs[idx]

                    err1 = norm(vji @ vii - vji)
                    err2 = norm(vji @ J @ vii @ J - vji)

                elif j > i:
                    idx = pairs.index((i, j))
                    vij = vijs[idx]

                    err1 = norm(vii @ vij - vij)
                    err2 = norm(J @ vii @ J @ vij - vij)

                else:
                    continue

                # Accumulate J consensus
                if err1 < err2:
                    J_consensus -= 1
                else:
                    J_consensus += 1

            if J_consensus > 0:
                viis[i] = J @ viis[i] @ J
        return vijs, viis

    def _estimate_third_rows(self, vijs, viis):
        """
        Find the third row of each rotation matrix given a collection of matrices
        representing the outer products of the third rows from each rotation matrix.

        :param vijs: An (n-choose-2)x3x3 array where each 3x3 slice holds the third rows
        outer product of the rotation matrices Ri and Rj.

        :param viis: An n_imgx3x3 array where the i'th 3x3 slice holds the outer product of
        the third row of Ri with itself.

        :param n_symm: The underlying molecular symmetry.

        :return: vis, An n_imgx3 matrix whose i'th row is the third row of the rotation matrix Ri.
        """

        n_img = self.n_img

        # Build 3nx3n matrix V whose (i,j)-th block of size 3x3 holds the outer product vij
        V = np.zeros((3 * n_img, 3 * n_img), dtype=vijs.dtype)

        # All pairs (i,j) where i<j
        pairs = [(i, j) for i in range(n_img) for j in range(n_img) if i < j]

        # Populate upper triangle of V with vijs
        for idx, (i, j) in enumerate(pairs):
            V[3 * i : 3 * (i + 1), 3 * j : 3 * (j + 1)] = vijs[idx]

        # Populate lower triangle of V with vjis, where vji = vij^T
        V = V + V.T

        # Populate diagonal of V with viis
        for i in range(n_img):
            V[3 * i : 3 * (i + 1), 3 * i : 3 * (i + 1)] = viis[i]

        # In a clean setting V is of rank 1 and its eigenvector is the concatenation
        # of the third rows of all rotation matrices.
        # In the noisy setting we use the eigenvector corresponding to the leading eigenvalue
        val, vec = eig(V)
        lead_idx = np.argsort(val)[-1]
        lead_vec = vec[:, lead_idx]

        # We decompose the leading eigenvector and normalize to obtain the third rows, vis.
        vis = lead_vec.reshape((n_img, 3))
        for i in range(n_img):
            vis[i] = vis[i] / norm(vis[i])

        return vis

    def _estimate_inplane_rotations(
        self, pf, vis, inplane_rot_res, max_shift, shift_step
    ):
        # return rots
        pass

    #################################################
    # Secondary Methods for computing outer product #
    #################################################

    def _self_clmatrix_c3_c4(self, pf, n_symm, max_shift, shift_step):
        # return sclmatrix
        pass

    def _estimate_all_Riis_c3_c4(n_symm, sclmatrix, n_theta):
        # return Riis
        pass

    def _estimate_all_Rijs_c3_c4(n_symm, clmatrix, n_theta):
        # return Rijs
        pass

    def local_sync_J_c3_c4(n_symm, Rijs, Riis):
        # return vijs, viis
        pass

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
        vec = np.random.randn(n_vijs)
        vec = vec / norm(vec)
        dd = 1
        itr = 0

        # Power method iterations
        while itr < max_iters and dd > epsilon:
            itr += 1
            vec_new = self._signs_times_v(vijs, vec)
            vec_new = vec_new / norm(vec_new)
            dd = norm(vec_new - vec)
            vec = vec_new

        logger.info(
            f"Power method used {itr} iterations. Maximum iterations set to {max_iters}."
        )

        # We need only the signs of the eigenvector
        J_sync = np.sign(vec)

        return J_sync

    def _signs_times_v(self, vijs, vec):
        """
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
        pairs = [(i, j) for i in range(n_img) for j in range(n_img) if i < j]
        trips = [
            (i, j, k)
            for i in range(n_img)
            for j in range(n_img)
            for k in range(n_img)
            if i < j < k
        ]
        # There are four possible signs configurations for each triplet of nodes vij, vik, vjk.
        signs = np.zeros((4, 3))
        signs[0] = [1, 1, 1]
        signs[1] = [-1, 1, -1]
        signs[2] = [-1, -1, 1]
        signs[3] = [1, -1, -1]

        J = np.diag((-1, -1, 1))
        v = vijs
        new_vec = np.zeros_like(vec)

        for (i, j, k) in trips:
            ij = pairs.index((i, j))
            jk = pairs.index((j, k))
            ik = pairs.index((i, k))

            # Conditions for relative handedness. The minimum of these conditions determines
            # the relative handedness of the triplet of vijs.
            c = np.zeros(4)
            c[0] = norm(v[ij] @ v[jk] - v[ik])
            c[1] = norm(J @ v[ij] @ J @ v[jk] - v[ik])
            c[2] = norm(v[ij] @ J @ v[jk] @ J - v[ik])
            c[3] = norm(v[ij] @ v[jk] - J @ v[ik] @ J)

            min_c = np.argmin(c)

            # Assign signs +-1 to edges between nodes vij, vik, vjk.
            s_ij_jk, s_ik_jk, s_ij_ik = signs[min_c]

            # Update multiplication of signs times vec
            new_vec[ij] += s_ij_jk * vec[jk] + s_ij_ik * vec[ik]
            new_vec[jk] += s_ij_jk * vec[ij] + s_ik_jk * vec[ik]
            new_vec[ik] += s_ij_jk * vec[ij] + s_ik_jk * vec[jk]

        return new_vec
