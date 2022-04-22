import logging

import numpy as np
from numpy.linalg import eigh, norm

from aspire.abinitio import CLOrient3D
from aspire.utils import J_conjugate, all_pairs, all_triplets, anorm
from aspire.utils.random import randn

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
            self.order = symmetry[1]
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

    def _compute_third_row_outer_prod_c34(self, max_shift_1d):
        """
        Compute the outer products of the third rows of the rotation matrices Rij and Rii.
        """

        pf = self.pf
        order = self.order
        max_shift = self.max_shift
        shift_step = self.shift_step
        n_theta = self.n_theta

        # Step 1: Detect a single pair of common-lines between each pair of images
        self.build_clmatrix()
        clmatrix = self.clmatrix

        # Step 2: Detect self-common-lines in each image
        sclmatrix = self.self_clmatrix_c3_c4(pf, order, max_shift, shift_step)

        # Step 3: Calculate self-relative-rotations
        Riis = self._estimate_all_Riis_c3_c4(order, sclmatrix, n_theta)

        # Step 4: Calculate relative rotations
        Rijs = self._estimate_all_Rijs_c3_c4(order, clmatrix, n_theta)

        # Step 5: Inner J-synchronization
        vijs, viis = self._local_sync_J_c3_c4(order, Rijs, Riis)

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

    def _estimate_inplane_rotations(
        self, pf, vis, inplane_rot_res, max_shift, shift_step
    ):
        # return rots
        pass

    #################################################
    # Secondary Methods for computing outer product #
    #################################################

    def _self_clmatrix_c3_c4(self, pf, order, max_shift, shift_step):
        # return sclmatrix
        pass

    def _estimate_all_Riis_c3_c4(order, sclmatrix, n_theta):
        # return Riis
        pass

    def _estimate_all_Rijs_c3_c4(order, clmatrix, n_theta):
        # return Rijs
        pass

    def local_sync_J_c3_c4(order, Rijs, Riis):
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
        Multiplication of the J-synchronization matrix by a candidate eigenvector.

        The J-synchronization matrix is a matrix representation of the handedness graph, Gamma, whose set of
        nodes consists of the estimates vijs and whose set of edges consists of the undirected edges between
        all triplets of estimates vij, vjk, and vik, where i<j<k. The weight of an edge is set to +1 if its
        incident nodes agree in handednes and -1 if not.

        The J-synchronization matrix is of size (n-choose-2)x(n-choose-2), where each entry corresponds to
        the relative handedness of vij and vjk. The entry (ij, jk), where ij and jk are retrieved from the
        all_pairs indexing, is 1 if vij and vjk are of the same handedness and -1 if not. All other entries
        (ij, kl) hold a zero.

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

        # Under the 4 possible configurations of the relative handedness of each triplet (vij, vjk, vik),
        # we determine which nodes to conjugate to achieve handedness synchronization.
        conjugate = np.empty((4, 3), bool)
        conjugate[0] = [False, False, False]
        conjugate[1] = [True, False, False]
        conjugate[2] = [False, True, False]
        conjugate[3] = [False, False, True]

        edges = np.empty((4, 3), bool)
        edges[:, 0] = conjugate[:, 0] == conjugate[:, 1]
        edges[:, 1] = conjugate[:, 1] == conjugate[:, 2]
        edges[:, 2] = conjugate[:, 2] == conjugate[:, 0]

        edge_signs = np.where(edges, 1, -1)

        v = vijs
        new_vec = np.zeros_like(vec)
        for (i, j, k) in triplets:
            ij = pairs.index((i, j))
            jk = pairs.index((j, k))
            ik = pairs.index((i, k))
            vij, vjk, vik = v[ij], v[jk], v[ik]
            vij_J = J_conjugate(vij)
            vjk_J = J_conjugate(vjk)
            vik_J = J_conjugate(vik)

            conjugated_pairs = np.where(
                conjugate[..., np.newaxis, np.newaxis],
                [vij_J, vjk_J, vik_J],
                [vij, vjk, vik],
            )
            residual = np.stack([norm(x @ y - z) for x, y, z in conjugated_pairs])

            min_residual = np.argmin(residual)

            # Assign edge weights
            s_ij_jk, s_ik_jk, s_ij_ik = edge_signs[min_residual]

            # Update multiplication of signs times vec
            new_vec[ij] += s_ij_jk * vec[jk] + s_ij_ik * vec[ik]
            new_vec[jk] += s_ij_jk * vec[ij] + s_ik_jk * vec[ik]
            new_vec[ik] += s_ij_jk * vec[ij] + s_ik_jk * vec[jk]

        return new_vec
