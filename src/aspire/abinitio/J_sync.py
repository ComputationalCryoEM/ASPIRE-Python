import logging

import numpy as np
from numpy.linalg import norm

from aspire.utils import J_conjugate, all_pairs, all_triplets, tqdm
from aspire.utils.random import randn

logger = logging.getLogger(__name__)


class JSync:
    """
    Class for handling J-synchronization methods.
    """

    def __init__(
        self,
        n,
        epsilon=1e-2,
        max_iters=1000,
        seed=None,
    ):
        """
        Initialize JSync object for estimating global handedness synchronization for a
        set of relative rotations, Rij = Ri @ Rj.T, where i <= j = 0, 1, ..., n.

        :param n: Number of images/rotations.
        :param epsilon: Tolerance for the power method.
        :param max_iters: Maximum iterations for the power method.
        :param seed: Optional seed for power method initial random vector.
        """
        self.n_img = n
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.seed = seed

    def global_J_sync(self, vijs):
        """
        Global J-synchronization of all third row outer products. Given 3x3 matrices vijs, each
        of which might contain a spurious J (ie. vij = J*vi*vj^T*J instead of vij = vi*vj^T),
        we return vijs that all have either a spurious J or not.

        :param vijs: An (n-choose-2)x3x3 array where each 3x3 slice holds an estimate for the corresponding
        outer-product vi*vj^T between the third rows of the rotation matrices Ri and Rj. Each estimate
        might have a spurious J independently of other estimates.

        :return: vijs, all of which have a spurious J or not.
        """

        # Determine relative handedness of vijs.
        sign_ij_J = self.power_method(vijs)

        # Synchronize vijs
        vijs_sync = vijs.copy()
        for i, sign in enumerate(sign_ij_J):
            if sign == -1:
                vijs_sync[i] = J_conjugate(vijs[i])

        return vijs_sync

    def power_method(self, vijs):
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
        logger.info(
            "Initiating power method to estimate J-synchronization matrix eigenvector."
        )
        while itr < max_iters and residual > epsilon:
            itr += 1
            # Note, this appears to need double precision for accuracy in the following division.
            vec_new = self._signs_times_v(vijs, vec).astype(np.float64, copy=False)
            vec_new = vec_new / norm(vec_new)
            residual = norm(vec_new - vec)
            vec = vec_new
            logger.info(
                f"Iteration {itr}, residual {round(residual, 5)} (target {epsilon})"
            )

        # We need only the signs of the eigenvector
        J_sync = np.sign(vec, dtype=vijs.dtype)

        return J_sync

    def sync_viis(self, vijs, viis):
        """
        Given a set of synchronized pairwise outer products vijs, J-synchronize the set of
        outer products viis.

        :param vijs: An (n-choose-2)x3x3 array where each 3x3 slice holds an estimate for the corresponding
        outer-product vi*vj^T between the third rows of the rotation matrices Ri and Rj. Each estimate
        might have a spurious J independently of other estimates.

        :param viis: An n_imgx3x3 array where the i'th slice holds an estimate for the outer product vi*vi^T
        between the third row of matrix Ri and itself. Each estimate might have a spurious J independently
        of other estimates.

        :return: J-synchronized viis.
        """

        # Synchronize viis
        # We use the fact that if v_ii and v_ij are of the same handedness, then v_ii @ v_ij = v_ij.
        # If they are opposite handed then Jv_iiJ @ v_ij = v_ij. We compare each v_ii against all
        # previously synchronized v_ij to get a consensus on the handedness of v_ii.
        _, pairs_to_linear = all_pairs(self.n_img, return_map=True)
        for i in range(self.n_img):
            vii = viis[i]
            vii_J = J_conjugate(vii)
            J_consensus = 0
            for j in range(self.n_img):
                if j < i:
                    idx = pairs_to_linear[j, i]
                    vji = vijs[idx]

                    err1 = norm(vji @ vii - vji)
                    err2 = norm(vji @ vii_J - vji)

                elif j > i:
                    idx = pairs_to_linear[i, j]
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
        return viis

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
        triplets = all_triplets(n_img)
        pairs, pairs_to_linear = all_pairs(n_img, return_map=True)

        # There are 4 possible configurations of relative handedness for each triplet (vij, vjk, vik).
        # 'conjugate' expresses which node of the triplet must be conjugated (True) to achieve synchronization.
        conjugate = np.empty((4, 3), bool)
        conjugate[0] = [False, False, False]
        conjugate[1] = [True, False, False]
        conjugate[2] = [False, True, False]
        conjugate[3] = [False, False, True]

        # 'edges' corresponds to whether conjugation agrees between the pairs (vij, vjk), (vjk, vik),
        # and (vik, vij). True if the pairs are in agreement, False otherwise.
        edges = np.empty((4, 3), bool)
        edges[:, 0] = conjugate[:, 0] == conjugate[:, 1]
        edges[:, 1] = conjugate[:, 1] == conjugate[:, 2]
        edges[:, 2] = conjugate[:, 2] == conjugate[:, 0]

        # The corresponding entries in the J-synchronization matrix are +1 if the pair of nodes agree, -1 if not.
        edge_signs = np.where(edges, 1, -1)

        # For each triplet of nodes we apply the 4 configurations of conjugation and determine the
        # relative handedness based on the condition that vij @ vjk - vik = 0 for synchronized nodes.
        # We then construct the corresponding entries of the J-synchronization matrix with 'edge_signs'
        # corresponding to the conjugation configuration producing the smallest residual for the above
        # condition. Finally, we the multiply the 'edge_signs' by the cooresponding entries of 'vec'.
        v = vijs
        new_vec = np.zeros_like(vec)
        pbar = tqdm(desc="Computing signs_times_v", total=len(triplets))
        for i, j, k in triplets:
            ij = pairs_to_linear[i, j]
            jk = pairs_to_linear[j, k]
            ik = pairs_to_linear[i, k]
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
            new_vec[ik] += s_ij_ik * vec[ij] + s_ik_jk * vec[jk]
            pbar.update()
        pbar.close()

        return new_vec
