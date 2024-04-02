import logging

import numpy as np
from numpy.linalg import norm

from aspire.abinitio import CLOrient3D, SyncVotingMixin
from aspire.utils import J_conjugate, all_pairs, all_triplets, nearest_rotations
from aspire.utils.matlab_compat import stable_eigsh
from aspire.utils.random import randn

logger = logging.getLogger(__name__)


class CLSync3N(CLOrient3D, SyncVotingMixin):
    """
    Define a class to estimate 3D orientations using common lines Sync3N methods (2017).
    """

    def __init__(
        self,
        src,
        n_rad=None,
        n_theta=None,
        max_shift=0.15,
        shift_step=1,
        epsilon=1e-2,
        max_iters=1000,
        degree_res=1,
        seed=None,
        mask=True,
    ):
        """
        Initialize object for estimating 3D orientations.

        :param src: The source object of 2D denoised or class-averaged images with metadata
        :param n_rad: The number of points in the radial direction
        :param n_theta: The number of points in the theta direction
        :param max_shift: Maximum range for shifts as a proportion of resolution. Default = 0.15.
        :param shift_step: Resolution of shift estimation in pixels. Default = 1 pixel.
        :param epsilon: Tolerance for the power method.
        :param max_iter: Maximum iterations for the power method.
        :param degree_res: Degree resolution for estimating in-plane rotations.
        :param seed: Optional seed for RNG.
        :param mask: Option to mask `src.images` with a fuzzy mask (boolean).
            Default, `True`, applies a mask.
        """

        super().__init__(
            src,
            n_rad=n_rad,
            n_theta=n_theta,
            max_shift=max_shift,
            shift_step=shift_step,
            mask=mask,
        )

        self.epsilon = epsilon
        self.max_iters = max_iters
        self.degree_res = degree_res
        self.seed = seed

    ###########################################
    # High level algorithm steps              #
    ###########################################
    def estimate_rotations(self):
        """
        Estimate rotation matrices.

        :return: Array of rotation matrices, size n_imgx3x3.
        """

        # Initial estimate of viewing directions
        Rij0 = self._estimate_relative_viewing_directions()

        # Compute and apply global handedness
        Rij = self._global_J_sync(Rij0)

        # Build sync3n matrix
        S = self._construct_sync3n_matrix(Rij)

        # Optionally S weights
        # todo

        # Yield rotations from S
        Ris = self._sync3n_S_to_rot(S)

        self.rotations = Ris

    ###########################################
    # The hackberries taste like hackberries  #
    ###########################################
    def _sync3n_S_to_rot(self, S, n_eigs=4):
        """
        Use eigen decomposition of S to estimate transforms,
        then project transforms to nearest rotations.
        """

        if n_eigs < 3:
            raise ValueError(
                f"n_eigs must be greater than 3, default is 4. Invoked with {n_eigs}"
            )

        # Extract three eigenvectors corresponding to non-zero eigenvalues.
        d, v = stable_eigsh(S, n_eigs)
        sort_idx = np.argsort(-d)
        logger.info(
            f"Top {n_eigs} eigenvalues from synchronization voting matrix: {d[sort_idx]}"
        )

        # Only need the top 3 eigen-vectors.
        v = v[:, sort_idx[:3]]

        # Yield estimated rotations from the eigen-vectors
        v = v.reshape(3, self.n_img, 3)
        rotations = np.transpose(v, (1, 0, 2))  # Check, may be (1, 2 , 0) for T

        # Enforce we are returning actual rotations
        rotations = nearest_rotations(rotations)

        return rotations

    def _construct_sync3n_matrix(self, Rij):
        """
        Construct sync3n matrix from estimated rotations Rij.
        """

        # Initialize S with diag identity blocks
        n = self.n_img
        S = np.eye(3 * n, dtype=self.dtype).reshape(n, 3, n, 3)

        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                # S( (3*i-2):(3*i) , (3*j-2):(3*j) ) = Rij(:,:,idx); % Rij
                S[i, :, j, :] = Rij[idx]
                # S( (3*j-2):(3*j) , (3*i-2):(3*i) ) = Rij(:,:,idx)'; % Rji = Rij'
                S[j, :, i, :] = Rij[idx].T
                idx += 1

        # Convert S shape to 3Nx3N
        S = S.reshape(3 * n, 3 * n)

        return S

    ###########################################
    # Primary Methods                         #
    ###########################################

    def _estimate_relative_viewing_directions(self):
        """
        Estimate the relative viewing directions vij = vi*vj^T, i<j, and vii = vi*vi^T, where
        vi is the third row of the i'th rotation matrix Ri.
        """
        logger.info(f"Estimating relative viewing directions for {self.n_img} images.")
        # Detect a single pair of common-lines between each pair of images
        self.build_clmatrix()

        # Calculate relative rotations
        Rijs = self._estimate_all_Rijs_c3_c4(self.clmatrix)

        return Rijs

    def _global_J_sync(self, vijs):
        """ """

        # Determine relative handedness of vijs.
        sign_ij_J = self._J_sync_power_method(vijs)

        # Synchronize vijs
        logger.info("Applying global handedness synchronization.")
        for i, sign in enumerate(sign_ij_J):
            if sign == -1:
                vijs[i] = J_conjugate(vijs[i])

        return vijs

    def _estimate_all_Rijs_c3_c4(self, clmatrix):
        """
        Estimate Rijs using the voting method.
        """
        n_img = self.n_img
        n_theta = self.n_theta
        pairs = all_pairs(n_img)
        Rijs = np.zeros((len(pairs), 3, 3))

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
            # This is for the case that images i and j correspond to the same
            # viewing direction and differ only by in-plane rotation.
            # We set to zero as in the Matlab code.
            rot_mean = np.zeros((3, 3))

        return rot_mean

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

        logger.info(
            "Initiating power method to estimate J-synchronization matrix eigenvector."
        )
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
                f"Iteration {itr}, residual {round(residual, 5)} (target {epsilon})"
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
            new_vec[ik] += s_ij_jk * vec[ij] + s_ik_jk * vec[jk]

        return new_vec
