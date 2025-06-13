import logging

import numpy as np
from scipy.sparse.linalg import eigsh

from aspire.abinitio import CommonlineLUD

logger = logging.getLogger(__name__)


class CommonlineIRLS(CommonlineLUD):
    """
    Estimate 3D orientations using Iteratively Reweighted Least Squares
    (IRLS) as described in the following publication:
    L. Wang, A. Singer, and  Z. Wen, Orientation Determination of Cryo-EM Images Using
    Least Unsquared Deviations, SIAM J. Imaging Sciences, 6, 2450-2483 (2013).
    """

    def __init__(
        self,
        src,
        *,
        num_itrs=10,
        eps_weighting=1e-3,
        alpha=None,
        max_rankZ=None,
        max_rankW=None,
        **kwargs,
    ):
        """
        Initialize a class for estimating 3D orientations using an IRLS-based optimization.
        See CommonlineLUD for additional arguments.

        :param num_itrs: Number of iterations for iterative reweighting. Default is 10.
        :param eps_weighting: Regularization value for reweighting factor. Default is 1e-3.
        :param alpha: Spectral norm constraint for IRLS algorithm. Default is None, which
            does not apply a spectral norm constraint. To apply a spectral norm constraint provide
            a value in the range [2/3, 1), 2/3 is recommended.
        :param max_rankZ: Maximum rank used for projecting the Z matrix (for adaptive projection).
            If None, defaults to max(6, n_img // 4).
        :param max_rankW: Maximum rank used for projecting the W matrix (for adaptive projection).
            If None, defaults to max(6, n_img // 4).
        """

        self.num_itrs = num_itrs
        self.eps_weighting = eps_weighting

        # Adjust rank limits
        max_rankZ = max_rankZ or max(6, src.n // 4)
        max_rankW = max_rankW or max(6, src.n // 4)

        super().__init__(
            src,
            max_rankZ=max_rankZ,
            max_rankW=max_rankW,
            alpha=alpha,
            **kwargs,
        )

    def estimate_rotations(self):
        """
        Estimate rotation matrices using the common lines method with IRLS optimization.
        """
        logger.info("Computing the common lines matrix.")
        self.build_clmatrix()

        self.S = self._construct_S(self.clmatrix)
        weights = np.ones(2 * self.n_img, dtype=self.dtype)
        gram = np.eye(2 * self.n_img, dtype=self.dtype)
        if self.alpha is None:
            A, b = self._sdp_prep()
            for _ in range(self.num_itrs):
                S_weighted = weights * self.S
                gram = self._compute_gram_matrix(S_weighted, A, b)
                weights = self._update_weights(gram)
        else:
            for _ in range(self.num_itrs):
                S_weighted = weights * self.S
                gram = self._compute_Gram(gram, S_weighted)
                weights = self._update_weights(gram)
        self.rotations = self._deterministic_rounding(gram)

        return self.rotations

    def _compute_Gram(self, G, S):
        """
        Perform the alternating direction method of multipliers (ADMM) for the SDP
        problem:

        min sum_{i<j} ||c_ij - G_ij c_ji||
        s.t. A(G) = b, G psd
            ||G||_2 <= lambda

        Equivalent to matlab function cryoEMADM.
        :param G: Gram matrix from previous iteration.
        :param S: Reweighted S matrix.
        :return: The updated gram matrix G.
        """
        logger.info("Performing ADMM to compute Gram matrix.")

        # Initialize problem parameters
        self._mu = self.mu
        n = 2 * self.n_img
        b = np.concatenate(
            [np.ones(n, dtype=self.dtype), np.zeros(self.n_img, dtype=self.dtype)]
        )

        # Initialize variables
        W = np.eye(n, dtype=self.dtype)
        Z = np.eye(n, dtype=self.dtype)

        # Compute initial values
        AS = self._compute_AX(S)
        resi = self._compute_AX(G) - b

        num_eigs = 0
        itmu_pinf = 0
        itmu_dinf = 0
        eigs_Z = 0
        num_eigs_Z = None
        eigs_H = 0
        for itr in range(self.maxit):
            #############
            # Compute y #
            #############
            y = -(AS + self._compute_AX(W) - self._compute_AX(Z)) - resi / self._mu

            #############
            # Compute Z #
            #############
            ATy = self._compute_ATy(y)
            Z, num_eigs_Z, eigs_Z = self._compute_Z(
                S, W, ATy, G, eigs_Z, num_eigs_Z, num_eigs
            )

            #############
            # Compute W #
            #############
            H = Z - S - ATy - G / self._mu
            H = (H + H.T) / 2

            if not self.adp_proj:
                D, V = np.linalg.eigh(H)
                eigs_mask = D > self.eps
                num_eigs = np.count_nonzero(eigs_mask)
                if num_eigs < n / 2:  # few positive eigenvalues
                    V = V[:, eigs_mask]
                    W = V @ np.diag(D[eigs_mask]) @ V.T
                else:  # few negative eigenvalues
                    V = V[:, ~eigs_mask]
                    W = V @ np.diag(-D[~eigs_mask]) @ V.T + H
            else:
                # Determine number of eigenvalues to compute for adaptive projection
                if itr == 0:
                    num_eigs = self.max_rankW
                else:
                    num_eigs = self._compute_num_eigs(num_eigs, eigs_H, num_eigs, 50, 5)

                # Compute Eigenvectors and sort by largest algebraic eigenvalue
                eigs_H, V = eigsh(
                    -H.astype(np.float64),
                    k=num_eigs,
                    which="LA",
                    tol=1e-15,
                )
                eigs_H = eigs_H[::-1].astype(self.dtype, copy=False)
                V = V[:, ::-1].astype(self.dtype, copy=False)

                nD = eigs_H > self.eps
                eigs_H = eigs_H[nD]
                num_eigs = np.count_nonzero(nD)
                W = V[:, nD] @ np.diag(eigs_H) @ V[:, nD].T + H if nD.any() else H

            ############
            # Update G #
            ############
            G = (1 - self.gam) * G + self.gam * self._mu * (W - H)

            # Check optimality
            pobj = -np.sum(S * G)
            dobj = (b.T @ y) - self.lambda_ * np.sum(abs(eigs_Z))

            gap = abs(dobj - pobj) / (1 + abs(dobj) + abs(pobj))

            resi = self._compute_AX(G) - b
            pinf = np.linalg.norm(resi) / max(np.linalg.norm(b), 1)

            dinf_term = S + W + ATy - Z
            dinf = np.linalg.norm(dinf_term, "fro") / max(np.linalg.norm(S, np.inf), 1)

            logger.info(
                f"Iteration: {itr}, residual: {max(pinf, dinf, gap)}, target: {self.tol}"
            )
            if max(pinf, dinf, gap) <= self.tol:
                return G

            # Update mu adaptively
            if self.adp_mu:
                if pinf / dinf <= self.delta_mu_l:
                    itmu_pinf = itmu_pinf + 1
                    itmu_dinf = 0
                    if itmu_pinf > self.max_mu_itr:
                        self._mu = max(self._mu * self.inc_mu, self.mu_min)
                        itmu_pinf = 0
                elif pinf / dinf > self.delta_mu_u:
                    itmu_dinf = itmu_dinf + 1
                    itmu_pinf = 0
                    if itmu_dinf > self.max_mu_itr:
                        self._mu = min(self._mu * self.dec_mu, self.mu_max)
                        itmu_dinf = 0
        return G

    def _update_weights(self, gram):
        """
        Update the weight matrix for the IRLS algorithm.

        :param gram: 2K x 2K Gram matrix.
        :return: 2K x 2K updated weight matrix.
        """
        K = self.n_img
        W = self.S * gram
        weights = W[:K, :K] + W[:K, K:] + W[K:, :K] + W[K:, K:]
        weights = np.sqrt(abs(2 - 2 * weights), dtype=self.dtype)
        weights = 1 / np.sqrt(weights**2 + self.eps_weighting**2)
        updated_W = np.block([[weights, weights], [weights, weights]])

        return updated_W

    @staticmethod
    def _compute_AX(X):
        """
        Compute the application of the linear operator A to the symmetric input
        matrix X, where A(X) is defined as:

        A(X) = [
            diag(X_11),
            diag(X_22),
            sqrt(2)/2 * diag(X_12 + sqrt(2)/2 * diag(X_21)
        ]

        where X_{ij} is the (i, j)'th K x K sub-block of X.

        :param X: 2D square array of shape (2K, 2K)..
        :return: Flattened array representing A(X)
        """
        K = X.shape[0] // 2

        # Extract the diagonal elements of (X_ii^(11) and X_ii^(22))
        diags = np.diag(X)

        # Extract the diagonal elements from upper right KxK sub-block.
        upper_right_diag_vals = np.diag(X[K:, :K])

        # Compute the second part of AX, which is sqrt(2)/2 times the sum of
        # the diagonal entries of the KxK off-diagonal sub-blocks of X.
        sqrt_2_X_col = np.sqrt(2, dtype=X.dtype) * upper_right_diag_vals

        # Form AX by concatenating the results.
        AX = np.concatenate((diags, sqrt_2_X_col))

        return AX

    @staticmethod
    def _compute_ATy(y):
        """
        Compute the application of the adjoint operator A^T to the input vector y,
        where

            y = [
                y_i^1,
                y_i^2,
                y_i^3
            ]   for i = 1, 2, ..., K,

        and the adjoint of the operator A is defined as:

            AT(y) = Y = [
                [Y_ii^(11), Y_ii^(12)],
                [Y_ii^(21), Y_ii^(22)]
            ],

        where for i = 1, 2, ..., K:

            Y_ii^(11) = y_i^1,
            Y_ii^(22) = y_i^2,
            Y_ii^(12) = Y_ii^(21) = y_i^3 / sqrt(2).

        :param y: 1D NumPy array of length 3K.
        :return: 2D NumPy array of shape (2K, 2K).
        """
        K = len(y) // 3
        n = 2 * K  # Size of the output matrix
        ATy = np.zeros((n, n), dtype=y.dtype)

        # Assign diagonal elements
        ATy[::1, ::1] = np.diag(y[:n])

        # Assign symmetric off-diagonal elements
        off_diag_vals = np.sqrt(2, dtype=y.dtype) / 2 * y[n:]
        ATy[:K, K:] = np.diag(off_diag_vals)
        ATy[K:, :K] = np.diag(off_diag_vals)

        return ATy
