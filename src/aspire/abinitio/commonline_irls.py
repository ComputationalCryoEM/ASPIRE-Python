import logging

import numpy as np
from scipy.sparse.linalg import eigsh

from aspire.abinitio import CLOrient3D, CommonlineLUD, CommonlineSDP

logger = logging.getLogger(__name__)


class CommonlineIRLS(CommonlineLUD):
    """
    Define a derived class to estimate 3D orientations using Iteratively Reweighted
    Least Squares (IRLS) as described in the following publication:
    L. Wang, A. Singer, and  Z. Wen, Orientation Determination of Cryo-EM Images Using
    Least Unsquared Deviations, SIAM J. Imaging Sciences, 6, 2450-2483 (2013).
    """

    def __init__(
        self,
        src,
        *,
        num_itrs=10,
        ctype=False,
        eps_weighting=1e-3,
        alpha=2 / 3,
        max_rankZ=None,
        max_rankW=None,
        **kwargs,
    ):
        """
        Initialize a class for estimating 3D orientations using an IRLS-based optimization.

        :param ctype: Constraint type for the optimization:
            - 1: ||G||_2 as a regularization term in the objective.
            - 0: ||G||_2 as a constraint in the optimization.
        :param alpha: Spectral norm constraint for ADMM algorithm. Default is None, which
            does not apply a spectral norm constraint. To apply a spectral norm constraint provide
            a value in the range [2/3, 1), 2/3 is recommended.
        :param tol: Tolerance for convergence. The algorithm stops when conditions reach this threshold.
            Default is 1e-3.
        :param mu: The penalty parameter (or dual variable scaling factor) in the optimization problem.
            Default is 1.
        :param gam: Relaxation factor for updating variables in the algorithm (typically between 1 and 2).
            Default is 1.618.
        :param eps: Small positive value used to filter out negligible eigenvalues.
            Default is 1e-12.
        :param maxit: Maximum number of iterations allowed for the algorithm.
            Default is 1000.
        :param adp_proj: Flag for using adaptive projection during eigenvalue computation:
            - True: Adaptive rank selection (Default).
            - False: Full eigenvalue decomposition.
        :param max_rankZ: Maximum rank used for projecting the Z matrix (for adaptive projection).
            If None, defaults to max(6, n_img // 4).
        :param max_rankW: Maximum rank used for projecting the W matrix (for adaptive projection).
            If None, defaults to max(6, n_img // 4).
        :param adp_mu: Adaptive adjustment of the penalty parameter `mu`:
            - True: Enabled (Default).
            - False: Disabled.
        :param dec_mu: Scaling factor for decreasing `mu`.
            Default is 0.5.
        :param inc_mu: Scaling factor for increasing `mu`.
            Default is 2.
        :param mu_min: Minimum allowable value for `mu`.
            Default is 1e-4.
        :param mu_max: Maximum allowable value for `mu`.
            Default is 1e4.
        :param min_mu_itr: Minimum number of iterations before `mu` is adjusted.
            Default is 5.
        :param max_mu_itr: Maximum number of iterations allowed for `mu` adjustment.
            Default is 20.
        :param delta_mu_l: Lower bound for relative drop ratio to trigger a decrease in `mu`.
            Default is 0.1.
        :param delta_mu_u: Upper bound for relative drop ratio to trigger an increase in `mu`.
            Default is 10.
        """

        self.num_itrs = num_itrs
        self.ctype = ctype
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

        self.lambda_ = self.alpha * self.n_img  # Spectral norm bound

    def estimate_rotations(self):
        """
        Estimate rotation matrices using the common lines method with LUD optimization.
        """
        logger.info("Computing the common lines matrix.")
        self.build_clmatrix()

        self.S = CommonlineSDP._construct_S(self, self.clmatrix)
        weights = np.ones(2 * self.n_img, dtype=self.dtype)
        gram = np.eye(2 * self.n_img, dtype=self.dtype)
        for itr in range(self.num_itrs):
            S = weights * self.S
            gram = self._compute_Gram(gram, S)
            weights = self._update_weights(gram)
        self.rotations = CommonlineSDP._deterministic_rounding(gram)

        return self.rotations

    def _compute_Gram(self, G, S):
        """
        Perform the alternating direction method of multipliers (ADMM) for the SDP
        problem:

        min sum_{i<j} ||c_ij - G_ij c_ji||
        s.t. A(G) = b, G psd
            ||G||_2 <= lambda

        Equivalent to matlab functions cryoEMSDPL12N/cryoEMSDPL12N_vsimple.
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
            # if self.ITR == 2:
            #     breakpoint()
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
                V = V[:, eigs_mask]
                W = V @ np.diag(D[eigs_mask]) @ V.T
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
            if self.ctype:
                spG = eigsh(G.astype(np.float64, copy=False), k=1, which="LM")
                pobj = -np.sum(S * G) + self.lambda_
                dobj = b.T @ y
            else:
                pobj = -np.sum(S * G)
                dobj = (b.T @ y) - self.lambda_ * np.sum(abs(eigs_Z))

            gap = abs(dobj - pobj) / (1 + abs(dobj) + abs(pobj))

            resi = self._compute_AX(G) - b
            pinf = np.linalg.norm(resi) / max(np.linalg.norm(b), 1)

            dinf_term = S + W + ATy - Z
            dinf = np.linalg.norm(dinf_term, "fro") / max(np.linalg.norm(S, np.inf), 1)

            logger.info(
                f"Iteration: {itr}, residual: {max(pinf, dinf)}, target: {self.tol}"
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

    def _compute_Z(self, S, W, ATy, G, eigs_Z, num_eigs_Z, num_eigs):
        """
        Update ADMM subproblem for enforcing the spectral norm constraint.

        :param S: A 2*n_img x 2*n_img symmetric matrix representing the fidelity term.
        :param W: A 2*n_img x 2*n_img array, primary ADMM subproblem matrix.
        :param ATy: A 2*n_img x 2*n_img array.
        :param G: Current value of the 2*n_img x 2*n_img optimization solution matrix.
        :param eigs_Z: eigenvalues from previous iteration.
        :param num_eigs_Z: Number of eigenvalues of Z to use to enforce spectral norm constraint.
        :param num_eigs: Number of eigenvalues of W used in previous iteration of ADMM.

        :returns:
            - Z, Updated 2*n_img x 2*n_img matrix for spectral norm constraint ADMM subproblem.
            - num_eigs_Z, Number of eigenvalues of Z to use to enforce spectral norm constraint in next iteration.
            - num_eigs, Number of eigenvalues of W to use in this iteration of ADMM.
        """
        B = S + W + ATy + G / self._mu
        B = (B + B.T) / 2

        if not self.adp_proj:
            U, pi = np.linalg.eigh(B)
        else:
            # Determine number of eigenvalues to compute for adaptive projection
            if num_eigs_Z is None:
                num_eigs_Z = self.max_rankZ
            else:
                num_eigs_Z = self._compute_num_eigs(num_eigs_Z, eigs_Z, num_eigs, 10, 3)

            pi, U = eigsh(
                B.astype(np.float64, copy=False),
                k=num_eigs_Z,
                which="LM",
            )  # Compute top `num_eigs_Z` eigenvalues and eigenvectors

            # Sort by eigenvalue magnitude. Note, eigsh does not return
            # ordered eigenvalues/vectors for which="LM".
            idx = np.argsort(np.abs(pi))[::-1]
            pi = pi[idx]
            U = U[:, idx]

        # Apply soft-threshold to eigenvalues to enforce spectral norm constraint.
        # Compute eigenvalues based on constraint type.
        if self.ctype:
            pass
            # Need to make this branch work. Compute projection onto simplex.
            # eigs_Z = projsplx(np.abs(pi) / self.lambda_)
            # eigs_Z = (self.lambda_ * np.sign(pi)) * eigs_Z
        else:
            eigs_Z = np.sign(pi) * np.maximum(np.abs(pi) - self.lambda_ / self._mu, 0)

        nD = abs(eigs_Z) > 0
        num_eigs_Z = np.count_nonzero(nD)
        if num_eigs_Z > 0:
            eigs_Z = eigs_Z[nD]
            Z = U[:, nD] @ np.diag(eigs_Z) @ U[:, nD].T
        else:
            Z = np.zeros_like(B)

        return Z, num_eigs_Z, eigs_Z

    def _update_weights(self, gram):
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
