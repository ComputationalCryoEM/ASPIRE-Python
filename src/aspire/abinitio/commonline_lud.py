import logging

import numpy as np
from scipy.sparse.linalg import eigsh

from aspire.abinitio import CLOrient3D, CommonlineSDP

logger = logging.getLogger(__name__)


class CommonlineLUD(CLOrient3D):
    """
    Define a derived class to estimate 3D orientations using Least Unsquared
    Deviations as described in the following publication:
    L. Wang, A. Singer, and  Z. Wen, Orientation Determination of Cryo-EM Images Using
    Least Unsquared Deviations, SIAM J. Imaging Sciences, 6, 2450-2483 (2013).
    """

    def __init__(
        self,
        src,
        alpha=None,
        tol=1e-3,
        mu=1,
        gam=1.618,
        EPS=1e-12,
        maxit=1000,
        adp_proj=True,
        max_rankZ=None,
        max_rankW=None,
        adp_mu=True,
        dec_mu=0.5,
        inc_mu=2,
        mu_min=1e-4,
        mu_max=1e4,
        min_mu_itr=5,
        max_mu_itr=20,
        delta_mu_l=0.1,
        delta_mu_u=10,
        **kwargs,
    ):
        """
        Initialize a class for estimating 3D orientations using a Least Unsquared Deviations algorithm.

        This class extends the `CLOrient3D` class, inheriting its initialization parameters. Additional
        parameters detailed below.

        :param alpha: Spectral norm constraint for ADMM algorithm. Default is None, which
            does not apply a spectral norm constraint. To apply a spectral norm constraint provide
            a value in the range [2/3, 1), 2/3 is recommended.
        :param tol: Tolerance for convergence. The algorithm stops when conditions reach this threshold.
            Default is 1e-3.
        :param mu: The penalty parameter (or dual variable scaling factor) in the optimization problem.
            Default is 1.
        :param gam: Relaxation factor for updating variables in the algorithm (typically between 1 and 2).
            Default is 1.618.
        :param EPS: Small positive value used to filter out negligible eigenvalues.
            Default is 1e-12.
        :param maxit: Maximum number of iterations allowed for the algorithm.
            Default is 1000.
        :param adp_proj: Flag for using adaptive projection during eigenvalue computation:
            - True: Adaptive rank selection (Default).
            - False: Full eigenvalue decomposition.
        :param max_rankZ: Maximum rank used for projecting the Z matrix (for adaptive projection).
            Default is None (will be computed based on `n_img`).
        :param max_rankW: Maximum rank used for projecting the W matrix (for adaptive projection).
            Default is None (will be computed based on `n_img`).
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

        # Handle parameters specific to CommonlineLUD
        self.alpha = alpha  # Spectral norm constraint bound
        if self.alpha is not None and not (2 / 3 <= self.alpha < 1):
            raise ValueError("Spectral norm constraint, alpha, must be in [2/3, 1).")

        logger.info(
            f"Initializing LUD algorithm using ADMM with spectral norm constraint: {self.alpha}"
        )

        self.tol = tol
        self.mu = mu
        self.gam = gam
        self.EPS = EPS
        self.maxit = maxit
        self.adp_proj = adp_proj
        self.max_rankZ = max_rankZ
        self.max_rankW = max_rankW

        # Parameters for adjusting mu
        self.adp_mu = adp_mu
        self.dec_mu = dec_mu
        self.inc_mu = inc_mu
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.min_mu_itr = min_mu_itr
        self.max_mu_itr = max_mu_itr
        self.delta_mu_l = delta_mu_l
        self.delta_mu_u = delta_mu_u

        # Initialize commonline base class
        super().__init__(src, **kwargs)

        # Upper-triangular mask used in `_Q_theta`
        ut_mask = np.zeros((self.n_img, self.n_img), dtype=bool)
        ut_mask[np.triu_indices(self.n_img, k=1)] = True
        self.ut_mask = ut_mask

    def estimate_rotations(self):
        """
        Estimate rotation matrices using the common lines method with LUD optimization.
        """
        logger.info("Computing the common lines matrix.")
        self.build_clmatrix()

        self._cl_to_C(self.clmatrix)
        gram = self._compute_Gram()
        gram = self._restructure_Gram(gram)
        self.rotations = CommonlineSDP._deterministic_rounding(gram)

        return self.rotations

    def _compute_Gram(self):
        """
        Perform the alternating direction method of multipliers (ADMM) for the SDP
        problem:

        min sum_{i<j} ||c_ij - G_ij c_ji||
        s.t. A(G) = b, G psd
            ||G||_2 <= lambda

        Equivalent to matlab functions cryoEMSDPL12N/cryoEMSDPL12N_vsimple.

        :param C: ndarray, A 3D array (n_img x n_img x 2) containing commonline coordinates (in Cartesian form)
            between pairs of images. Each C[i, j] stores the x and y coordinates of the common
            line between image i and image j.
        :return: The gram matrix G.
        """
        logger.info("Performing ADMM to compute Gram matrix.")

        # Initialize problem parameters
        n = 2 * self.n_img
        b = np.concatenate(
            [np.ones(n, dtype=self.dtype), np.zeros(self.n_img, dtype=self.dtype)]
        )

        # Adjust rank limits
        self.max_rankW = self.max_rankW or max(6, self.n_img // 2)
        if self.alpha is not None:
            self.max_rankZ = self.max_rankZ or max(6, self.n_img // 2)

        # Initialize variables
        G = np.eye(n, dtype=self.dtype)
        W = np.eye(n, dtype=self.dtype)
        Z = np.eye(n, dtype=self.dtype)
        Phi = G / self.mu
        if self.alpha is None:
            Phi += W

        # Compute initial values
        S, theta = self._Q_theta(Phi)
        AS = self._compute_AX(S)
        resi = self._compute_AX(G) - b

        nev = 0
        itmu_pinf = 0
        itmu_dinf = 0
        zz = 0
        kk = 0
        dH = 0
        for itr in range(self.maxit):
            #############
            # Compute y #
            #############
            y = -(AS + self._compute_AX(W)) - resi / self.mu
            if self.alpha is not None:
                y += self._compute_AX(Z)

            #################
            # Compute theta #
            #################
            ATy = self._compute_ATy(y)
            Phi = W + ATy + G / self.mu
            if self.alpha is not None:
                Phi -= Z
            S, theta = self._Q_theta(Phi)

            #############
            # Compute Z #
            #############
            if self.alpha is not None:
                Z, kk, zz = self._compute_Z(S, W, ATy, G, zz, itr, kk, nev)

            #############
            # Compute W #
            #############
            H = -S - ATy - G / self.mu
            if self.alpha is not None:
                H += Z
            H = (H + H.T) / 2

            if not self.adp_proj:
                D, V = np.linalg.eigh(H)
                eigs_mask = D > self.EPS
                V = V[:, eigs_mask]
                W = V @ np.diag(D[eigs_mask]) @ V.T
            else:
                # Determine number of eigenvalues to compute for adaptive projection
                if itr == 0:
                    nev = self.max_rankW
                else:
                    nev = self._compute_num_eigs(nev, dH, nev, 50, 5)

                # If using a spectral norm constraint cap num_eigs at 2*n_img
                if self.alpha is not None:
                    nev = min(nev, n)

                # Compute Eigenvectors and sort by largest algebraic eigenvalue
                dH, V = eigsh(-H.astype(np.float64), k=nev, which="LA")
                dH = dH[::-1].astype(self.dtype, copy=False)
                V = V[:, ::-1].astype(self.dtype, copy=False)

                nD = dH > self.EPS
                dH = dH[nD]
                nev = np.count_nonzero(nD)
                W = V[:, nD] @ np.diag(dH) @ V[:, nD].T + H if nD.any() else H

            ############
            # Update G #
            ############
            G = (1 - self.gam) * G + self.gam * self.mu * (W - H)

            # Check optimality
            resi = self._compute_AX(G) - b
            pinf = np.linalg.norm(resi) / max(np.linalg.norm(b), 1)

            dinf_term = S + W + ATy
            if self.alpha is not None:
                dinf_term -= Z
            dinf = np.linalg.norm(dinf_term, "fro") / max(np.linalg.norm(S, np.inf), 1)

            logger.info(
                f"Iteration: {itr}, residual: {max(pinf, dinf)}, target: {self.tol}"
            )
            if max(pinf, dinf) <= self.tol:
                return G

            # Update mu adaptively
            if self.adp_mu:
                if pinf / dinf <= self.delta_mu_l:
                    itmu_pinf = itmu_pinf + 1
                    itmu_dinf = 0
                    if itmu_pinf > self.max_mu_itr:
                        self.mu = max(self.mu * self.inc_mu, self.mu_min)
                        itmu_pinf = 0
                elif pinf / dinf > self.delta_mu_u:
                    itmu_dinf = itmu_dinf + 1
                    itmu_pinf = 0
                    if itmu_dinf > self.max_mu_itr:
                        self.mu = min(self.mu * self.dec_mu, self.mu_max)
                        itmu_dinf = 0

        return G

    def _compute_Z(self, S, W, ATy, G, zz, itr, kk, nev):
        """
        Update ADMM subproblem for enforcing the spectral norm constraint.

        :param S: A 2*n_img x 2*n_img symmetric matrix representing the fidelity term.
        :param W: A 2*n_img x 2*n_img array, primary ADMM subproblem matrix.
        :param ATy: A 2*n_img x 2*n_img array.
        :param G: Current value of the 2*n_img x 2*n_img optimization solution matrix.
        :param zz: eigenvalues from previous iteration.
        :param itr: ADMM loop iteration.
        :param kk: Number of eigenvalues of Z to use to enforce spectral norm constraint.
        :param nev: Number of eigenvalues of W used in previous iteration of ADMM.

        :returns:
            - Z, Updated 2*n_img x 2*n_img matrix for spectral norm constraint ADMM subproblem.
            - kk, Number of eigenvalues of Z to use to enforce spectral norm constraint in next iteration.
            - nev, Number of eigenvalues of W to use in this iteration of ADMM.
        """
        lambda_ = self.alpha * self.n_img  # Spectral norm bound
        B = S + W + ATy + G / self.mu
        B = (B + B.T) / 2

        if not self.adp_proj:
            U, pi = np.linalg.eigh(B)
        else:
            # Determine number of eigenvalues to compute for adaptive projection
            if itr == 0:
                kk = self.max_rankZ
            else:
                kk = self._compute_num_eigs(kk, zz, nev, 10, 3)

            kk = min(kk, 2 * self.n_img)
            pi, U = eigsh(
                B.astype(np.float64, copy=False), k=kk, which="LM"
            )  # Compute top `kk` eigenvalues and eigenvectors

            # Sort by eigenvalue magnitude. Note, eigsh does not return
            # ordered eigenvalues/vectors for which="LM".
            idx = np.argsort(np.abs(pi))[::-1]
            pi = pi[idx]
            U = U[:, idx]

        # Apply soft-threshold to eigenvalues to enforce spectral norm constraint.
        zz = np.sign(pi) * np.maximum(np.abs(pi) - lambda_ / self.mu, 0)
        nD = zz > 0
        kk = np.count_nonzero(nD)
        if kk > 0:
            zz = zz[nD]
            Z = U[:, nD] @ np.diag(zz) @ U[:, nD].T
        else:
            Z = np.zeros_like(B)

        return Z, kk, zz

    def _Q_theta(self, phi):
        """
        Compute the matrix S and auxiliary variables theta for the optimization problem.

        This function calculates the fidelity matrix S and the auxiliary variable theta as part of
        the ADMM framework for solving the semidefinite programming (SDP) relaxation of the
        Least Unsquared Deviations (LUD) problem. It ensures consistency between the Gram
        matrix G and the detected common-line coordinates.

        Python equivalent of matlab Qtheta MEX function.

        :param phi: ndarray, A 2*n_img x 2*n_img scaled dual variable matrix (Phi) used in the ADMM iterations.
        :returns:
            - S, A 2*n_img x 2*n_img matrix representing the fidelity term. It is a symmetric matrix
                derived from the commonline constraints, normalized by theta.
            - theta, A 3D array (n_img x n_img x 2) containing the normalized commonline directions
                used to compute S. Each theta[i, j] stores the normalized adjustments for the common
                line between image i and image j.
        """
        # Initialize theta, shape = (n_img, n_img, 2).
        theta = np.zeros_like(self.C)

        # Compute theta
        phi = phi.reshape(self.n_img, 2, self.n_img, 2).transpose(0, 2, 1, 3)
        sum_prod = (phi[self.ut_mask] * self.C_t[self.ut_mask, None]).sum(axis=2)
        theta[self.ut_mask] = self.C[self.ut_mask] - self.mu * sum_prod

        # Normalize theta
        theta_norm = np.linalg.norm(theta[self.ut_mask], axis=-1)[..., None]
        if self.alpha is not None:
            theta[self.ut_mask] /= theta_norm
        else:
            theta[self.ut_mask] /= np.maximum(theta_norm, self.mu)

        # Construct S
        S = theta[..., None] * self.C_t[:, :, None]
        S = S.transpose(0, 2, 1, 3).reshape(2 * self.n_img, 2 * self.n_img)

        # Ensure S is symmetric
        S = (S + S.T) / 2

        return S, theta

    def _cl_to_C(self, clmatrix):
        """
        For each pair of commonline indices cl[i, j] and cl[j, i], convert
        from polar commonline indices to cartesion coordinates.

        :param clmatrix: n_img x n_img commonline matrix.
        :return: n_img x n_img x 2 array of commonline cartesian coordinates.
        """
        C = np.zeros((self.n_img, self.n_img, 2), dtype=self.dtype)
        for i in range(self.n_img):
            for j in range(i + 1, self.n_img):
                cl_ij = clmatrix[i, j]
                cl_ji = clmatrix[j, i]

                # Compute (xij, yij) and (xji, yji) from common lines
                C[i, j, 0] = np.cos(2 * np.pi * cl_ij / self.n_theta)
                C[i, j, 1] = np.sin(2 * np.pi * cl_ij / self.n_theta)
                C[j, i, 0] = np.cos(2 * np.pi * cl_ji / self.n_theta)
                C[j, i, 1] = np.sin(2 * np.pi * cl_ji / self.n_theta)

        self.C = C
        self.C_t = np.ascontiguousarray(C.transpose(1, 0, 2))

    @staticmethod
    def _compute_num_eigs(num_eigs_prev, eig_vec, num_eigs_W, rel_drp_thresh, eigs_inc):
        """
        Compute number of eigenvalues to use when implementing adaptive projection.

        :param num_eigs_prev: Number of eigenvalues used in previous iteration of ADMM.
        :param eig_vec: Eigenvector result from previous iteration of ADMM.
        :param num_eigs_W: Number of eigenvalues used in previous computation of
            ADMM subproblem for solving W.
        :param rel_drp_thresh: Relative drop threshold for determining number of
            eigenvalues to use.
        :param eigs_inc: Number of eigenvalues to increase by if relative drop
            threshold is not met.

        :return: Number of eigenvalues to use in current iteration.
        """
        if len(eig_vec) == 1:
            # Handles the case were `drops` will be empty
            return num_eigs_prev + eigs_inc

        if num_eigs_prev > 0:
            drops = eig_vec[:-1] / eig_vec[1:]
            imx = np.argmax(drops)
            dmx = drops[imx]

            # Relative drop
            rel_drp = (
                (num_eigs_W - 1) * dmx / (np.sum(drops) - dmx)
                if len(drops) > 1
                else np.inf
            )
            if rel_drp > rel_drp_thresh:
                num_eigs = max(imx, 6)
            else:
                num_eigs = num_eigs_prev + eigs_inc
        else:
            num_eigs = 6

        return num_eigs

    @staticmethod
    def _compute_AX(X):
        """
        Compute the application of the linear operator A to the symmetric input
        matrix X, where A(X) is defined as:

        A(X) = [
            X_ii^(11),
            X_ii^(22),
            sqrt(2)/2 * X_ii^(12) + sqrt(2)/2 * X_ii^(21)
        ]

        i = 1, 2, ..., K

        where X_{ii}^{pq} denotes the (p,q)-th element in the 2x2 sub-block X_{ii}.

        :param X: 2D square array of shape (2K, 2K)..
        :return: Flattened array representing A(X)
        """
        # Extract the diagonal elements of (X_ii^(11) and X_ii^(22))
        diags = np.diag(X)

        # Extract the off-diagonal elements from each 2x2 sub-block
        off_diag_vals = np.diag(X, k=1)[::2]  # Every other superdiagonal element

        # Compute the second part of AX, which is sqrt(2)/2 times the sum of
        # the off-diagonal entries of each 2x2 sub-block on the diagonal of X.
        # Since each sub-block is symmetric, we take just one entry and multiply
        # by sqrt(2).
        sqrt_2_X_col = np.sqrt(2, dtype=X.dtype) * off_diag_vals

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
        ATy[::2, 1::2] = np.diag(off_diag_vals)  # Y_ii^(12)
        ATy[1::2, ::2] = np.diag(off_diag_vals)  # Y_ii^(21)

        return ATy

    def _restructure_Gram(self, G):
        """
        Restructures the input Gram matrix into a block structure based on the following
        format:

        G =
        [ G^(11)   G^(12) ]
        [ G^(21)   G^(22) ]

        =
        [ (R^1)^T R^1   (R^1)^T R^2 ]
        [ (R^2)^T R^1   (R^2)^T R^2 ]

        where R^i is the concatenation of all i'th columns of the rotations R.

        :param G: Gram matrix from ADMM method.
        :return: Restructured Gram matrix.
        """
        # Get odd and even indices
        odd_indices = np.arange(0, G.shape[0], 2)
        even_indices = np.arange(1, G.shape[0], 2)

        # Extract blocks
        G_11 = G[np.ix_(odd_indices, odd_indices)]
        G_12 = G[np.ix_(odd_indices, even_indices)]
        G_21 = G[np.ix_(even_indices, odd_indices)]
        G_22 = G[np.ix_(even_indices, even_indices)]

        # Combine blocks into the new structure
        restructured_G = np.block([[G_11, G_12], [G_21, G_22]])

        return restructured_G
