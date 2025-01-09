import logging

import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.linalg import eigs

from aspire.abinitio import CLOrient3D

logger = logging.getLogger(__name__)


class CommonlineLUD(CLOrient3D):
    """
    Define a derived class to estimate 3D orientations using Least Unsquared
    Deviations as described in the following publication:
    L. Wang, A. Singer, and  Z. Wen, Orientation Determination of Cryo-EM Images Using
    Least Unsquared Deviations, SIAM J. Imaging Sciences, 6, 2450-2483 (2013).
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a class for estimating 3D orientations using a Least Unsquared Deviations algorithm.

        This class extends the `CLOrient3D` class, inheriting its initialization parameters.

        :param alpha: Spectral norm constraint for ADMM algorithm. Default is None, which
            does not apply a spectral norm constraint. To apply a spectral norm constraint provide
            a value in the range [2/3, 1), 2/3 is recommended.
        :param tol: Tolerance for convergence. The algorithm stops when conditions reach this threshold.
            Default is 1e-3.
        :param mu: The penalty parameter (or dual variable scaling factor) in the optimization problem.
            Default is 1.
        :param gam: Relaxation factor for updating variables in the algorithm (typically between 1 and 2).
            Default is 1.618.
        :param EPS: Small positive value used to filter out negligible eigenvalues or avoid numerical issues.
            Default is 1e-12.
        :param maxit: Maximum number of iterations allowed for the algorithm.
            Default is 1000.
        :param adp_proj: Flag for using adaptive projection during eigenvalue computation:
            - 0: Full eigenvalue decomposition.
            - 1: Adaptive rank selection (Default).
        :param max_rankZ: Maximum rank used for projecting the Z matrix (for adaptive projection).
            Default is None (will be computed based on `n_img`).
        :param max_rankW: Maximum rank used for projecting the W matrix (for adaptive projection).
            Default is None (will be computed based on `n_img`).
        :param adp_mu: Adaptive adjustment of the penalty parameter `mu`:
            - 1: Enabled.
            - 0: Disabled.
            Default is 1.
        :param dec_mu: Scaling factor for decreasing `mu` when conditions warrant.
            Default is 0.5.
        :param inc_mu: Scaling factor for increasing `mu` when conditions warrant.
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
        self.alpha = kwargs.pop("alpha", None)  # Spectral norm constraint bound
        if self.alpha is not None:
            if not (2 / 3 <= self.alpha < 1):
                raise ValueError(
                    "Spectral norm constraint, alpha, must be in [2/3, 1)."
                )
            else:
                logger.info(
                    f"Initializing LUD algorithm using ADMM with spectral norm constraint {self.alpha}."
                )
                self.spectral_norm_constraint = True
        else:
            logger.info(
                "Initializing LUD algorithm using ADMM without spectral norm constraint."
            )
            self.spectral_norm_constraint = False

        self.tol = kwargs.pop("tol", 1e-3)
        self.mu = kwargs.pop("mu", 1)
        self.gam = kwargs.pop("gam", 1.618)
        self.EPS = kwargs.pop("EPS", 1e-12)
        self.maxit = kwargs.pop("maxit", 1000)
        self.adp_proj = kwargs.pop("adp_proj", 1)
        self.max_rankZ = kwargs.pop("max_rankZ", None)
        self.max_rankW = kwargs.pop("max_rankW", None)

        # Parameters for adjusting mu
        self.adp_mu = kwargs.pop("adp_mu", 1)
        self.dec_mu = kwargs.pop("dec_mu", 0.5)
        self.inc_mu = kwargs.pop("inc_mu", 2)
        self.mu_min = kwargs.pop("mu_min", 1e-4)
        self.mu_max = kwargs.pop("mu_max", 1e4)
        self.min_mu_itr = kwargs.pop("min_mu_itr", 5)
        self.max_mu_itr = kwargs.pop("max_mu_itr", 20)
        self.delta_mu_l = kwargs.pop("delta_mu_l", 0.1)
        self.delta_mu_u = kwargs.pop("delta_mu_u", 10)

        # Call the parent class initializer
        super().__init__(*args, **kwargs)

    def estimate_rotations(self):
        """
        Estimate rotation matrices using the common lines method with semi-definite programming.
        """
        logger.info("Computing the common lines matrix.")
        self.build_clmatrix()

        C = self.cl_to_C(self.clmatrix)
        gram = self.cryoEMSDPL12N(C)
        gram = self._restructure_Gram(gram)
        self.rotations = self._deterministic_rounding(gram)

        return self.rotations

    def cryoEMSDPL12N(self, C):
        """
        Perform the alternating direction method of multipliers (ADMM) for the SDP
        problem:

        min sum_{i<j} ||c_ij - G_ij c_ji||
        s.t. A(G) = b, G psd
            ||G||_2 <= lambda

        :param C:
        :return: The gram matrix G.
        """
        # Initialize problem parameters
        n = 2 * self.n_img
        b = np.concatenate([np.ones(n), np.zeros(self.n_img)])

        # Adjust rank limits
        self.max_rankW = self.max_rankW or max(6, self.n_img // 2)
        if self.spectral_norm_constraint:
            self.max_rankZ = self.max_rankZ or max(6, self.n_img // 2)

        # Initialize variables
        G = np.eye(n, dtype=self.dtype)
        W = np.eye(n, dtype=self.dtype)
        if self.spectral_norm_constraint:
            Z = W
            Phi = G / self.mu
        else:
            Phi = W + G / self.mu

        # Compute initial values
        S, theta = self.Qtheta(Phi, C, self.mu)
        S = (S + S.T) / 2
        AS = self.ComputeAX(S)
        resi = self.ComputeAX(G) - b

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
            y = -(AS + self.ComputeAX(W)) - resi / self.mu
            if self.spectral_norm_constraint:
                y += self.ComputeAX(Z)

            #################
            # Compute theta #
            #################
            ATy = self.ComputeATy(y)
            Phi = W + ATy + G / self.mu
            if self.spectral_norm_constraint:
                Phi -= Z
            S, theta = self.Qtheta(Phi, C, self.mu)
            S = (S + S.T) / 2

            #############
            # Compute Z #
            #############
            if self.spectral_norm_constraint:
                Z, kk, zz = self._compute_Z(S, W, ATy, G, zz, itr, kk, nev)

            #############
            # Compute W #
            #############
            H = -S - ATy - G / self.mu
            if self.spectral_norm_constraint:
                H += Z
            H = (H + H.T) / 2

            if self.adp_proj == 0:
                D, V = np.linalg.eigh(H)
                W = V[:, D > self.EPS] @ np.diag(D[D > self.EPS]) @ V[:, D > self.EPS].T
            else:
                if itr == 0:
                    nev = self.max_rankW
                else:
                    if nev > 0:
                        drops = dH[:-1] / dH[1:]
                        dmx, imx = max((val, idx) for idx, val in enumerate(drops))
                        rel_drp = (nev - 1) * dmx / (np.sum(drops) - dmx)

                        if rel_drp > 50:
                            nev = max(imx + 1, 6)
                        else:
                            nev = nev + 5
                    else:
                        nev = 6

                n_eigs = nev
                if self.spectral_norm_constraint:
                    n_eigs = min(nev, n)
                dH, V = eigs(-H, k=n_eigs, which="LR")

                # Sort by eigenvalue magnitude.
                dH = dH.real
                idx = np.argsort(dH)[::-1]
                dH = dH[idx]
                V = V[:, idx].real
                nD = dH > self.EPS
                dH = dH[nD]
                nev = np.count_nonzero(nD)
                W = V[:, nD] @ np.diag(dH) @ V[:, nD].T + H if nD.any() else H

            ############
            # Update G #
            ############
            G = (1 - self.gam) * G + self.gam * self.mu * (W - H)

            # Check optimality
            resi = self.ComputeAX(G) - b
            pinf = np.linalg.norm(resi) / max(np.linalg.norm(b), 1)

            dinf_term = S + W + ATy
            if self.spectral_norm_constraint:
                dinf_term -= Z
            dinf = np.linalg.norm(dinf_term, "fro") / max(np.linalg.norm(S, np.inf), 1)

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
        """
        lambda_ = self.alpha * self.n_img  # Spectral norm bound
        B = S + W + ATy + G / self.mu
        B = (B + B.T) / 2

        if self.adp_proj == 0:
            U, pi = np.linalg.eigh(B)
        else:
            if itr == 0:
                kk = self.max_rankZ
            else:
                if kk > 0:
                    # Initialize relative drop
                    rel_drp = 0
                    imx = 0
                    # Calculate relative drop based on `zz`
                    if len(zz) == 2:
                        rel_drp = np.inf
                    elif len(zz) > 2:
                        drops = zz[:-1] / zz[1:]
                        dmx, imx = max(
                            (val, idx) for idx, val in enumerate(drops)
                        )  # Find max drop and its index
                        rel_drp = (nev - 1) * dmx / (np.sum(drops) - dmx)

                    # Update `kk` based on relative drop
                    kk = max(imx, 6) if rel_drp > 10 else kk + 3
                else:
                    kk = 6

            kk = min(kk, 2 * self.n_img)
            pi, U = eigs(
                B, k=kk, which="LM"
            )  # Compute top `kk` eigenvalues and eigenvectors

            # Sort by eigenvalue magnitude.
            idx = np.argsort(np.abs(pi))[::-1]
            pi = pi[idx]
            U = U[:, idx].real
            pi = pi.real  # Ensure real eigenvalues for subsequent calculations

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

    def Qtheta(self, phi, C, mu):
        """
        Python equivalent of Qtheta MEX function.

        Compute the matrix S and auxiliary variables theta for the optimization problem.

        This function calculates the fidelity matrix S and the auxiliary variable theta as part of
        the ADMM framework for solving the semidefinite programming (SDP) relaxation of the
        Least Unsquared Deviations (LUD) problem. It ensures consistency between the Gram
        matrix G and the detected common-line coordinates.

        :param phi: ndarray, A 2*n_img x 2*n_img scaled dual variable matrix (Phi) used in the ADMM iterations.
        :param C: ndarray, A 3D array (n_img x n_img x 2) containing commonline coordinates (in Cartesian form)
            between pairs of images. Each C[i, j] stores the x and y coordinates of the common
            line between image i and image j.
        :param mu: float, The penalty parameter in the augmented Lagrangian. It controls the scaling of the
            dual variable contribution in the ADMM updates.
        :returns:
            - S, A 2*n_img x 2*n_img matrix representing the fidelity term. It is a symmetric matrix
                derived from the commonline constraints, normalized by theta.
            - theta, A 3D array (n_img x n_img x 2) containing the normalized commonline directions
                used to compute S. Each theta[i, j] stores the normalized adjustments for the common
                line between image i and image j.
        """
        # Initialize outputs
        S = np.zeros((2 * self.n_img, 2 * self.n_img))
        theta = np.zeros_like(C)

        # Main routine
        for i in range(self.n_img - 1):
            for j in range(i + 1, self.n_img):
                t = 0
                for k in range(2):
                    theta[i, j, k] = C[i, j, k] - mu * (
                        phi[2 * i + k, 2 * j] * C[j, i, 0]
                        + phi[2 * i + k, 2 * j + 1] * C[j, i, 1]
                    )
                    t += theta[i, j, k] ** 2

                t = np.sqrt(t)
                for k in range(2):
                    if self.spectral_norm_constraint:
                        theta[i, j, k] /= t
                    else:
                        theta[i, j, k] /= max(t, self.mu)
                    S[2 * i + k, 2 * j] = theta[i, j, k] * C[j, i, 0]
                    S[2 * i + k, 2 * j + 1] = theta[i, j, k] * C[j, i, 1]

        return S, theta

    def cl_to_C(self, clmatrix):
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

        return C

    def ComputeAX(self, X):
        n = 2 * self.n_img
        rows = np.arange(1, n, 2)
        cols = np.arange(0, n, 2)

        # Create diagonal matrix with X on the main diagonal
        diags = np.diag(X)

        # Compute the second part of AX
        sqrt_2_X_col = np.sqrt(2) * X[rows, cols]

        # Concatenate results vertically
        AX = np.concatenate((diags, sqrt_2_X_col))

        return AX

    def ComputeATy(self, y):
        n = 2 * self.n_img
        m = 3 * self.n_img
        idx = np.arange(n)
        rows = np.concatenate([np.arange(1, n, 2), np.arange(0, n, 2)])
        cols = np.concatenate([np.arange(0, n, 2), np.arange(1, n, 2)])
        data = np.concatenate([(np.sqrt(2) / 2) * y[n:m], (np.sqrt(2) / 2) * y[n:m]])

        # Combine diagonal elements
        diag_data = y[:n]
        diag_idx = idx

        # Construct the full matrix
        data = np.concatenate([data, diag_data])
        rows = np.concatenate([rows, diag_idx])
        cols = np.concatenate([cols, diag_idx])

        ATy = csr_array((data, (rows, cols)), shape=(n, n))
        return ATy

    def _lud_prep(self):
        """
        Prepare optimization problem constraints.

        The constraints for the LUD optimization, max tr(SG), performed in `_compute_gram_matrix()`
        as min tr(-SG), are that the Gram matrix, G, is semidefinite positive and G11_ii = G22_ii = 1,
        G12_ii = G21_ii = 0, i=1,2,...,N, for the block representation of G = [[G11, G12], [G21, G22]].

        We build a corresponding constraint in the form of tr(A_j @ G) = b_j, j = 1,...,p.
        For the constraint G11_ii = G22_ii = 1, we have A_j[i, i] = 1 (zeros elsewhere) and b_j = 1.
        For the constraint G12_ii = G21_ii = 0, we have A_j[i, i] = 1 (zeros elsewhere) and b_j = 0.

        :returns: Constraint data A, b.
        """
        logger.info("Preparing LUD optimization constraints.")

        n = 2 * self.n_img
        A = []
        b = []
        data = np.ones(1, dtype=self.dtype)
        for i in range(n):
            row_ind = np.array([i])
            col_ind = np.array([i])
            A_i = csr_array((data, (row_ind, col_ind)), shape=(n, n), dtype=self.dtype)
            A.append(A_i)
            b.append(1)

        for i in range(self.n_img):
            row_ind = np.array([i])
            col_ind = np.array([self.n_img + i])
            A_i = csr_array((data, (row_ind, col_ind)), shape=(n, n), dtype=self.dtype)
            A.append(A_i)
            b.append(0)

        b = np.array(b, dtype=self.dtype)

        return A, b

    def _restructure_Gram(self, G):
        """
        Restructures the input Gram matrix into a block structure based on odd and even
        indexed rows and columns.

        The new structure is:
            New G = [[Top Left Block,  Top Right Block],
                     [Bottom Left Block, Bottom Right Block]]

        Blocks:
        - Top Left Block: Rows and columns with odd indices.
        - Top Right Block: Odd rows and even columns.
        - Bottom Left Block: Even rows and odd columns.
        - Bottom Right Block: Even rows and columns.
        """
        # Get odd and even indices
        odd_indices = np.arange(0, G.shape[0], 2)
        even_indices = np.arange(1, G.shape[0], 2)

        # Extract blocks
        top_left = G[np.ix_(odd_indices, odd_indices)]
        top_right = G[np.ix_(odd_indices, even_indices)]
        bottom_left = G[np.ix_(even_indices, odd_indices)]
        bottom_right = G[np.ix_(even_indices, even_indices)]

        # Combine blocks into the new structure
        restructured_G = np.block([[top_left, top_right], [bottom_left, bottom_right]])

        return restructured_G
