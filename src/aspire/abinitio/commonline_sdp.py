import logging

import cvxpy as cp
import numpy as np
from scipy.sparse import csr_array

from aspire.abinitio import CLOrient3D

logger = logging.getLogger(__name__)


class CommonlineSDP(CLOrient3D):
    """
    Class to estimate 3D orientations using semi-definite programming.

    See the following publication for more details:

    A. Singer and Y. Shkolnisky,
    "Three-Dimensional Structure Determination from Common Lines in Cryo-EM
        by Eigenvectors and Semidefinite Programming"
    SIAM J. Imaging Sciences, Vol. 4, No. 2, (2011): 543-572. doi:10.1137/090767777
    """

    def estimate_rotations(self):
        """
        Estimate rotation matrices using the common lines method with semi-definite programming.
        """
        logger.info("Computing the common lines matrix.")
        self.build_clmatrix()

        S = self._construct_S(self.clmatrix)
        A, b = self._sdp_prep()
        gram = self._compute_gram_matrix(S, A, b)
        self.rotations = self._deterministic_rounding(gram)

        return self.rotations

    def _construct_S(self, clmatrix):
        """
        Construct the 2*n_img x 2*n_img quadratic form matrix S corresponding to the common-lines
        matrix 'clmatrix'.

        :param clmatrix: n_img x n_img common-lines matrix.

        :return: 2*n_img x 2*n_img quadratic form matrix S.
        """
        logger.info("Constructing the common line quadratic form matrix S.")

        S11 = np.zeros((self.n_img, self.n_img), dtype=self.dtype)
        S12 = np.zeros((self.n_img, self.n_img), dtype=self.dtype)
        S21 = np.zeros((self.n_img, self.n_img), dtype=self.dtype)
        S22 = np.zeros((self.n_img, self.n_img), dtype=self.dtype)

        for i in range(self.n_img):
            for j in range(i + 1, self.n_img):
                cij = clmatrix[i, j]
                cji = clmatrix[j, i]

                xij = np.cos(2 * np.pi * cij / self.n_theta)
                yij = np.sin(2 * np.pi * cij / self.n_theta)
                xji = np.cos(2 * np.pi * cji / self.n_theta)
                yji = np.sin(2 * np.pi * cji / self.n_theta)

                S11[i, j] = xij * xji
                S11[j, i] = xji * xij

                S12[i, j] = xij * yji
                S12[j, i] = xji * yij

                S21[i, j] = yij * xji
                S21[j, i] = yji * xij

                S22[i, j] = yij * yji
                S22[j, i] = yji * yij

        S = np.block([[S11, S12], [S21, S22]])

        return S

    def _sdp_prep(self):
        """
        Prepare optimization problem constraints.

        The constraints for the SDP optimization, max tr(SG), performed in `_compute_gram_matrix()`
        as min tr(-SG), are that the Gram matrix, G, is semidefinite positive and G11_ii = G22_ii = 1,
        G12_ii = G21_ii = 0, i=1,2,...,N, for the block representation of G = [[G11, G12], [G21, G22]].

        We build a corresponding constraint for CVXPY in the form of tr(A_j @ G) = b_j, j = 1,...,p.
        For the constraint G11_ii = G22_ii = 1, we have A_j[i, i] = 1 (zeros elsewhere) and b_j = 1.
        For the constraint G12_ii = G21_ii = 0, we have A_j[i, i] = 1 (zeros elsewhere) and b_j = 0.

        :returns: Constraint data A, b.
        """
        logger.info("Preparing SDP optimization constraints.")

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

    def _compute_gram_matrix(self, S, A, b):
        """
        Compute the Gram matrix by solving an SDP optimization.

        The Gram matrix will be of the form G = R.T @ R, where R = [R1 R2] or the concatenation
        of the first columns of every rotation, R1, and the second columns of every rotation, R2.
        From this Gram matrix, the rotations can be recovered using the deterministic rounding
        procedure below.

        Here we optimize over G, max tr(SG), written as min tr(-SG), subject to the constraints
        described in `_spd_prep()`. It should be noted that tr(SG) = sum(dot(R_i @ c_ij, R_j @ c_ji)),
        and that maximizing this objective function is equivalently to minimizing the L2 norm
        of R_i @ c_ij -  R_j @ c_ji, ie. finding the best approximation for the rotations R_i.

        :param S: The common-line quadratic form matrix of shape 2 * n_img x 2 * n_img.
        :param A: 3 * n_img sparse arrays of constraint data.
        :param b: 3 * n_img scalars such that tr(A_i G) = b_i.

        :return: Gram matrix.
        """
        logger.info("Solving SDP to approximate Gram matrix.")

        n = 2 * self.n_img
        # Define and solve the CVXPY problem.
        # Create a symmetric matrix variable.
        G = cp.Variable((n, n), symmetric=True)
        # The operator >> denotes matrix inequality.
        constraints = [G >> 0]
        constraints += [cp.trace(A[i] @ G) == b[i] for i in range(3 * self.n_img)]
        prob = cp.Problem(cp.Minimize(cp.trace(-S @ G)), constraints)
        prob.solve()

        return G.value
