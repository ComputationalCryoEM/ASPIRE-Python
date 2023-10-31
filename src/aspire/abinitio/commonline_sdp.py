import logging

import cvxpy as cp
import numpy as np
from scipy.sparse import csr_array

from aspire.abinitio import CLOrient3D
from aspire.utils import nearest_rotations
from aspire.utils.matlab_compat import stable_eigsh

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
        rotations = self._deterministic_rounding(gram)
        self.rotations = rotations

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

    def _deterministic_rounding(self, gram):
        """
        Deterministic rounding procedure to recover the rotations from the Gram matrix.

        The Gram matrix contains information about the first two columns of every rotation
        matrix. These columns are extracted and used to form the remaining column of every
        rotation matrix.

        :param gram: A 2n_img x 2n_img Gram matrix.

        :return: An n_img x 3 x 3 stack of rotation matrices.
        """
        logger.info("Recovering rotations from Gram matrix.")

        # Obtain top eigenvectors from Gram matrix.
        d, v = stable_eigsh(gram, 5)
        sort_idx = np.argsort(-d)
        logger.info(f"Top 5 eigenvalues from (rank-3) Gram matrix: {d[sort_idx]}")

        # Only need the top 3 eigen-vectors.
        v = v[:, sort_idx[:3]]

        # According to the structure of the Gram matrix, the first `n_img` rows, denoted v1,
        # correspond to the linear combination of the vectors R_{i}^{1}, i=1,...,K, that is of
        # column 1 of all rotation matrices. Similarly, the second `n_img` rows of v,
        # denoted v2, are linear combinations of R_{i}^{2}, i=1,...,K, that is, the second
        # column of all rotation matrices.
        v1 = v[: self.n_img].T
        v2 = v[self.n_img : 2 * self.n_img].T

        # Use a least-squares method to get A.T*A and a Cholesky decomposition to find A.
        A = self._ATA_solver(v1, v2)

        # Recover the rotations. The first two columns of all rotation
        # matrices are given by unmixing V1 and V2 using A. The third
        # column is the cross product of the first two.
        r1 = np.dot(A.T, v1)
        r2 = np.dot(A.T, v2)
        r3 = np.cross(r1, r2, axis=0)
        rotations = np.stack((r1.T, r2.T, r3.T), axis=-1)

        # Make sure that we got rotations by enforcing R to be
        # a rotation (in case the error is large)
        rotations = nearest_rotations(rotations)

        return rotations

    @staticmethod
    def _ATA_solver(v1, v2):
        """
        Uses a least squares method to solve for the linear transformation A
        such that A*v1=R1 and A*v2=R2 correspond to the first and second columns
        of a sequence of rotation matrices.

        :param v1: 3 x n_img array corresponding to linear combinations of the first
            columns of all rotation matrices.
        :param v2: 3 x n_img array corresponding to linear combinations of the second
            columns of all rotation matrices.

        :return: 3x3 linear transformation mapping v1, v2 to first two columns of rotations.
        """
        # We look for a linear transformation (3 x 3 matrix) A such that
        # A*v1'=R1 and A*v2=R2 are the columns of the rotations matrices.
        # Therefore:
        # v1 * A'*A v1' = 1
        # v2 * A'*A v2' = 1
        # v1 * A'*A v2' = 0
        # These are 3*K linear equations for 9 matrix entries of A'*A
        # Actually, there are only 6 unknown variables, because A'*A is symmetric.
        # So we will truncate from 9 variables to 6 variables corresponding
        # to the upper half of the matrix A'*A
        n_img = v1.shape[-1]
        truncated_equations = np.zeros((3 * n_img, 9), dtype=v1.dtype)
        k = 0
        for i in range(3):
            for j in range(3):
                truncated_equations[0::3, k] = v1[i] * v1[j]
                truncated_equations[1::3, k] = v2[i] * v2[j]
                truncated_equations[2::3, k] = v1[i] * v2[j]
                k += 1

        # b = [1 1 0 1 1 0 ...]' is the right hand side vector
        b = np.ones(3 * n_img)
        b[2::3] = 0

        # Find the least squares approximation of A'*A in vector form
        ATA_vec = np.linalg.lstsq(truncated_equations, b, rcond=None)[0]

        # Construct the matrix A'*A from the vectorized matrix.
        # Note, this is only the lower triangle of A'*A.
        ATA = ATA_vec.reshape(3, 3)

        # The Cholesky decomposition of A'*A gives A (lower triangle).
        # Note, that `np.linalg.cholesky()` only uses the lower-triangular
        # and diagonal elements of ATA.
        A = np.linalg.cholesky(ATA)

        return A
