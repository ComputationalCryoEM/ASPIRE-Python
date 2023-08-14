import logging

import cvxpy as cp
import numpy as np
from scipy.sparse import csr_array

from aspire.abinitio import CLOrient3D
from aspire.utils.matlab_compat import stable_eigsh

logger = logging.getLogger(__name__)


class CommonlineSDP(CLOrient3D):
    """
    Class to estimate 3D orientations using Semi-Definite Programming
    :cite:`DBLP:journals/siamis/SingerS11`
    """

    def __init__(self, src, n_rad=None, n_theta=360, max_shift=0.15, shift_step=1):
        """
        Initialize an object for estimating 3D orientations using semi-definite programming.

        :param src: The source object of 2D denoised or class-averaged images with metadata
        :param n_rad: The number of points in the radial direction
        :param n_theta: The number of points in the theta direction.
            Default is 360.
        :param max_shift: Determines maximum range for shifts as a proportion
            of the resolution. Default is 0.15.
        :param shift_step: Resolution for shift estimation in pixels. Default is 1 pixel.
        """
        super().__init__(
            src,
            n_rad=n_rad,
            n_theta=n_theta,
            max_shift=max_shift,
            shift_step=shift_step,
        )

    def estimate_rotations(self):
        """
        perform estimation of orientations
        """
        logger.info("Computing the common lines matrix.")
        self.build_clmatrix()

        S = self._construct_S(self.clmatrix)
        A, b = self._sdp_prep()
        Gram = self._compute_Gram_matrix(S, A, b)
        rotations = self._deterministic_rounding(Gram)
        self.rotations = rotations

    def _construct_S(self, clmatrix):
        """
        Compute the 2*n_img x 2*n_img commonline matrix S.
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
            b.append(np.ones(1, dtype=self.dtype))

        for i in range(self.n_img):
            data = np.ones(1, dtype=self.dtype)
            # row_ind = np.array([i, self.n_img + i])
            # col_ind = np.array([self.n_img + i, i])
            row_ind = np.array([i])
            col_ind = np.array([self.n_img + i])
            A_i = csr_array((data, (row_ind, col_ind)), shape=(n, n), dtype=self.dtype)
            A.append(A_i)
            b.append(np.zeros(1, dtype=self.dtype))

        return A, b

    def _compute_Gram_matrix(self, S, A, b):
        """
        Compute the Gram matrix by solving a SDP.
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

        Gram = G.value

        return Gram

    def _deterministic_rounding(self, Gram):
        """
        Deterministic rounding procedure to recover the rotations from the Gram matrix.

        :param Gram: A 2n_img x 2n_img Gram matrix.

        :return: An n_img x 3 x 3 stack of rotation matrices.
        """
        logger.info("Recovering rotations from Gram matrix.")

        # Obtain top eigenvectors from Gram matrix.
        d, v = stable_eigsh(Gram, 5)
        sort_idx = np.argsort(-d)
        logger.info(f"Top 5 eigenvalues from (rank-3) Gram matrix: {d[sort_idx]}")

        # Only need the top 3 eigen-vectors.
        v = v[:, sort_idx[:3]]

        # According to the structure of the Gram matrix, the first `n_img` rows, denoted v1,
        # correspond to the linear combination of the vectors R_{i}^{1}, i=1,...,K, that is of
        # column 1 of all rotation matrices. Similarly, the second `n_img` rows of v,
        # denoted v2, are linear combinations of R_{i}^{2}, i=1,...,K, that is, the second
        # column of all rotation matrices.
        v1 = v[: self.n_img].T.copy()
        v2 = v[self.n_img : 2 * self.n_img].T.copy()

        # Use a least-squares method to get A.T*A and a Cholesky decomposition to find A.
        A = self._ATA_solver(v1, v2, self.n_img)

        # Recover the rotations. The first two columns of all rotation
        # matrices are given by unmixing V1 and V2 using A. The third
        # column is the cross product of the first two.
        r1 = np.dot(A.T, v1)
        r2 = np.dot(A.T, v2)
        r3 = np.cross(r1, r2, axis=0)

        rotations = np.empty((self.n_img, 3, 3), dtype=self.dtype)
        rotations[:, :, 0] = r1.T
        rotations[:, :, 1] = r2.T
        rotations[:, :, 2] = r3.T
        # Make sure that we got rotations by enforcing R to be
        # a rotation (in case the error is large)
        u, _, v = np.linalg.svd(rotations)
        np.einsum("ijk, ikl -> ijl", u, v, out=rotations)

        return rotations

    @staticmethod
    def _ATA_solver(v1, v2, n_img):
        # We look for a linear transformation (3 x 3 matrix) A such that
        # A*V1'=R1 and A*V2=R2 are the columns of the rotations matrices.
        # Therefore:
        # v1 * A'*A v1' = 1
        # v2 * A'*A v2' = 1
        # v1 * A'*A v2' = 0
        # These are 3*K linear equations for 9 matrix entries of A'*A
        # Actually, there are only 6 unknown variables, because A'*A is symmetric.
        # So we will truncate from 9 variables to 6 variables corresponding
        # to the upper half of the matrix A'*A
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
        A = np.linalg.cholesky(ATA)

        return A
