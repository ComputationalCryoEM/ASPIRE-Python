import logging

import cvxpy as cp
import numpy as np
from scipy.sparse import csr_array

from aspire.abinitio import CLOrient3D

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
        self.build_clmatrix()
        S = self.construct_S(self.clmatrix)
        A, b = self.sdp_prep()
        Gram = self.compute_Gram_matrix(S, A, b)
        return Gram

    def construct_S(self, clmatrix):
        """
        Compute the 2*n_img x 2*n_img commonline matrix S.
        """
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

    def sdp_prep(self):
        """
        Prepare optimization problem constraints.
        """
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
            data = np.ones(2, dtype=self.dtype)
            row_ind = np.array([i, self.n_img + i])
            col_ind = np.array([self.n_img + i, i])
            A_i = csr_array((data, (row_ind, col_ind)), shape=(n, n), dtype=self.dtype)
            A.append(A_i)
            b.append(np.zeros(1, dtype=self.dtype))

        return A, b

    def compute_Gram_matrix(self, S, A, b):
        """
        Compute the Gram matrix by solving a SDP.
        """
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
