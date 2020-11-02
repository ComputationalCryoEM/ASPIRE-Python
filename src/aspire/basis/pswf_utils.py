import numpy as np
from scipy.linalg import eigh_tridiagonal


class BNMatrix:
    """
    Define a class to compute the B_N matrix ( with elements of b^N_mn) denoting the matrix of the operator L_c
    with respect to the basis T_Nn(x) by elements of b^N_mn. The matrix element b^N_mn is calculated as
    < T_Nm, L_c T_Nn > as shown in the paper below:
      Yoel Shkolnisky, "Prolate spheroidal wave functions on a disc-Integration and approximation of
      two-dimensional bandlimited functions", Appl. Comput. Harmon. Anal. 22, 235-256 (2007).
    """

    def __init__(self, big_n, band_limit, approx_length, dtype=np.float32):
        """
        Initial an object to compute the B_N matrix ( with elements of b^N_mn).

        :param big_n: A positive integer represented by N.
        :param band_limit: The band limit.
        :param approx_length: The approximated length represented by n.
        """

        self.dtype = np.dtype(dtype)

        k = np.arange(1, approx_length, dtype=float)
        self.diagonal = np.ones(approx_length, dtype=self.dtype)
        self.diagonal[0] = self._generate_bn_mat_b_n_on_diagonal(big_n, 0, band_limit)
        # BN matrix is a symmetric tridiagonal matrix
        self.diagonal[1:] = self._generate_bn_mat_b_n_on_diagonal(big_n, k, band_limit)
        self.off_diagonal = self._generate_bn_mat_b_n_above_diagonal(
            big_n, k, band_limit
        )
        self.off_diagonal += self._generate_bn_mat_b_n_below_diagonal(
            big_n, k - 1, band_limit
        )
        self.off_diagonal /= 2

    def get_eig_vectors(self):
        """
        Calculated the eigenvalues and eigenvectors of B_N matrix.

        :return: v: (M,M) ndarray
                    The normalized eigenvectors corresponding to the eigenvalues, v[:, i] is corresponding to the w[i].
                    In each eigenvector v[:, i], v[argmax(abs(v[:, i])), i] >= 0.
                w: (M,) ndarray
                    The eigenvalues in descending order.
        """

        w, v = eigh_tridiagonal(self.diagonal, self.off_diagonal, select="a")

        sorted_idx = np.argsort(-w)
        v = v[:, sorted_idx]
        w = w[sorted_idx]
        # We need to rescale the eigenvectors to fix the sign problem and make consistent with
        # the Matlab version.
        a = np.argmax(np.absolute(v), axis=0)
        b = np.array([np.sign(v[a[k], k]) for k in range(len(v))], dtype=self.dtype)
        v = v * b

        return v, w

    def dense_mat(self):
        """
        Represent the full B_N matrix by a 2D array.

        :return: mat: (M,M), ndarray
                The full BN matrix.
        """

        diagonal = self.diagonal
        off_diagonal = self.off_diagonal
        m = len(diagonal)

        mat = np.zeros((m, m), dtype=self.dtype)
        mat[0, 0] = diagonal[0]

        for i in range(m - 1):
            mat[i, i + 1] = off_diagonal[i]
            mat[i + 1, i] = off_diagonal[i]
            mat[i + 1, i + 1] = diagonal[i + 1]

        return mat

    def shape(self):
        """
        Ruturn the shape of B_N matrix.

        :return: tuple: (n, n)
            The dense matrix shape
        """

        return len(self.diagonal), len(self.diagonal)

    def _generate_bn_mat_h(self, n, k):
        """
        Defined in the paper eq (17) (basic equation) + (19) (the usage equation)
        """

        return np.sqrt(2 * (2 * k + n + 1))

    def _generate_bn_mat_k(self, n, k):
        """
        Defined in the paper below eq (20)
        """

        return (n + 2 * k + 0.5) * (n + 2 * k + 1.5)

    def _generate_bn_mat_gamma_plus_1(self, n, k):
        """
        Defined in the paper eq (24)
        """

        return -(
            (np.square(k + n + 1) * self._generate_bn_mat_h(n, k))
            / ((2 * k + n + 1) * (2 * k + n + 2) * self._generate_bn_mat_h(n, k + 1))
        ) * ((k + 1) / (k + n + 1))

    def _generate_bn_mat_gamma_0(self, n, k):
        """
        Defined in the paper eq (24)
        """

        if n == 0:
            return 0.5
        return (2.0 * k * (k + 1) + n * (2 * k + n + 1)) / (
            (2 * k + n) * (2 * k + n + 2)
        )

    def _generate_bn_mat_gamma_minus_1(self, n, k):
        """
        Defined in the paper eq (24)
        """

        return -(
            (np.square(k) * self._generate_bn_mat_h(n, k))
            / ((2 * k + n + 1) * (2 * k + n) * self._generate_bn_mat_h(n, k - 1))
        ) * ((n + k) / k)

    def _generate_bn_mat_b_n_above_diagonal(self, n, k, c):
        """
        Defined in the paper eq (26)
        """

        return -np.square(c) * self._generate_bn_mat_gamma_minus_1(n, k)

    def _generate_bn_mat_b_n_on_diagonal(self, n, k, c):
        """
        Defined in the paper eq (26)
        """

        return -(
            self._generate_bn_mat_k(n, k)
            + np.square(c) * self._generate_bn_mat_gamma_0(n, k)
        )

    def _generate_bn_mat_b_n_below_diagonal(self, n, k, c):
        """
        Defined in the paper eq (26)
        """

        return -np.square(c) * self._generate_bn_mat_gamma_plus_1(n, k)
