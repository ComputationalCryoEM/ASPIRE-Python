import numpy as np
from scipy.linalg import eigh_tridiagonal
from aspire.em_classavg.image_denoising.image_denoising.PSWF2D.BN.BN_init_utils import generate_bn_mat


class BN:
    """
    Class to compute the BN matrix based on the paper TODO: put Yoel's paper title sec 3.2.
    and compute the eig vectors of this matrix.
    Properties:
        TODO

    """
    def __init__(self, n, bandlimit, approx_length):
        self.diagonal, self.off_diagonal = generate_bn_mat(n, bandlimit, approx_length)

    def get_eig_vectors(self):
        """
        :return: v: (M,M) ndarray
            The normalized eigenvectors corresponding to the eigenvalues, v[:, i] is corresponding to the w[i].
            In each eigenvector v[:, i], v[argmax(abs(v[:, i])), i] >= 0.
                w: (M,) ndarray
            The eigenvalues in descending order.
        """

        w, v = eigh_tridiagonal(self.diagonal, self.off_diagonal, select='a')

        sorted_idx = np.argsort(-w)
        v = v[:, sorted_idx]
        w = w[sorted_idx]

        a = np.argmax(np.absolute(v), axis=0)
        b = np.array([np.sign(v[a[k], k]) for k in range(len(v))])
        v = v * b

        return v, w

    def dense_mat(self):
        """
        :return: mat: (M,M), ndarray
            The full BN matrix .
        """

        diagonal = self.diagonal
        off_diagonal = self.off_diagonal
        m = len(diagonal)

        mat = np.zeros((m, m))
        mat[0, 0] = diagonal[0]

        for i in range(m - 1):
            mat[i, i + 1] = off_diagonal[i]
            mat[i + 1, i] = off_diagonal[i]
            mat[i + 1, i + 1] = diagonal[i + 1]

        return mat

    def shape(self):
        """
        :return: tuple: (n, n)
            The dense matrix shape
        """

        return len(self.diagonal), len(self.diagonal)
