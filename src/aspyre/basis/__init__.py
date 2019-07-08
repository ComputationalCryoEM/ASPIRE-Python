import logging
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from aspyre.utils import ensure
from aspyre.utils.matrix import mdim_mat_fun_conj, roll_dim, unroll_dim
from aspyre.utils.matlab_compat import m_reshape
from aspyre.basis.basis_func import num_besselj_zeros

logger = logging.getLogger(__name__)


class Basis:
    """
    Define a base class of mathematical basis for mapping 2D particle images
    and 3D structure volumes.

    """

    def __init__(self, size, ell_max=None):
        if ell_max is None:
            ell_max = np.inf

        d = len(size)
        N = size[0]
        self.sz = size
        self.N = N
        self.basis_count = 0
        self.ell_max = ell_max
        self.d = d

        self._build()

    def _getfbzeros(self):

        # get upper_bound of zeros of Bessel functions
        upper_bound = min(self.ell_max + 1, 2 * self.N + 1)

        # List of number of zeros
        n = []
        # List of zero values (each entry is an ndarray; all of possibly different lengths)
        zeros = []

        # generate zeros of Bessel functions for each ell
        for ell in range(upper_bound):
            _n, _zeros = num_besselj_zeros(ell + (self.d - 2) / 2, self.N * np.pi / 2)
            if _n == 0:
                break
            else:
                n.append(_n)
                zeros.append(_zeros)

        #  get maximum number of ell
        self.ell_max = len(n) - 1

        #  set the maximum of k for each ell
        self.k_max = np.array(n, dtype=int)

        max_num_zeros = max(len(z) for z in zeros)
        for i, z in enumerate(zeros):
            zeros[i] = np.hstack((z, np.zeros(max_num_zeros - len(z))))

        self.r0 = m_reshape(np.hstack(zeros), (-1, self.ell_max + 1))

    def _build(self):
        raise NotImplementedError('subclasses must implement this')

    def indices(self):
        raise NotImplementedError('subclasses must implement this')

    def precomp(self):
        raise NotImplementedError('subclasses must implement this')

    def norms(self):
        raise NotImplementedError('subclasses must implement this')

    def expand(self, v):
        raise NotImplementedError('subclasses must implement this')

    def evaluate(self, v):
        """
        Evaluate coefficient vector in basis
        :param v: A coefficient vector (or an array of coefficient vectors) to be evaluated.
            The first dimension must equal `self.basis_count`.
        :return: The evaluation of the coefficient vector(s) `v` for this basis.
            This is an array whose first dimensions equal `self.z` and the remaining dimensions correspond to
            dimensions two and higher of `v`.
        """
        raise NotImplementedError('subclasses must implement this')

    def evaluate_t(self, v):
        """
        Evaluate coefficient in dual basis
        :param v: The coefficient array to be evaluated. The first dimensions must equal `self.sz`.
        :return: The evaluation of the coefficient array `v` in the dual basis of `basis`.
            This is an array of vectors whose first dimension equals `self.basis_count` and whose remaining dimensions
            correspond to higher dimensions of `v`.
        """
        raise NotImplementedError('Subclasses should implement this')

    def mat_evaluate(self, V):
        """
        Evaluate coefficient matrix in basis
        :param V: A coefficient matrix of size `self.basis_count`-by-`self.basis_count` to be evaluated.
        :return: A multidimensional matrix of size `self.sz`-by-`self.sz` corresponding to the evaluation of `V` in
            this basis.
        """
        return mdim_mat_fun_conj(V, 1, len(self.sz), self.evaluate)

    def mat_evaluate_t(self, X):
        """
        Evaluate coefficient matrix in dual basis
        :param X: The coefficient array of size `self.sz`-by-`self.sz` to be evaluated.
        :return: The evaluation of `X` in the dual basis. This is `self.count`-by-`self.count`. matrix

        If `V` is a matrix of size `self.basis_count`-by-`self.basis_count`, `B` is the change-of-basis matrix of
        `basis`, and `x` is a multidimensional matrix of size `basis.sz`-by-`basis.sz`, the function calculates

        V = B' * X * B

        where the rows of `B`, rows of 'X', and columns of `X` are read as vectorized arrays.
        """
        return mdim_mat_fun_conj(X, len(self.sz), 1, self.evaluate_t)

    def expand(self, v):
        """
        Expand array in basis

        If `v` is a matrix of size `basis.ct`-by-..., `B` is the change-of-basis matrix of this basis, and `x` is a
        matrix of size `self.sz`-by-..., the function calculates

            v = (B' * B)^(-1) * B' * x

        where the rows of `B` and columns of `x` are read as vectorized arrays.

        :param v: An array whose first few dimensions are to be expanded in this basis.
            These dimensions must equal `self.sz`.
        :return: The coefficients of `v` expanded in this basis. If more than one array of size `self.sz` is found in
            `v`, the second and higher dimensions of the return value correspond to those higher dimensions of `v`.

        .. seealso:: evaluate
        """
        ensure(v.shape[:self.d] == self.sz, f'First {self.d} dimensions of v must match {self.sz}.')

        v, sz_roll = unroll_dim(v, self.d + 1)
        b = self.evaluate_t(v)
        operator = LinearOperator(
            shape=(self.basis_count, self.basis_count),
            matvec=lambda x: self.evaluate_t(self.evaluate(x))
        )

        # TODO: (from MATLAB implementation) - Check that this tolerance make sense for multiple columns in v
        tol = 10 * np.finfo(v.dtype).eps
        logger.info('Expanding array in basis')
        v, info = cg(operator, b, tol=tol)

        if info != 0:
            raise RuntimeError('Unable to converge!')

        v = roll_dim(v, sz_roll)

        return v

    def expand_t(self, v):
        """
        Expand array in dual basis

        If `v` is a matrix of size `basis.ct`-by-..., `B` is the change-of-basis matrix of this basis, and `x` is a
        matrix of size `self.sz`-by-..., the function calculates

            x = (B * B')^(-1) * B * v

        where the rows of `B` and columns of `x` are read as vectorized arrays.

        :param v: An array whose first dimension is to be expanded in this basis's dual.
            This dimension must be equal to `self.basis_count`.
        :return: The coefficients of `v` expanded in the dual of `basis`. If more than one vector is supplied in `v`,
            the higher dimensions of the return value correspond to second and higher dimensions of `v`.

        .. seealso:: expand
        """
        raise NotImplementedError('subclasses should implement this')
