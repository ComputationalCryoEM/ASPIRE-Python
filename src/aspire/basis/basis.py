import logging

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from aspire.basis.basis_utils import num_besselj_zeros
from aspire.utils import ensure, mdim_mat_fun_conj
from aspire.utils.matlab_compat import m_reshape

logger = logging.getLogger(__name__)


class Basis:
    """
    Define a base class for expanding 2D particle images and 3D structure volumes

    """

    def __init__(self, size, ell_max=None, dtype=np.float32):
        """
        Initialize an object for the base of basis class

        :param size: The size of the vectors for which to define the basis.
            Currently only square images and cubic volumes are supported.
        :ell_max: The maximum order ell of the basis elements. If no input
            (= None), it will be set to np.Inf and the basis includes all
            ell such that the resulting basis vectors are concentrated
            below the Nyquist frequency (default Inf).
        """
        if ell_max is None:
            ell_max = np.inf

        ndim = len(size)
        nres = size[0]
        self.sz = size
        self.nres = nres
        self.count = 0
        self.ell_max = ell_max
        self.ndim = ndim
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.float32, np.float64):
            raise NotImplementedError(
                "Currently only implemented for float32 and float64 types"
            )

        self._build()

    def _getfbzeros(self):
        """
        Generate zeros of Bessel functions
        """
        # get upper_bound of zeros of Bessel functions
        upper_bound = min(self.ell_max + 1, 2 * self.nres + 1)

        # List of number of zeros
        n = []
        # List of zero values (each entry is an ndarray; all of possibly different lengths)
        zeros = []

        # generate zeros of Bessel functions for each ell
        for ell in range(upper_bound):
            _n, _zeros = num_besselj_zeros(
                ell + (self.ndim - 2) / 2, self.nres * np.pi / 2
            )
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
            zeros[i] = np.hstack(
                (z, np.zeros(max_num_zeros - len(z), dtype=self.dtype))
            )

        self.r0 = m_reshape(np.hstack(zeros), (-1, self.ell_max + 1))

    def _build(self):
        """
        Build the internal data structure to represent basis
        """
        raise NotImplementedError("subclasses must implement this")

    def indices(self):
        """
        Create the indices for each basis function
        """
        raise NotImplementedError("subclasses must implement this")

    def _precomp(self):
        """
        Precompute the basis functions at defined sample points
        """
        raise NotImplementedError("subclasses must implement this")

    def norms(self):
        """
        Calculate the normalized factors of basis functions
        """
        raise NotImplementedError("subclasses must implement this")

    def evaluate(self, v):
        """
        Evaluate coefficient vector in basis

        :param v: A coefficient vector (or an array of coefficient vectors)
            to be evaluated. The first dimension must equal `self.count`.
        :return: The evaluation of the coefficient vector(s) `v` for this basis.
            This is an array whose first dimensions equal `self.z` and the
            remaining dimensions correspond to dimensions two and higher of `v`.
        """
        raise NotImplementedError("subclasses must implement this")

    def evaluate_t(self, v):
        """
        Evaluate coefficient in dual basis

        :param v: The coefficient array to be evaluated. The first dimensions
            must equal `self.sz`.
        :return: The evaluation of the coefficient array `v` in the dual
            basis of `basis`.
            This is an array of vectors whose first dimension equals `self.count`
            and whose remaining dimensions correspond to higher dimensions of `v`.
        """
        raise NotImplementedError("Subclasses should implement this")

    def mat_evaluate(self, V):
        """
        Evaluate coefficient matrix in basis

        :param V: A coefficient matrix of size `self.count`-by-
            `self.count` to be evaluated.
        :return: A multidimensional matrix of size `self.sz`-by
            -`self.sz` corresponding to the evaluation of `V` in
            this basis.
        """
        return mdim_mat_fun_conj(V, 1, len(self.sz), self.evaluate)

    def mat_evaluate_t(self, X):
        """
        Evaluate coefficient matrix in dual basis

        :param X: The coefficient array of size `self.sz`-by-`self.sz`
            to be evaluated.
        :return: The evaluation of `X` in the dual basis. This is
            `self.count`-by-`self.count`. matrix.
            If `V` is a matrix of size `self.count`-by-`self.count`,
            `B` is the change-of-basis matrix of `basis`, and `x` is a
            multidimensional matrix of size `basis.sz`-by-`basis.sz`, the
            function calculates V = B' * X * B, where the rows of `B`, rows
            of 'X', and columns of `X` are read as vectorized arrays.
        """
        return mdim_mat_fun_conj(X, len(self.sz), 1, self.evaluate_t)

    def expand(self, x):
        """
        Obtain coefficients in the basis from those in standard coordinate basis

        This is a similar function to evaluate_t but with more accuracy by using
        the cg optimizing of linear equation, Ax=b.

        :param x: An array whose last two or three dimensions are to be expanded
            the desired basis. These dimensions must equal `self.sz`.
        :return : The coefficients of `v` expanded in the desired basis.
            The last dimension of `v` is with size of `count` and the
            first dimensions of the return value correspond to
            those first dimensions of `x`.

        """
        # ensure the first dimensions with size of self.sz
        sz_roll = x.shape[: -self.ndim]

        x = x.reshape((-1, *self.sz))

        ensure(
            x.shape[-self.ndim :] == self.sz,
            f"Last {self.ndim} dimensions of x must match {self.sz}.",
        )

        operator = LinearOperator(
            shape=(self.count, self.count),
            matvec=lambda v: self.evaluate_t(self.evaluate(v)),
            dtype=self.dtype,
        )

        # TODO: (from MATLAB implementation) - Check that this tolerance make sense for multiple columns in v
        tol = 10 * np.finfo(x.dtype).eps
        logger.info("Expanding array in basis")

        # number of image samples
        n_data = x.shape[0]
        v = np.zeros((n_data, self.count), dtype=x.dtype)

        for isample in range(0, n_data):
            b = self.evaluate_t(x[isample]).T
            # TODO: need check the initial condition x0 can improve the results or not.
            v[isample], info = cg(operator, b, tol=tol, atol=0)
            if info != 0:
                raise RuntimeError("Unable to converge!")

        # return v coefficients with the last dimension of self.count
        v = v.reshape((-1, *sz_roll))
        return v
