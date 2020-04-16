import numpy as np

from numpy.linalg import norm
from numpy.linalg import solve
from scipy.linalg import block_diag
from scipy.special import jv

from aspire.utils.cell import Cell2D
from aspire.basis.ffb_2d import FFBBasis2D
from aspire.basis.basis_utils import lgwt

SCALAR_TYPES=(int, float, complex)

class BlkDiagMatrix:

    def __init__(self, nblocks, partition=None, dtype=np.float64):
        self.nblocks = nblocks
        #todo, warn
        if dtype == 'single':
            dtype = np.float32
        elif dtype == 'double':
            dtype = np.float64
        #else:
        #    enforce numpy types
        self.dtype = dtype
        self.data = [[]] * nblocks
        self._cached_blk_sizes = None
        if partition:
            self._cached_blk_sizes = np.array(partition)
            assert self._cached_blk_sizes.shape[1] == 2

    def __repr__(self):
        return "BlkDiagMatrix({},{})".format(repr(self.nblocks), repr(self.dtype))

    def _check_key(self, key):
        if not isinstance(key, int):
            raise TypeError("BlkDiagMatrix is indexed by integers. Got {}".format(repr(key)))
        elif key >= self.nblocks:
            raise IndexError("Key {} is outside nblocks {}".format(key, self.nblocks))
        elif key <= -self.nblocks:
            raise KeyError("Key {} is outside reversed nblocks {}".format(key, self.nblocks))
        return

    def copy(self):
        return BlkDiagMatrix.from_blk_diag(self.data)

    # Manually overload list methods
    #    we could make BlkDiagMatrix a subclass of list, but this and len() might be all we need....
    def __getitem__(self, key):
        self._check_key(key)
        return self.data[key]

    def __setitem__(self, key, value):
        self._check_key(key)
        self.data[key] = value

    def __len__(self):
        #return self.data.__len__
        return self.nblocks


    @staticmethod
    def zeros(blk_partition, dtype=np.float64):
        """
        Build a BlkDiagMatrix zeros matrix

        :param blk_partition: The matrix block partition in the form of a
        K-element list storing all shapes of K diagonal matrix blocks,
        where `blk_partition[i]` corresponds to the shape (number of rows and columns)
        of the `i` diagonal matrix block.
        :param dtype: The data type to set precision of diagonal matrix block.
        :return: A BlkDiagMatrix matrix consisting of `K` zero blocks.
        """

        n = len(blk_partition)
        A = BlkDiagMatrix(n, dtype=dtype)

        for i, blk_sz in enumerate(blk_partition):
            A[i] = np.zeros(blk_sz, dtype=dtype)
        return A

    @staticmethod
    def ones(blk_partition, dtype=np.float64):
        """
        Build a BlkDiagMatrix ones matrix

        :param blk_partition: The matrix block partition in the form of a
        K-element list storing all shapes of K diagonal matrix blocks,
        where `blk_partition[i]` corresponds to the shape (number of rows and columns)
        of the `i` diagonal matrix block.
        :param dtype: The data type to set precision of diagonal matrix block.
        :return: A BlkDiagMatrix matrix consisting of `K` ones blocks.
        """

        n = len(blk_partition)
        A = BlkDiagMatrix(n, dtype=dtype)

        for i, blk_sz in enumerate(blk_partition):
            A[i] = np.ones(blk_sz, dtype=dtype)
        return A

    @staticmethod
    def eye(blk_partition, dtype=np.float64):
        """
        Build a BlkDiagMatrix eye (identity) matrix

        :param blk_partition: The matrix block partition in the form of a
            K-element list storing all shapes of K diagonal matrix blocks,
            where `blk_partition[i]` corresponds to the shape (number of rows and columns)
            of the `i` diagonal matrix block.
        :param dtype: The data type to set the pricision of diagonal matrix block.
        :return: A BlkDiagMatrix matrix consisting of `K` eye (identity) blocks.
        """
        n = len(blk_partition)
        A = BlkDiagMatrix(n,dtype=dtype)

        for i, blk_sz in enumerate(blk_partition):
            rows, cols = blk_sz
            A[i] = np.eye(N=rows, M=cols, dtype=dtype)
        return A

    @staticmethod
    def get_partition(blk_diag):
        """
        Create a partition of block diagonal matrix

        :param blk_diag: A block diagonal matrix in the form of a list. Each
           element corresponds to a diagonal block.
        :return: The matrix block partition of `blk_diag` in the form of a
            K-element list storing all shapes of K diagonal matrix blocks,
        where `blk_partition[i]` corresponds to the shape (n rows and columns)
        of the `i` diagonal matrix block.
        """
        blk_partition = [[]]*len(blk_diag)
        for i, mat in enumerate(blk_diag):
            blk_partition[i] = np.shape(mat)
        return blk_partition

    @property
    def partition(self):
        """
        Return the partitions (block sizes) of this BlkDiagMatrix
        :return: The matrix block partition in the form of a
            K-element list storing all shapes of K diagonal matrix blocks,
            where `partition[i]` corresponds to the shape (number of rows and columns)
            of the `i` diagonal matrix block.
        """

        if self._cached_blk_sizes is None:
            blk_sizes = np.empty((self.nblocks,2))
            for i, blk in enumerate(self.data):
                blk_sizes[i] = np.shape(blk)
            self._cached_blk_sizes = blk_sizes
        return self._cached_blk_sizes

    def check_compatible(self, other):
        """
        Sanity check two BlkDiagMatrix matrices are compatible in size.
        :para other: The BlkDiagMatrix to compare with self
        :return: Returns True if no error is raised.
        """

        if not isinstance(other, BlkDiagMatrix):
            raise RuntimeError("Currently BlkDiagMatrix only interfaces with "
                               "its own instances, got {}".format(repr(other)))
        elif len(self) != len(other):
            raise RuntimeError('Number of blocks {} {} are not equal!'.format(len(self), len(other) ) )
        elif np.any(self.partition != other.partition):
            # be helpful and find the first one as an example
            for i, (a,b) in enumerate(zip(self.partition, other.partition)):
                if a != b:
                    break
            raise RuntimeError('{}th block of BlkDiagMatrix matrices are not same shape {} {}!'.format(i, a, b))

        return True

    @property
    def isnumeric(self):
        """
        Check a block diag matrix is numeric or not
        :return: Bool
        """
        #Note, I think numpy has a method for this somewhere...
        try:
            return 0 == self*0
        except:
            return False

    def add(self, other):
        """
        Define the element addition of BlkDiagMatrix matrix

        :param other: The rhs BlkDiagMatrix matrix
        :return:  BlkDiagMatrix matrix with elementwise sum equal to self + other.
        """
        if isinstance(other, SCALAR_TYPES):
            return self.scalar_add(other)

        self.check_compatible(other)

        C = BlkDiagMatrix(self.nblocks, dtype=self.dtype)

        # the can be done in parallel later (prange)
        for i in range(self.nblocks):
            C[i] = self[i] + other[i]

        return C

    def __add__(self, other):
        return self.add(other)

    def __iadd__(self, other):
        if isinstance(other, SCALAR_TYPES):
            return self.scalar_add(other, inplace=True)
        elif not self.check_compatible(other):
            raise TypeError('{} is not understood by BlkDiagMatrix.__iadd__'.format(repr(other)))
        for i in range(self.nblocks):
            self[i] += other[i]

        return self

    def __radd__(self, other):
        """ Convenient function for elementwise scalar addition """
        # Note, the case of BlkDiagMatrix_L + BlkDiagMatrix_R would be
        #   evaluated as L.add(R), so this is only for other
        #   Object + BlkDiagMatrix situations, namely scalars.
        # addition commutes
        return self.scalar_add(other)

    def scalar_add(self, scalar, inplace=False):
        """
        Define the element addition of BlkDiagMatrix matrix

        :param scalar: constant addend value
        :return:  BlkDiagMatrix matrix with elementwise sum equal to self + other.
        """
        assert any((isinstance(scalar, x) for x in SCALAR_TYPES))

        if not inplace:
            C = BlkDiagMatrix.from_blk_diag(self.data)
        else:
            C = self

        # the can be done in parallel later (prange)
        for i in range(self.nblocks):
            C[i] += scalar

        return C


    def sub(self, other):
        """
        Define the element subtraction of BlkDiagMatrix matrix

        :param other: The rhs BlkDiagMatrix matrix
        :return: A BlkDiagMatrix matrix with elementwise subraction equal to self - other.
        """
        if isinstance(other, SCALAR_TYPES):
            return self.scalar_sub(other)

        self.check_compatible(other)

        C = BlkDiagMatrix(self.nblocks, dtype=self.dtype)

        for i in range(self.nblocks):
            C[i] = self[i] - other[i]

        return C

    def __sub__(self, other):
        return self.sub(other)

    # Minus exists to match old implementation, deprecate?
    def minus(self, other):
        return self.__sub__(other)

    def __isub__(self, other):
        if isinstance(other, SCALAR_TYPES):
            return self.scalar_sub(other, inplace=True)
        self.check_compatible(other)

        for i in range(self.nblocks):
            self[i] -= other[i]

        return self

    def __rsub__(self, other):
        """ Convenient function for elementwise scalar subtraction """
        # Note, the case of BlkDiagMatrix_L - BlkDiagMatrix_R would be
        #   evaluated as L.sub(R), so this is only for other
        #   Object - BlkDiagMatrix situations, namely scalars.
        return -(self - other)

    def scalar_sub(self, scalar, inplace=False):
        """
        Define the elementwise subtraction from BlkDiagMatrix matrix

        :param scalar: constant subtractend value
        :return:  BlkDiagMatrix matrix with elementwise sum equal to self + other.
        """
        assert any((isinstance(scalar, x) for x in SCALAR_TYPES))

        if not inplace:
            C = BlkDiagMatrix.from_blk_diag(self.data)
        else:
            C = self

        # the can be done in parallel later (prange)
        for i in range(self.nblocks):
            C[i] -= scalar

        return C

    def matmul(self, other):
        """
        Compute the Matrix multiplication of two BlkDiagMatrix matrices

        :param other: The rhs BlkDiagMatrix matrix
        :return: A BlkDiagMatrix of self @ other.
        """
        if not isinstance(other, BlkDiagMatrix):
            raise RuntimeError(
                "Attempt BlkDiagMatrix Matrix multiplication "
                "(matmul,@) of non BlkDiagMatrix {}, try (*,mul)".format(
                    repr(other)))

        self.check_compatible(other)

        C = BlkDiagMatrix(self.nblocks, dtype=self.dtype)
        # note, we can do this in parallel
        for i in range(self.nblocks):
            C[i] = self[i] @ other[i]
        return C

    def __matmul__(self, other):
        return self.matmul(other)

    def __imatmul__(self, other):
        self.check_compatible(other)
        for i in range(self.nblocks):
            self[i] @= other[i]

        return self

    def mul(self, val):
        """
        Compute the numeric (elementwise) multiplication of a BlkDiagMatrix matrix and a scalar.

        :param other: The rhs BlkDiagMatrix matrix
        :return: A BlkDiagMatrix of self * other.
        """
        # TODO, probably I should only handle a few specfic instances instead...
        if isinstance(val, BlkDiagMatrix):
            raise RuntimeError("Attempt numeric multiplication (*,mul) of two BlkDiagMatrixs, try (matmul,@)")

        C = BlkDiagMatrix(self.nblocks, dtype=self.dtype)
        # note, we can do this in parallel
        for i in range(self.nblocks):
            C[i] = self[i] * val
        return C

    def __mul__(self, val):
        return self.mul(val)

    def __imul__(self, val):
        if isinstance(val, BlkDiagMatrix):
            raise RuntimeError("Attempt numeric multiplication (*,mul) of two BlkDiagMatrixs, try (matmul,@)")
        for i in range(self.nblocks):
            self[i] *= val

        return self

    def __rmul__(self, other):
        """ Convenience function, elementwise rmul commutes to mul """
        return self.mul(other)

    def __neg__(self):
        """
        Compute the unary negation of BlkDiagMatrix matrix.

        :return: A BlkDiagMatrix like self
        """
        C = BlkDiagMatrix(self.nblocks, dtype=self.dtype)
        for i in range(self.nblocks):
            C[i] = -self[i]
        return C

    def __abs__(self):
        """
        Compute the elementwise absolute value of BlkDiagMatrix matrix.

        :return: A BlkDiagMatrix like self
        """
        C = BlkDiagMatrix(self.nblocks, dtype=self.dtype)
        for i in range(self.nblocks):
            C[i] = np.abs(self[i])
        return C

    def __pow__(self, val):
        """
        Compute the elementwise power of BlkDiagMatrix matrix.

        :return: A BlkDiagMatrix like self.
        """

        C = BlkDiagMatrix(self.nblocks, dtype=self.dtype)
        for i in range(self.nblocks):
            C[i] = np.power(self[i], val)
        return C

    def __ipow__(self, val):
        """
        Compute the in place elementwise power of BlkDiagMatrix matrix.

        :return: self raised to power, elementwise.
        """

        for i in range(self.nblocks):
            self[i] **= val
        return self


    def norm(self, order=2):
        """
        Compute the norm of a BlkDiagMatrix matrix.

        :param order: Norm order, see np.norm. Defaults to order 2 norm.
        :return: The norm of the BlkDiagMatrix matrix
        """
        IMPLEMENTED_ORDERS  = (2,)
        if order not in IMPLEMENTED_ORDERS:
            raise NotImplementedError("Order {} not yet implemented, only {}".format(ord, IMPLEMENTED_ORDERS))

        return np.max([norm(blk, ord=order) for blk in self])

    def transpose(self):
        """
        Get the transpose matrix of a BlkDiagMatrix matrix

        :return: The corresponding transpose form as a BlkDiagMatrix matrix instance
        """

        T = BlkDiagMatrix(self.nblocks, dtype=self.dtype)

        for i in range(self.nblocks):
            T[i] = self[i].T

        return T

    @property
    def T(self):
        return self.transpose()


    def dense(self):
        """
        Convert list representation of BlkDiagMatrix matrix into full matrix

        :param blk_diag: The BlkDiagMatrix matrix

        :return: The BlkDiagMatrix matrix including the zero elements of
        non-diagonal blocks.
        """
        return block_diag(self.data)

    # not sure about the naming conventions atm
    def to_mat(self):
        return self.dense()

    @staticmethod
    def from_blk_diag(blk_diag, dtype=np.float64):
        """
        Convert full from list representation into BlkDiagMatrix

        :param blk_diag; The blk_diag representation in the form of a
            K-element list storing all shapes of K diagonal matrix blocks,
            where `blk_partition[i]` corresponds to the shape (number of rows and columns)
            of the `i` diagonal matrix block.

        :return: The BlkDiagMatrix matrix
        """

        # todo, maybe can dress this up a little better
        # get the partition (just sizes)
        partition = BlkDiagMatrix.get_partition(blk_diag)

        # instantiate an empty BlkDiagMatrix with that structure
        A = BlkDiagMatrix(len(partition), partition=partition, dtype=dtype)

        # set the data
        for i in range(A.nblocks):
            A.data[i] = blk_diag[i].copy()

        return A


    @staticmethod
    def from_mat(mat, blk_partition, dtype=np.float64):
        """
        Convert full block diagonal matrix into list representation
        :param mat; The full block diagonal matrix including the zero elements of
            non-diagonal blocks.
        :param blk_partition: The matrix block partition in the form of a
            K-element list storing all shapes of K diagonal matrix blocks,
            where `blk_partition[i]` corresponds to the shape (number of rows and columns)
            of the `i` diagonal matrix block.

        :return: The BlkDiagMatrix matrix
        """

        # todo, maybe can dress this up a little better
        A = BlkDiagMatrix(len(blk_partition), dtype=dtype)

        rows = blk_partition[:, 0]
        cols = blk_partition[:, 1]
        cellarray = Cell2D(rows, cols, dtype=mat.dtype)
        blk_diag = cellarray.mat2blk_diag(mat, rows, cols)
        A.data = BlkDiagMatrix.from_blk_diag(blk_diag)
        return A


    def solve(self, Y):
        """
        Solve a linear system involving a block diagonal matrix

        :param Y: The right-hand side in the linear system.  May be a matrix
            consisting of  coefficient vectors, in which case each column is
            solved for separately.
        :return: The result of solving the linear system formed by the matrix.
        """
        rows = np.array([np.size(mat_a, 0) for mat_a in self])
        if sum(rows) != np.size(Y, 0):
            raise RuntimeError('Sizes of matrix `self` and `Y` are not compatible.')

        vector = False
        if np.ndim(Y) == 1:
            Y = Y[:, np.newaxis]
            vector = True

        cols = np.array([np.size(Y, 1)])
        cellarray = Cell2D(rows, cols, dtype=Y.dtype)
        Y = cellarray.mat2cell(Y, rows, cols)
        X = []
        for i in range(0,self.nblocks):
            X.append(solve(self[i], Y[i]))
        X = np.concatenate(X, axis=0)

        if vector:
            X = X[:, 0]

        return X


    def apply(self, X):
        """
        Define the apply option of a block diagonal matrix with a matrix of coefficient vectors

        :param X: The coefficient matrix with each column is a coefficient vector
        :return: A matrix with new coefficient vectors
        """
        cols = np.array([np.size(mat, 1) for mat in self])

        if np.sum(cols) != np.size(X, 0):
            raise RuntimeError('Sizes of matrix `self` and `X` are not compatible.')

        vector = False
        if np.ndim(X) == 1:
            X = X[:, np.newaxis]
            vector = True

        rows = np.array([np.size(X, 1), ])
        cellarray = Cell2D(cols, rows, dtype=X.dtype)
        x_cell = cellarray.mat2cell(X, cols, rows)
        Y = []
        for i in range(0, self.nblocks):
            mat = self[i] @ x_cell[i]
            Y.append(mat)
        Y = np.concatenate(Y, axis=0)

        if vector:
            Y = Y[:, 0]

        return Y


def get_partition(blk_diag):
    """ Convenience wrapper of BlkDiagMatrix"""
    return BlkDiagMatrix.get_partition(blk_diag)

def filter_to_fb_mat(h_fun, fbasis):
    """
    Convert a nonradial function in k space into a basis representation

    :param h_fun: The function form in k space
    :param fbasis: The basis object for expanding
    :return: a BlkDiagMatrix matrix representation using the `fbasis` expansion
    """
    if not isinstance(fbasis, FFBBasis2D):
            raise NotImplementedError('Currently only fast FB method is supported')
    # Set same dimensions as basis object
    n_k = int(np.ceil(4 * fbasis.rcut * fbasis.kcut))
    n_theta = np.ceil(16 * fbasis.kcut * fbasis.rcut)
    n_theta = int((n_theta + np.mod(n_theta, 2)) / 2)
    # get 2D grid in polar coordinate
    k_vals, wts = lgwt(n_k, 0, 0.5)
    k, theta = np.meshgrid(k_vals, np.arange(n_theta) * 2 * np.pi / (2 * n_theta), indexing='ij')
    # Get function values in polar 2D grid and average out angle contribution
    omegax = k*np.cos(theta)
    omegay = k*np.sin(theta)
    omega = 2 * np.pi * np.vstack((omegax.flatten('C'), omegay.flatten('C')))
    h_vals2d = h_fun(omega).reshape(n_k, n_theta)
    h_vals = np.sum(h_vals2d, axis=1)/n_theta

    # Represent 1D function values in fbasis
    h_fb = BlkDiagMatrix(2 * fbasis.ell_max + 1, dtype=fbasis.dtype)
    ind = 0
    for ell in range(0, fbasis.ell_max+1):
        k_max = fbasis.k_max[ell]
        rmat = 2*k_vals.reshape(n_k, 1)*fbasis.r0[0:k_max, ell].T
        fb_vals = np.zeros_like(rmat)
        for ik in range(0, k_max):
            fb_vals[:, ik] = jv(ell, rmat[:, ik])
        fb_nrms = 1/np.sqrt(2)*abs(jv(ell+1, fbasis.r0[0:k_max, ell].T))/2
        fb_vals = fb_vals/fb_nrms
        h_fb_vals = fb_vals*h_vals.reshape(n_k, 1)
        h_fb_ell = fb_vals.T @ (h_fb_vals*k_vals.reshape(n_k, 1)*wts.reshape(n_k, 1))
        h_fb[ind] =h_fb_ell
        ind += 1
        if ell > 0:
            h_fb[ind] = h_fb[ind-1]
            ind += 1

    return h_fb
