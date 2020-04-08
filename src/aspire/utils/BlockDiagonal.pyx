import numpy as np

from numpy.linalg import norm
from numpy.linalg import solve
from scipy.linalg import block_diag
from scipy.special import jv

from aspire.utils.cell import Cell2D
from aspire.basis.ffb_2d import FFBBasis2D
from aspire.basis.basis_utils import lgwt

from aspire.utils.blk_diag_func import blk_diag_apply
from aspire.utils.blk_diag_func import blk_diag_solve

SCALAR_TYPES=(int, float, complex)

class BlockDiagonal:

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
        return "BlockDiagonal({},{})".format(repr(self.nblocks), repr(self.dtype))

    def _check_key(self, key):
        if not isinstance(key, int):
            raise TypeError("BlockDiagonal is indexed by integers. Got {}".format(repr(key)))
        elif key >= self.nblocks:
            raise IndexError("Key {} is outside nblocks {}".format(key, self.nblocks))
        elif key <= -self.nblocks:
            raise KeyError("Key {} is outside reversed nblocks {}".format(key, self.nblocks))
        return

    def __copy__(self):
        raise NotImplementedError("Not yet implemented")

    def __deepcopy__(self):
        raise NotImplementedError("Not yet implemented")

    # Manually overload list methods
    #    we could make BlockDiagonal a subclass of list, but this and len() might be all we need....
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
        Build a BlockDiagonal zeros matrix

        :param blk_partition: The matrix block partition in the form of a
        K-element list storing all shapes of K diagonal matrix blocks,
        where `blk_partition[i]` corresponds to the shape (number of rows and columns)
        of the `i` diagonal matrix block.
        :param dtype: The data type to set precision of diagonal matrix block.
        :return: A BlockDiagonal matrix consisting of `K` zero blocks.
        """

        n = len(blk_partition)
        A = BlockDiagonal(n, dtype=dtype)

        for i, blk_sz in enumerate(blk_partition):
            A[i] = np.zeros(blk_sz, dtype=dtype)
        return A

    @staticmethod
    def ones(blk_partition, dtype=np.float64):
        """
        Build a BlockDiagonal ones matrix

        :param blk_partition: The matrix block partition in the form of a
        K-element list storing all shapes of K diagonal matrix blocks,
        where `blk_partition[i]` corresponds to the shape (number of rows and columns)
        of the `i` diagonal matrix block.
        :param dtype: The data type to set precision of diagonal matrix block.
        :return: A BlockDiagonal matrix consisting of `K` ones blocks.
        """

        n = len(blk_partition)
        A = BlockDiagonal(n, dtype=dtype)

        for i, blk_sz in enumerate(blk_partition):
            A[i] = np.ones(blk_sz, dtype=dtype)
        return A

    @staticmethod
    def eye(blk_partition, dtype=np.float64):
        """
        Build a BlockDiagonal eye (identity) matrix

        :param blk_partition: The matrix block partition in the form of a
            K-element list storing all shapes of K diagonal matrix blocks,
            where `blk_partition[i]` corresponds to the shape (number of rows and columns)
            of the `i` diagonal matrix block.
        :param dtype: The data type to set the pricision of diagonal matrix block.
        :return: A BlockDiagonal matrix consisting of `K` eye (identity) blocks.
        """
        n = len(blk_partition)
        A = BlockDiagonal(n,dtype=dtype)

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
        Return the partitions (block sizes) of this BlockDiagonal
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
        Sanity check two BlockDiagonal matrices are compatible in size.
        :para other: The BlockDiagonal to compare with self
        :return: Returns True if no error is raised.
        """

        if not isinstance(other, BlockDiagonal):
            raise RuntimeError("Currently BlockDiagonal only interfaces with "
                               "its own instances, got {}".format(repr(other)))
        elif len(self) != len(other):
            raise RuntimeError('Number of blocks {} {} are not equal!'.format(len(self), len(other) ) )
        elif np.any(self.partition != other.partition):
            # be helpful and find the first one as an example
            for i, (a,b) in enumerate(zip(self.partition, other.partition)):
                if a != b:
                    break
            raise RuntimeError('{}th block of BlockDiagonal matrices are not same shape {} {}!'.format(i, a, b))

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
        Define the element addition of BlockDiagonal matrix

        :param other: The rhs BlockDiagonal matrix
        :return:  BlockDiagonal matrix with elementwise sum equal to self + other.
        """
        if isinstance(other, SCALAR_TYPES):
            return self.scalar_add(other)

        self.check_compatible(other)

        C = BlockDiagonal(self.nblocks, dtype=self.dtype)

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
            raise TypeError('{} is not understood by BlockDiagonal.__iadd__'.format(repr(other)))
        for i in range(self.nblocks):
            self[i] += other[i]

        return self

    def __radd__(self, other):
        """ Convenient function for elementwise scalar addition """
        # Note, the case of BlockDiagonal_L + BlockDiagonal_R would be
        #   evaluated as L.add(R), so this is only for other
        #   Object + BlockDiagonal situations, namely scalars.
        # addition commutes
        return self.scalar_add(other)

    def scalar_add(self, scalar, inplace=False):
        """
        Define the element addition of BlockDiagonal matrix

        :param scalar: constant addend value
        :return:  BlockDiagonal matrix with elementwise sum equal to self + other.
        """
        assert any((isinstance(scalar, x) for x in SCALAR_TYPES))

        if not inplace:
            C = BlockDiagonal.from_blk_diag(self.data)
        else:
            C = self

        # the can be done in parallel later (prange)
        for i in range(self.nblocks):
            C[i] += scalar

        return C


    def sub(self, other):
        """
        Define the element subtraction of BlockDiagonal matrix

        :param other: The rhs BlockDiagonal matrix
        :return: A BlockDiagonal matrix with elementwise subraction equal to self - other.
        """
        if isinstance(other, SCALAR_TYPES):
            return self.scalar_sub(other)

        self.check_compatible(other)

        C = BlockDiagonal(self.nblocks, dtype=self.dtype)

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
        # Note, the case of BlockDiagonal_L - BlockDiagonal_R would be
        #   evaluated as L.sub(R), so this is only for other
        #   Object - BlockDiagonal situations, namely scalars.
        return -(self - other)

    def scalar_sub(self, scalar, inplace=False):
        """
        Define the elementwise subtraction from BlockDiagonal matrix

        :param scalar: constant subtractend value
        :return:  BlockDiagonal matrix with elementwise sum equal to self + other.
        """
        assert any((isinstance(scalar, x) for x in SCALAR_TYPES))

        if not inplace:
            C = BlockDiagonal.from_blk_diag(self.data)
        else:
            C = self

        # the can be done in parallel later (prange)
        for i in range(self.nblocks):
            C[i] -= scalar

        return C

    def matmul(self, other):
        """
        Compute the Matrix multiplication of two BlockDiagonal matrices

        :param other: The rhs BlockDiagonal matrix
        :return: A BlockDiagonal of self @ other.
        """
        if not isinstance(other, BlockDiagonal):
            raise RuntimeError(
                "Attempt BlockDiagonal Matrix multiplication "
                "(matmul,@) of non BlockDiagonal {}, try (*,mul)".format(
                    repr(other)))

        self.check_compatible(other)

        C = BlockDiagonal(self.nblocks, dtype=self.dtype)
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
        Compute the numeric (elementwise) multiplication of a BlockDiagonal matrix and a scalar.

        :param other: The rhs BlockDiagonal matrix
        :return: A BlockDiagonal of self * other.
        """
        # TODO, probably I should only handle a few specfic instances instead...
        if isinstance(val, BlockDiagonal):
            raise RuntimeError("Attempt numeric multiplication (*,mul) of two BlockDiagonals, try (matmul,@)")

        C = BlockDiagonal(self.nblocks, dtype=self.dtype)
        # note, we can do this in parallel
        for i in range(self.nblocks):
            C[i] = self[i] * val
        return C

    def __mul__(self, val):
        return self.mul(val)

    def __imul__(self, val):
        if isinstance(val, BlockDiagonal):
            raise RuntimeError("Attempt numeric multiplication (*,mul) of two BlockDiagonals, try (matmul,@)")
        for i in range(self.nblocks):
            self[i] *= val

        return self

    def __rmul__(self, other):
        """ Convenience function, elementwise rmul commutes to mul """
        return self.mul(other)

    def __neg__(self):
        """
        Compute the unary negation of BlockDiagonal matrix.

        :return: A BlockDiagonal like self
        """
        C = BlockDiagonal(self.nblocks, dtype=self.dtype)
        for i in range(self.nblocks):
            C[i] = -self[i]
        return C

    def __abs__(self):
        """
        Compute the elementwise absolute value of BlockDiagonal matrix.

        :return: A BlockDiagonal like self
        """
        C = BlockDiagonal(self.nblocks, dtype=self.dtype)
        for i in range(self.nblocks):
            C[i] = np.abs(self[i])
        return C

    def norm(self, order=2):
        """
        Compute the norm of a BlockDiagonal matrix.

        :param order: Norm order, see np.norm. Defaults to order 2 norm.
        :return: The norm of the BlockDiagonal matrix
        """
        IMPLEMENTED_ORDERS  = (2,)
        if order not in IMPLEMENTED_ORDERS:
            raise NotImplementedError("Order {} not yet implemented, only {}".format(ord, IMPLEMENTED_ORDERS))

        return np.max([norm(blk, ord=order) for blk in self])

    def transpose(self):
        """
        Get the transpose matrix of a BlockDiagonal matrix

        :return: The corresponding transpose form as a BlockDiagonal matrix instance
        """

        T = BlockDiagonal(self.nblocks, dtype=self.dtype)

        for i in range(self.nblocks):
            T[i] = self[i].T

        return T

    @property
    def T(self):
        return self.transpose()


    def dense(self):
        """
        Convert list representation of BlockDiagonal matrix into full matrix

        :param blk_diag: The BlockDiagonal matrix

        :return: The BlockDiagonal matrix including the zero elements of
        non-diagonal blocks.
        """
        return block_diag(self.data)

    # not sure about the naming conventions atm
    def to_mat(self):
        return self.dense()

    @staticmethod
    def from_blk_diag(blk_diag, dtype=np.float64):
        """
        Convert full from list representation into BlockDiagonal

        :param blk_diag; The blk_diag representation in the form of a
            K-element list storing all shapes of K diagonal matrix blocks,
            where `blk_partition[i]` corresponds to the shape (number of rows and columns)
            of the `i` diagonal matrix block.

        :return: The BlockDiagonal matrix
        """

        # todo, maybe can dress this up a little better
        # get the partition (just sizes)
        partition = BlockDiagonal.get_partition(blk_diag)

        # instantiate an empty BlockDiagonal with that structure
        A = BlockDiagonal(len(partition), partition=partition, dtype=dtype)

        # set the data
        A.data = blk_diag

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

        :return: The BlockDiagonal matrix
        """

        # todo, maybe can dress this up a little better
        A = BlockDiagonal(len(blk_partition), dtype=dtype)

        rows = blk_partition[:, 0]
        cols = blk_partition[:, 1]
        cellarray = Cell2D(rows, cols, dtype=mat.dtype)
        blk_diag = cellarray.mat2blk_diag(mat, rows, cols)
        A.data = BlockDiagonal.from_blk_diag(blk_diag)
        return A


    def solve(self, y):
        # todo
        X = blk_diag_solve(self.data, y)
        return X

    def apply(self, x):
        # todo
        Y = blk_diag_apply(self.data, x)
        return Y


def get_partition(blk_diag):
    """ Convenience wrapper of BlockDiagonal"""
    return BlockDiagonal.get_partition(blk_diag)

def filter_to_fb_mat(h_fun, fbasis):
    """
    Convert a nonradial function in k space into a basis representation

    :param h_fun: The function form in k space
    :param fbasis: The basis object for expanding
    :return: a matrix representation using the `fbasis` expansion
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
    h_fb = []
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
        h_fb.append(h_fb_ell)
        ind = ind+1
        if ell > 0:
            h_fb.append(h_fb[ind-1])
            ind = ind+1

    return h_fb
