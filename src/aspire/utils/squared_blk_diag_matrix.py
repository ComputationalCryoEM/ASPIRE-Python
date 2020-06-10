"""
Define a SquaredBlkDiagMatrix module which implements operations for
block diagonal matrices as used by ASPIRE.
"""

import pdb
#import numpy as xp
import cupy as xp

from numpy.linalg import norm
from numpy.linalg import solve
from scipy.linalg import block_diag
from scipy.special import jv

from aspire.utils.cell import Cell2D
from aspire.basis.ffb_2d import FFBBasis2D
from aspire.basis.basis_utils import lgwt

from aspire.utils.blk_diag_matrix import BlkDiagMatrix

class SquaredBlkDiagMatrix(BlkDiagMatrix):
    """
    Define a SquaredBlkDiagMatrix class which implements operations for
    block diagonal matrices as used by ASPIRE.

    Currently SquaredBlkDiagMatrix is implemented only for square blocks.
    While in the future this can be extended, at this time assigning
    a non square array will raise NotImplementedError.
    """

    __array_ufunc__ = None

    def __init__(self, partition, dtype=xp.float64):
        self.nblocks = len(partition)
        self.dtype = xp.dtype(dtype)
        self.max_blk_size = 0
        self.must_update = True
        if len(partition):
            self.max_blk_size = 0
            for b in partition:
                self.max_blk_size = max(self.max_blk_size, int(b[0]))
            self._cached_blk_sizes = xp.array(partition)
            assert self._cached_blk_sizes.shape[1] == 2
            assert all([SquaredBlkDiagMatrix.__check_square(s) for s in partition])
            self.data  = None
            self._data = [None] * len(partition)

    def copy(self):
        """
        Returns new BlkDiagMatrix which is a copy of `self`.

        :return BlkDiagMatrix like self
        """
        C = SquaredBlkDiagMatrix(self.partition, dtype=self.dtype)

        C.data = self.data.copy()
        C._data = self._data.copy()

        return C

    def __check_size_compatible(self, other):
        """
        Sanity check two SquaredBlkDiagMatrix instances are compatible in size.

        :param other: The SquaredBlkDiagMatrix to compare with self.
        """
        if xp.any(self.partition != other.partition):
            # be helpful and find the first one as an example
            for i, (a, b) in enumerate(zip(self.partition, other.partition)):
                if not xp.allclose(a,b):
                    break
            raise RuntimeError(
                'Block i={} of SquaredBlkDiagMatrix instances are '
                'not same shape {} {}'.format(i, a, b))

    def __check_dtype_compatible(self, other):
        """
        Sanity check two BlkDiagMatrix instances are compatible in dtype.

        :param other: The BlkDiagMatrix to compare with self.
        """

        if self.dtype != other.dtype:
            raise RuntimeError('SquaredBlkDiagMatrix received different types,'
                               ' {} and {}.  Please validate and cast'
                               ' as appropriate.'.format(
                                   self.dtype, other.dtype))

    def __check_compatible(self, other):
        """
        Sanity check two BlkDiagMatrix instances are compatible in size.

        :param other: The BlkDiagMatrix to compare with self.
        """

        if not isinstance(other, BlkDiagMatrix):
            raise NotImplementedError(
                "Currently BlkDiagMatrix only interfaces "
                "with its own instances, got {}".format(repr(other)))

        if len(self) != len(other):
            raise RuntimeError('Number of blocks {} {} are not equal.'.format(
                len(self), len(other)))

        self.__check_size_compatible(other)
        self.__check_dtype_compatible(other)

    def __getitem__(self, key):
        """
        Convenience wrapper, getter on self.data.
        Use data without 0-padding (non squared)
        update nopadded_data if needed
        """
        if self.must_update:
            for i,blk in enumerate(self.data):
                rownb = self._cached_blk_sizes[i][0]
                colnb = self._cached_blk_sizes[i][1]
                self._data[i] = blk[:rownb,:colnb]
            self.must_update = False
        return self._data[key]


    def __setitem__(self, key, value):
        """
        Convenience wrapper, setter on self.data.
        """

        SquaredBlkDiagMatrix.__check_square(value.shape)
        self._data[key] = value
        self.reset_cache()

    def add(self, other, inplace=False):
        """
        Define the elementwise addition of BlkDiagMatrix instance.

        :param other: The rhs BlkDiagMatrix instance.
        :param inplace: Boolean, when set to True change values in place,
        otherwise return a new instance (default).
        :return:  BlkDiagMatrix instance with elementwise sum equal
        to self + other.
        """

        if self._is_scalar_type(other):
            return self.__scalar_add(other, inplace=inplace)

        self.__check_compatible(other)

        if inplace:
            self.data = self.data + other.data
            C = self
        else:
            C = SquaredBlkDiagMatrix(self.partition, dtype=self.dtype)
            C.data = self.data + other.data
        C.must_update = True

        return C

    def __add__(self, other):
        """
        Operator overloading for addition.
        """

        return self.add(other)

    def __iadd__(self, other):
        """
        Operator overloading for in-place addition.
        """

        return self.add(other, inplace=True)

    def __radd__(self, other):
        """
        Convenience function for elementwise scalar addition.
        """

        return self.add(other)

    def __scalar_add(self, scalar, inplace=False):
        """
        Define the element addition of BlkDiagMatrix instance.

        :param scalar: constant addend value.
        :param inplace: Boolean, when set to True change values in place,
        otherwise return a new instance (default).
        :return:  BlkDiagMatrix instance with elementwise sum equal
        to self + other.
        """

        assert self._is_scalar_type(scalar)

        if inplace:
            C = self
        else:
            C = self.copy()

        C.data = C.data + scalar
        C.must_update = True

        return C

    def sub(self, other, inplace=False):
        """
        Define the element subtraction of BlkDiagMatrix instance.

        :param other: The rhs BlkDiagMatrix instance.
        :param inplace: Boolean, when set to True change values in place,
        otherwise return a new instance (default).
        :return: A BlkDiagMatrix instance with elementwise subraction equal to
         self - other.
        """

        if self._is_scalar_type(other):
            return self.__scalar_sub(other, inplace=inplace)

        self.__check_compatible(other)
        if inplace:
            self.data = self.data - other.data
            C = self
        else:
            C = SquaredBlkDiagMatrix(self.partition, dtype=self.dtype)
            C.data = self.data - other.data
        C.must_update = True

        return C

    def __sub__(self, other):
        """
        Operator overloading for subtraction.
        """

        return self.sub(other)

    def __isub__(self, other):
        """
        Operator overloading for in-place subtraction.
        """
        if self._is_scalar_type(other):
            return self.__scalar_sub(other, inplace=True)
        return self.sub(other, inplace=True)

    def __rsub__(self, other):
        """
        Convenience function for elementwise scalar subtraction.
        """

        # Note, the case of BlkDiagMatrix_L - BlkDiagMatrix_R would be
        #   evaluated as L.sub(R), so this is only for other
        #   Object - BlkDiagMatrix situations, namely scalars.

        return -(self - other)

    def __scalar_sub(self, scalar, inplace=False):
        """
        Define the elementwise subtraction from BlkDiagMatrix instance.

        :param scalar: constant subtractend value.
        :param inplace: bool, when false (default) return new instance.
        :return:  BlkDiagMatrix instance with elementwise sum equal to
         self + other.
        """

        assert self._is_scalar_type(scalar)

        if inplace:
            C = self
        else:
            C = self.copy()

        C.data = C.data - scalar
        C.must_update = True

        return C

    def matmul(self, other, inplace=False):
        """
        Compute the matrix multiplication of two SquaredBlkDiagMatrix instances.

        :param other: The rhs SquaredBlkDiagMatrix instance.
        :param inplace: Boolean, when set to True change values in place,
        otherwise return a new instance (default).
        :return: A SquaredBlkDiagMatrix of self @ other.
        """

        if not isinstance(other, SquaredBlkDiagMatrix):
            raise RuntimeError(
                "Attempt SquaredBlkDiagMatrix matrix multiplication "
                "(matmul,@) of non SquaredBlkDiagMatrix {}, try (*,mul)".format(
                    repr(other)))

        self.__check_compatible(other)

        if inplace:
            self.data = self.data @ other.data
            C = self
        else:
            C = SquaredBlkDiagMatrix(self.partition, dtype=self.dtype)
            C.data = self.data @ other.data
        self.must_update = True

        return C

    def __matmul__(self, other):
        """
        Operator overload for matrix multiply of SquaredBlkDiagMatrix instances.
        """

        return self.matmul(other)

    def __imatmul__(self, other):
        """
        Operator overload for in-place matrix multiply of SquaredBlkDiagMatrix
         instances.
        """
        return self.matmul(other, inplace=True)

    def mul(self, val, inplace=False):
        """
        Compute the numeric multiplication of a SquaredBlkDiagMatrix instance and a
        scalar.

        :param other: The rhs SquaredBlkDiagMatrix instance.
        :param inplace: Boolean, when set to True change values in place,
        otherwise return a new instance (default).
        :return: A SquaredBlkDiagMatrix of self * other.
        """

        if isinstance(val, SquaredBlkDiagMatrix):
            raise RuntimeError("Attempt numeric multiplication (*,mul) of two "
                               "SquaredBlkDiagMatrix instances, try (matmul,@).")

        elif not self._is_scalar_type(val):
            raise RuntimeError("Attempt numeric multiplication (*,mul) of a "
                               "SquaredBlkDiagMatrix and {}.".format(
                                   type(val)))

        if inplace:
            self.data = self.data * val

            C = self
        else:
            C = SquaredBlkDiagMatrix(self.partition, dtype=self.dtype)

            C.data = self.data * val
        self.must_update = True

        return C

    def __mul__(self, val):
        """
        Operator overload for SquaredBlkDiagMatrix scalar multiply.
        """

        return self.mul(val)

    def __imul__(self, val):
        """
        Operator overload for in-place SquaredBlkDiagMatrix scalar multiply.
        """

        return self.mul(val, inplace=True)

    def __rmul__(self, other):
        """
        Convenience function, elementwise rmul commutes to mul.
        """

        return self.mul(other)

    def neg(self):
        """
        Compute the unary negation of SquaredBlkDiagMatrix instance.

        :return: A SquaredBlkDiagMatrix like self.
        """

        C = SquaredBlkDiagMatrix(self.partition, dtype=self.dtype)
        C.data = -self.data
        self.must_update = True

        return C

    def __neg__(self):
        """
        Operator overload for unary negation of SquaredBlkDiagMatrix instance.
        """

        return self.neg()

    def abs(self):
        """
        Compute the elementwise absolute value of SquaredBlkDiagMatrix instance.

        :return: A SquaredBlkDiagMatrix like self.
        """

        C = SquaredBlkDiagMatrix(self.partition, dtype=self.dtype)
        C.data = xp.abs(self.data)
        return C

    def __abs__(self):
        """
        Operator overload for absolute value of SquaredBlkDiagMatrix instance.
        """

        return self.abs()

    def pow(self, val, inplace=False):
        """
        Compute the elementwise power of SquaredBlkDiagMatrix instance.
        :param inplace: Boolean, when set to True change values in place,
        otherwise return a new instance (default).
        :return: A SquaredBlkDiagMatrix like self.
        """

        if inplace:
            self.data = xp.power(self.data,val)
            C = self
        else:
            C = SquaredBlkDiagMatrix(self.partition, dtype=self.dtype)
            C.data = xp.power(self.data, val)
        C.must_update = True
        return C

    def __pow__(self, val):
        """
        Operator overload for inplace pow of SquaredBlkDiagMatrix instance.
        """

        return self.pow(val)

    def __ipow__(self, val):
        """
        Compute the in-place elementwise power of SquaredBlkDiagMatrix instance.

        :return: self raised to power, elementwise.
        """

        return self.pow(val, inplace=True)

    def norm(self):
        """
        Compute the norm of a SquaredBlkDiagMatrix instance.

        :param inplace: Boolean, when set to True change values in place,
        otherwise return a new instance (default).
        :return: The norm of the SquaredBlkDiagMatrix instance.
        """
        #n = 0
        #for blk in self._data:
        #    n = max(n,xp.linalg.norm(blk,ord=2))
        #return n

        return xp.max(xp.array([norm(blk, ord=2) for blk in self.data]))

    def transpose(self):
        """
        Get the transpose matrix of a SquaredBlkDiagMatrix instance.

        :return: The corresponding transpose form as a SquaredBlkDiagMatrix.
        """

        T = SquaredBlkDiagMatrix(self.partition, dtype=self.dtype)
        T.data = xp.transpose(self.data,(0,2,1))
        T.must_update = True

        return T

    @property
    def T(self):
        """
        Syntactic sugar for self.transpose().
        """

        return self.transpose()

    def dense(self):
        """
        Convert list representation of SquaredBlkDiagMatrix instance into full matrix.

        :param blk_diag: The SquaredBlkDiagMatrix instance.
        :return: The SquaredBlkDiagMatrix instance including the zero elements of
        non-diagonal blocks.
        """

        return block_diag(self.data)

    @property
    def partition(self):
        """
        Return the partitions (block sizes) of this SquaredBlkDiagMatrix

        :return: The matrix block partition in the form of a
        K-element list storing all shapes of K diagonal matrix blocks,
        where `partition[i]` corresponds to the shape (number of rows and
        columns) of the `i` diagonal matrix block.
        """

        if self._cached_blk_sizes is None:
            blk_sizes = xp.empty((self.nblocks, 2), dtype=xp.int)
            for i, blk in enumerate(self.data):
                blk_sizes[i] = xp.array(blk.shape)
            self._cached_blk_sizes = blk_sizes
        return self._cached_blk_sizes

    def solve(self, Y):
        """
        Solve a linear system involving a block diagonal matrix.

        :param Y: The right-hand side in the linear system.  May be a matrix
        consisting of coefficient vectors, in which case each column is
        solved for separately.

        :return: The result of solving the linear system formed by the matrix.
        """

        rows = self.partition[:, 0]
        if sum(rows) != xp.size(Y, 0):
            raise RuntimeError('Sizes of `self` and `Y` are not compatible.')

        vector = False
        if Y.ndim == 1:
            Y = Y[:, xp.newaxis]
            vector = True

        cols = xp.array([xp.size(Y, 1)])
        cellarray = Cell2D(rows, cols, dtype=Y.dtype)
        Y = cellarray.mat2cell(Y, rows, cols)
        X = []
        for i in range(0, self.nblocks):
            X.append(solve(self[i], Y[i]))
        X = xp.concatenate(X, axis=0)

        if vector:
            X = X[:, 0]

        return X

    def apply(self, X):
        """
        Define the apply option of a block diagonal matrix with a matrix of
        coefficient vectors.

        :param X: The coefficient matrix with each column is a coefficient
        vector.

        :return: A matrix with new coefficient vectors.
        """

        cols = self.partition[:, 1]

        if xp.sum(cols) != xp.size(X, 0):
            raise RuntimeError(
                'Sizes of matrix `self` and `X` are not compatible.')

        vector = False
        if X.ndim == 1:
            X = X[:, xp.newaxis]
            vector = True

        rows = xp.array([xp.size(X, 1), ])
        cellarray = Cell2D(cols, rows, dtype=X.dtype)
        x_cell = cellarray.mat2cell(X, cols, rows)
        Y = []
        for i in range(0, self.nblocks):
            mat = self[i] @ x_cell[i]
            Y.append(mat)
        Y = xp.concatenate(Y, axis=0)

        if vector:
            Y = Y[:, 0]

        return Y

    @staticmethod
    def __check_square(shp):
        """
        Check if supplied shape tuple is square.

        :param shp:  Shape to test, expressed as a 2-tuple.
        """

        if shp[0] != shp[1]:
            raise NotImplementedError("Currently SquaredBlkDiagMatrix only supports"
                                      " square blocks.  Received {}".format(
                                          shp))

        return True

    @staticmethod
    def empty(nblocks, dtype=xp.float64):
        """
        Instantiate an empty SquaredBlkDiagMatrix with `nblocks`, where each
        data block is initially None with size (0,0).

        This is used for incrementally building up a SquaredBlkDiagMatrix, by
        using nblocks=0 in conjunction with `append` method or
        situations where blocks are immediately assigned in a loop,
        such as in a `copy`.

        :param nblocks: Number of diagonal matrix blocks.
        :return SquaredBlkDiagMatrix instance where each block is None.
        """

        # Empty partition has block dims of zero until they are assigned
        partition = [(0, 0)] * nblocks

        return SquaredBlkDiagMatrix(partition, dtype=dtype)

    @staticmethod
    def zeros(blk_partition, dtype=xp.float64):
        """
        Build a SquaredBlkDiagMatrix zeros matrix.

        :param blk_partition: The matrix block partition in the form of a
        K-element list storing all shapes of K diagonal matrix blocks,
        where `blk_partition[i]` corresponds to the shape (number of rows and
        columns) of the `i` diagonal matrix block.
        :param dtype: The data type to set precision of diagonal matrix block.
        :return: A SquaredBlkDiagMatrix instance consisting of `K` zero blocks.
        """

        A = SquaredBlkDiagMatrix(blk_partition, dtype=dtype)
        A.data = xp.zeros((len(blk_partition),A.max_blk_size,A.max_blk_size), dtype=dtype)
        A.must_update = True

        return A

    @staticmethod
    def ones(blk_partition, dtype=xp.float64):
        """
        Build a SquaredBlkDiagMatrix ones matrix.

        :param blk_partition: The matrix block partition in the form of a
        K-element list storing all shapes of K diagonal matrix blocks,
        where `blk_partition[i]` corresponds to the shape (number of rows and
        columns) of the `i` diagonal matrix block.
        :param dtype: The data type to set precision of diagonal matrix block.
        :return: A SquaredBlkDiagMatrix instance consisting of `K` ones blocks.
        """

        blk_diag = []
        for blk_sz in blk_partition:
            blk_diag.append(xp.ones(blk_sz, dtype=dtype))
        A = SquaredBlkDiagMatrix.from_list(blk_diag, dtype=dtype)

        return A

    @staticmethod
    def eye(blk_partition, dtype=xp.float64):
        """
        Build a SquaredBlkDiagMatrix eye (identity) matrix

        :param blk_partition: The matrix block partition in the form of
        a K-element list storing all shapes of K diagonal matrix blocks,
        where `blk_partition[i]` corresponds to the shape (number of rows
        and columns) of the `i` diagonal matrix block.
        :param dtype: The data type of the diagonal matrix blocks.
        :return: A SquaredBlkDiagMatrix instance consisting of `K` eye (identity)
        blocks.
        """

        blk_diag = []
        for i, blk_sz in enumerate(blk_partition):
            rows, cols = blk_sz
            blk_diag.append(xp.eye(N=int(rows), M=int(cols), dtype=dtype))
        A = SquaredBlkDiagMatrix.from_list(blk_diag, dtype)

        return A

    @staticmethod
    def eye_like(A, dtype=None):
        """
        Build a SquaredBlkDiagMatrix eye (identity) matrix with the partition
        structure of SquaredBlkDiagMatrix A.  Defaults to dtype of A.

        :param A: SquaredBlkDiagMatrix instance.
        :param dtype: Optional, data type of the new diagonal matrix blocks.
        :return: SquaredBlkDiagMatrix instance consisting of `K` eye (identity)
        blocks.
        """

        if dtype is None:
            dtype = A.dtype

        return SquaredBlkDiagMatrix.eye(A.partition, dtype=dtype)

    @staticmethod
    def zeros_like(A, dtype=None):
        """
        Build a SquaredBlkDiagMatrix zeros matrix with the partition
        structure of SquaredBlkDiagMatrix A.  Defaults to dtype of A.

        :param A: SquaredBlkDiagMatrix instance.
        :param dtype: Optional, data type of the new diagonal matrix blocks.
        :return: SquaredBlkDiagMatrix instance consisting of `K` zeros blocks.
        """

        if dtype is None:
            dtype = A.dtype

        B = SquaredBlkDiagMatrix.zeros(A.partition, dtype=dtype)
        return B

    @staticmethod
    def from_list(blk_diag, dtype=xp.float64):
        """
        Convert full from python list representation into SquaredBlkDiagMatrix.

        This is to facilitate integration with code that may not be
        using the SquaredBlkDiagMatrix class yet.

        :param blk_diag; The blk_diag representation in the form of a
        K-element list storing all shapes of K diagonal matrix blocks,
        where `blk_partition[i]` corresponds to the shape (number of rows
        and columns) of the `i` diagonal matrix block.

        :return: The SquaredBlkDiagMatrix instance.
        """

        # get the partition (just sizes)
        blk_partition = [None] * len(blk_diag)
        for i, blk in enumerate(blk_diag):
            blk_partition[i] = blk.shape
        max_blk_size = 0
        for b in blk_partition:
            max_blk_size = max(max_blk_size, b[0])
            max_blk_size = max(max_blk_size, b[1])

        A = SquaredBlkDiagMatrix(blk_partition,dtype=dtype)
        new_blk_diag = []
        for i, blk in enumerate(blk_diag):
            newRowShape = max_blk_size-blk.shape[0]
            newColShape = max_blk_size-blk.shape[1]
            newShape = ((0,newRowShape),(0,newColShape))
            new_blk_diag.append(xp.pad(blk,newShape,'constant',constant_values=(0,0)))
            A._data[i] = blk

        # instantiate an empty SquaredBlkDiagMatrix with that structure
        A.data = xp.array(new_blk_diag)

        return A


def filter_to_fb_mat(h_fun, fbasis):
    """
    Convert a nonradial function in k space into a basis representation.

    :param h_fun: The function form in k space.
    :param fbasis: The basis object for expanding.

    :return: a SquaredBlkDiagMatrix instance representation using the
    `fbasis` expansion.
    """

    if not isinstance(fbasis, FFBBasis2D):
        raise NotImplementedError('Currently only fast FB method is supported')
    # Set same dimensions as basis object
    n_k = int(xp.ceil(4 * fbasis.rcut * fbasis.kcut))
    n_theta = xp.ceil(16 * fbasis.kcut * fbasis.rcut)
    n_theta = int((n_theta + xp.mod(n_theta, 2)) / 2)

    # get 2D grid in polar coordinate
    k_vals, wts = lgwt(n_k, 0, 0.5)
    k, theta = xp.meshgrid(
        k_vals, xp.arange(n_theta) * 2 * xp.pi / (2 * n_theta), indexing='ij')

    # Get function values in polar 2D grid and average out angle contribution
    omegax = k * xp.cos(theta)
    omegay = k * xp.sin(theta)
    omega = 2 * xp.pi * xp.vstack((omegax.flatten('C'), omegay.flatten('C')))
    h_vals2d = h_fun(omega).reshape(n_k, n_theta)
    h_vals = xp.sum(h_vals2d, axis=1)/n_theta

    # Represent 1D function values in fbasis
    h_fb = SquaredBlkDiagMatrix.empty(2 * fbasis.ell_max + 1, dtype=fbasis.dtype)
    ind = 0
    for ell in range(0, fbasis.ell_max+1):
        k_max = fbasis.k_max[ell]
        rmat = 2*k_vals.reshape(n_k, 1)*fbasis.r0[0:k_max, ell].T
        fb_vals = xp.zeros_like(rmat)
        for ik in range(0, k_max):
            fb_vals[:, ik] = jv(ell, rmat[:, ik])
        fb_nrms = 1/xp.sqrt(2)*abs(jv(ell+1, fbasis.r0[0:k_max, ell].T))/2
        fb_vals = fb_vals/fb_nrms
        h_fb_vals = fb_vals*h_vals.reshape(n_k, 1)
        h_fb_ell = fb_vals.T @ (
            h_fb_vals * k_vals.reshape(n_k, 1) * wts.reshape(n_k, 1))
        h_fb[ind] = h_fb_ell
        ind += 1
        if ell > 0:
            h_fb[ind] = h_fb[ind-1]
            ind += 1

    return h_fb
