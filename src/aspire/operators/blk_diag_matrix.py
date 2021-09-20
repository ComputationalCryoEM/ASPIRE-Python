"""
Define a BlkDiagMatrix module which implements operations for
block diagonal matrices as used by ASPIRE.
"""

import numpy as np
from numpy.linalg import norm, solve
from scipy.linalg import block_diag

from aspire.utils import make_psd
from aspire.utils.cell import Cell2D


class BlkDiagMatrix:
    """
    Define a BlkDiagMatrix class which implements operations for
    block diagonal matrices as used by ASPIRE.

    Currently BlkDiagMatrix is implemented only for square blocks.
    While in the future this can be extended, at this time assigning
    a non square array will raise NotImplementedError.
    """

    # Developers' Note:
    # All instances of this class should have priority over ndarray ops
    #   because we implement them here ourselves.
    # This is a more np current implementation of __array_priority__
    #   operator precedence schedule.
    # Mainly this effects rmul, radd, rsub eg:
    #   blk_y = scalar_a + ( scalar_b * blk_x)
    __array_ufunc__ = None

    def __init__(self, partition, dtype=np.float32):
        """
        Instantiate a BlkDiagMatrix.

        :param partition: The matrix block partition
         in the form of a `nblock`-element list storing all shapes of
         diagonal matrix blocks, where `partition[i]` corresponds to
         the shape (number of rows and columns) of the `i` matrix block.
        :param dtype: Datatype for blocks, defaults to np.float32.
        :return: BlkDiagMatrix instance.
        """

        self.nblocks = len(partition)
        self.dtype = np.dtype(dtype)
        self.data = [None] * self.nblocks
        self._cached_blk_sizes = np.array(partition)
        if len(partition):
            assert self._cached_blk_sizes.shape[1] == 2

    def reset_cache(self):
        """
        Resets this objects internal cache. This should trigger the cache
        to be recalculated on the next request (ie lazily).
        """

        self._cached_blk_sizes = None

    def append(self, blk):
        """
        Append `blk` to `self`. Used to incrementally build up a BlkDiagMatrix
        instance where the number of blocks and/or shapes are derived
        incrementally.

        :param blk: Block to append (ndarray).
        """

        self.data.append(blk)
        self.nblocks += 1
        self.reset_cache()

    def __repr__(self):
        """
        String represention describing instance.
        """
        return "BlkDiagMatrix({}, {})".format(repr(self.nblocks), repr(self.dtype))

    def copy(self):
        """
        Returns new BlkDiagMatrix which is a copy of `self`.

        :return BlkDiagMatrix like self
        """

        C = BlkDiagMatrix(self.partition, dtype=self.dtype)

        for i in range(self.nblocks):
            C[i] = self[i].copy()

        return C

    # Manually overload list methods,
    #   This is just for syntax which allows us to reference self[i] etc
    #     instead of writing self.data all the time. You may use either.
    def __getitem__(self, key):
        """
        Convenience wrapper, getter on self.data.
        """

        return self.data[key]

    def __setitem__(self, key, value):
        """
        Convenience wrapper, setter on self.data.
        """

        self.data[key] = value
        self.reset_cache()

    def __len__(self):
        """
        Convenience function for getting nblocks.
        """

        return self.nblocks

    def _is_scalar_type(self, x):
        """
        Internal helper function checking scalar-ness for elementwise ops.

        Essentially we are checking for a single numeric object, as opposed to
        something like an `ndarray` or `BlkDiagMatrix`. We do this by
        checking `numpy.isscalar(x)`.

        In the future this check may require extension to include ASPIRE or
        other third party types beyond what is provided by numpy, so we
        implement it now as a class method.

        :param x: Value to check

        :return: bool.
        """

        return np.isscalar(x)

    def __check_size_compatible_add(self, other):
        """
        Sanity check two BlkDiagMatrix instances are compatible in size
        for addition operators. (Same size)

        :param other: The BlkDiagMatrix to compare with self.
        """

        if np.any(self.partition != other.partition):
            # be helpful and find the first one as an example
            for _i, (a, b) in enumerate(zip(self.partition, other.partition)):
                if any(a != b):
                    break
            raise RuntimeError(
                "Block i={} of BlkDiagMatrix instances are "
                "not same shape {} {}".format(_i, a, b)
            )

    def __check_size_compatible_mul(self, other):
        """
        Sanity check two BlkDiagMatrix instances are compatible in size
        for multiplication operators. (m n) @ (n k).

        :param other: The BlkDiagMatrix to compare with self.
        """

        for _i, a in enumerate(self.partition):
            b = other.partition[_i]
            if a[1] != b[0]:
                raise RuntimeError(
                    "Block i={} of BlkDiagMatrix instances are "
                    "not compatible. {} {}".format(_i, a, b)
                )

    def __check_dtype_compatible(self, other):
        """
        Sanity check two BlkDiagMatrix instances are compatible in dtype.

        :param other: The BlkDiagMatrix to compare with self.
        """

        if self.dtype != other.dtype:
            raise RuntimeError(
                "BlkDiagMatrix received different types,"
                "self: {} and other: {}.  Please validate and cast"
                " as appropriate.".format(self.dtype, other.dtype)
            )

    def __check_compatible(self, other, size_compat="add"):
        """
        Sanity check two BlkDiagMatrix instances are compatible in size.

        :param other: The BlkDiagMatrix to compare with self.
        """

        if not isinstance(other, BlkDiagMatrix):
            raise NotImplementedError(
                "Currently BlkDiagMatrix only interfaces "
                "with its own instances, got {}".format(repr(other))
            )

        if len(self) != len(other):
            raise RuntimeError(
                "Number of blocks {} {} are not equal.".format(len(self), len(other))
            )

        if size_compat == "add":
            self.__check_size_compatible_add(other)
        elif size_compat == "mul":
            self.__check_size_compatible_mul(other)
        else:
            raise RuntimeError("Unknown compatibility type {}".format(size_compat))

        self.__check_dtype_compatible(other)

    @property
    def is_square(self):
        """
        Check if all blocks are square.

        :return: boolean
        """
        return all([shp[0] == shp[1] for shp in self.partition])

    @property
    def isfinite(self):
        """
        Check if all blocks in diag matrix are finite.

        Calls numpy.isfinite for every entry in self.  This has the effect of
        checking values are not += `inf` or `nan`s.

        :return: Bool.
        """

        for blk in self:
            if not np.all(np.isfinite(blk)):
                return False
        else:
            return True

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
            for i in range(self.nblocks):
                self[i] += other[i]

            C = self
        else:
            C = BlkDiagMatrix(self.partition, dtype=self.dtype)

            for i in range(self.nblocks):
                C[i] = self[i] + other[i]

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

        for i in range(self.nblocks):
            C[i] += scalar

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
            for i in range(self.nblocks):
                self[i] -= other[i]

            C = self
        else:
            C = BlkDiagMatrix(self.partition, dtype=self.dtype)

            for i in range(self.nblocks):
                C[i] = self[i] - other[i]

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

        for i in range(self.nblocks):
            C[i] -= scalar

        return C

    def matmul(self, other, inplace=False):
        """
        Compute the matrix multiplication of two BlkDiagMatrix instances.

        :param other: The rhs BlkDiagMatrix instance.
        :param inplace: Boolean, when set to True change values in place,
        otherwise return a new instance (default).
        :return: A BlkDiagMatrix of self @ other.
        """

        if not isinstance(other, BlkDiagMatrix):
            if inplace:
                raise RuntimeError(
                    "`inplace` method not supported when "
                    "mixing `BlkDiagMatrix` and `Numpy`."
                )
            return self.apply(other)

        self.__check_compatible(other, size_compat="mul")

        if inplace:
            for i in range(self.nblocks):
                self[i] @= other[i]
            C = self
        else:
            C = BlkDiagMatrix(self.partition, dtype=self.dtype)

            for i in range(self.nblocks):
                C[i] = self[i] @ other[i]

        return C

    def __matmul__(self, other):
        """
        Operator overload for matrix multiply of BlkDiagMatrix instances.
        """

        return self.matmul(other)

    def __rmatmul__(self, lhs):
        """
        Compute the right matrix multiplication with a BlkDiagMatrix instance,
        and a numpy array, lhs @ self.

        :param other: The lhs Numpy instance.
        :return: Returns numpy array representing `other @ self`.
        """

        # Note, we should only hit this method when mixing BlkDiagMatrix with numpy.
        #   This is because if both a and b are BlkDiagMatrix,
        #   then a@b would be handled first by a.__matmul__(b), never reaching here.
        if not isinstance(lhs, np.ndarray):
            raise RuntimeError(
                "__rmatmul__ only defined for np.ndarray @ BlkDiagMatrix."
            )

        return self.rapply(lhs)

    def __imatmul__(self, other):
        """
        Operator overload for in-place matrix multiply of BlkDiagMatrix
         instances.
        """

        return self.matmul(other, inplace=True)

    def mul(self, val, inplace=False):
        """
        Compute the numeric multiplication of a BlkDiagMatrix instance and a
        scalar.

        :param other: The rhs BlkDiagMatrix instance.
        :param inplace: Boolean, when set to True change values in place,
        otherwise return a new instance (default).
        :return: A BlkDiagMatrix of self * other.
        """

        if isinstance(val, BlkDiagMatrix):
            raise RuntimeError(
                "Attempt numeric multiplication (*,mul) of two "
                "BlkDiagMatrix instances, try (matmul,@)."
            )

        elif not self._is_scalar_type(val):
            raise RuntimeError(
                "Attempt numeric multiplication (*,mul) of a "
                "BlkDiagMatrix and {}.".format(type(val))
            )

        if inplace:
            for i in range(self.nblocks):
                self[i] *= val

            C = self
        else:
            C = BlkDiagMatrix(self.partition, dtype=self.dtype)

            for i in range(self.nblocks):
                C[i] = self[i] * val

        return C

    def __mul__(self, val):
        """
        Operator overload for BlkDiagMatrix scalar multiply.
        """

        return self.mul(val)

    def __imul__(self, val):
        """
        Operator overload for in-place BlkDiagMatrix scalar multiply.
        """

        return self.mul(val, inplace=True)

    def __rmul__(self, other):
        """
        Convenience function, elementwise rmul commutes to mul.
        """

        return self.mul(other)

    def neg(self):
        """
        Compute the unary negation of BlkDiagMatrix instance.

        :return: A BlkDiagMatrix like self.
        """

        C = BlkDiagMatrix(self.partition, dtype=self.dtype)

        for i in range(self.nblocks):
            C[i] = -self[i]

        return C

    def __neg__(self):
        """
        Operator overload for unary negation of BlkDiagMatrix instance.
        """

        return self.neg()

    def abs(self):
        """
        Compute the elementwise absolute value of BlkDiagMatrix instance.

        :return: A BlkDiagMatrix like self.
        """

        C = BlkDiagMatrix(self.partition, dtype=self.dtype)

        for i in range(self.nblocks):
            C[i] = np.abs(self[i])

        return C

    def __abs__(self):
        """
        Operator overload for absolute value of BlkDiagMatrix instance.
        """

        return self.abs()

    def pow(self, val, inplace=False):
        """
        Compute the elementwise power of BlkDiagMatrix instance.
        :param inplace: Boolean, when set to True change values in place,
        otherwise return a new instance (default).
        :return: A BlkDiagMatrix like self.
        """

        if inplace:
            for i in range(self.nblocks):
                self[i] **= val
            C = self
        else:
            C = BlkDiagMatrix(self.partition, dtype=self.dtype)
            for i in range(self.nblocks):
                C[i] = np.power(self[i], val)
        return C

    def __pow__(self, val):
        """
        Operator overload for inplace pow of BlkDiagMatrix instance.
        """

        return self.pow(val)

    def __ipow__(self, val):
        """
        Compute the in-place elementwise power of BlkDiagMatrix instance.

        :return: self raised to power, elementwise.
        """

        return self.pow(val, inplace=True)

    def norm(self):
        """
        Compute the norm of a BlkDiagMatrix instance.

        :param inplace: Boolean, when set to True change values in place,
        otherwise return a new instance (default).
        :return: The norm of the BlkDiagMatrix instance.
        """

        return np.max([norm(blk, ord=2) for blk in self])

    def transpose(self):
        """
        Get the transpose matrix of a BlkDiagMatrix instance.

        :return: The corresponding transpose form as a BlkDiagMatrix.
        """

        T = BlkDiagMatrix(self.partition, dtype=self.dtype)

        for i in range(self.nblocks):
            T[i] = self[i].T

        return T

    @property
    def T(self):
        """
        Syntactic sugar for self.transpose().
        """

        return self.transpose()

    def dense(self):
        """
        Convert list representation of BlkDiagMatrix instance into full matrix.

        :param blk_diag: The BlkDiagMatrix instance.
        :return: The BlkDiagMatrix instance including the zero elements of
        non-diagonal blocks.
        """

        return block_diag(*self.data)

    @property
    def partition(self):
        """
        Return the partitions (block sizes) of this BlkDiagMatrix

        :return: The matrix block partition in the form of a
        K-element list storing all shapes of K diagonal matrix blocks,
        where `partition[i]` corresponds to the shape (number of rows and
        columns) of the `i` diagonal matrix block.
        """

        if self._cached_blk_sizes is None:
            blk_sizes = np.empty((self.nblocks, 2), dtype=int)
            for i, blk in enumerate(self.data):
                blk_sizes[i] = np.shape(blk)
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
        if sum(rows) != Y.shape[0]:
            raise RuntimeError("Sizes of `self` and `Y` are not compatible.")

        # Use `np.linalg.solve` for square matrices/blocks.
        #   If user requires solving non square, we'll need to extend for
        #   lstsq or qr,triangle solvers.
        if not self.is_square:
            raise NotImplementedError(
                "BlkDiagMatrix.solve is only defined for square arrays. "
                "If you require solving non square BlkDiagMatrix please "
                "report to developers."
            )

        vector = False
        if np.ndim(Y) == 1:
            Y = Y[:, np.newaxis]
            vector = True

        cols = np.array([np.size(Y, 1)])
        cellarray = Cell2D(rows, cols, dtype=Y.dtype)
        Y = cellarray.mat2cell(Y, rows, cols)
        X = []
        for i in range(0, self.nblocks):
            X.append(solve(self[i], Y[i]))
        X = np.concatenate(X, axis=0)

        if vector:
            X = X[:, 0]

        return X

    def apply(self, X):
        """
        Define the apply option of a block diagonal matrix with a matrix of
        coefficient vectors.

        :param X: Coefficient matrix, each column is a coefficient vector.
        :return: A matrix with new coefficient vectors.
        """

        cols = self.partition[:, 1]

        if np.sum(cols) != np.size(X, 0):
            raise RuntimeError("Sizes of matrix `self` and `X` are not compatible.")

        vector = False
        if np.ndim(X) == 1:
            X = X[:, np.newaxis]
            vector = True

        rows = np.array(
            [
                np.size(X, 1),
            ]
        )
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

    def rapply(self, X):
        """
        Right apply.  Given a matrix of coefficient vectors,
        applies the block diagonal matrix on the right hand side.
        Example, X @ self.

        This is the right hand side equivalent to `apply`.

        :param X: Coefficient matrix, each column is a coefficient vector.

        :return: A matrix with new coefficient vectors.
        """

        # For now do the transposes a @ b = (b.T @ a.T).T.
        #  Note there is an optimization opportunity here,
        #  but the current application of this method is only called once
        #  per FSPCA/RIR classification.
        return self.T.apply(X.T).T

    def eigvals(self):
        """
        Compute the eigenvalues of a BlkDiagMatrix.
        :return: Array of eigvals, with length equal to the fully expanded matrix diagonal.

        """
        return np.concatenate([np.linalg.eigvals(blk).flatten() for blk in self])

    def check_psd(self):
        """
        Check the positive semidefinite property of all blocks

        :return: True if all blocks have non-negative eigenvalues.
        """
        return np.alltrue(self.eigvals() > 0.0)

    def make_psd(self):
        """
        Convert all blocks to positive semidefinite

        :return: The BlkDiagMatrix instance with all blocks
            positive semidefinite
        """

        C = BlkDiagMatrix(self.partition, dtype=self.dtype)

        for i in range(self.nblocks):
            C[i] = make_psd(self[i])

        return C

    @staticmethod
    def empty(nblocks, dtype=np.float32):
        """
        Instantiate an empty BlkDiagMatrix with `nblocks`, where each
        data block is initially None with size (0,0).

        This is used for incrementally building up a BlkDiagMatrix, by
        using nblocks=0 in conjunction with `append` method or
        situations where blocks are immediately assigned in a loop,
        such as in a `copy`.

        :param nblocks: Number of diagonal matrix blocks.
        :return BlkDiagMatrix instance where each block is None.
        """

        # Empty partition has block dims of zero until they are assigned
        partition = [(0, 0)] * nblocks

        return BlkDiagMatrix(partition, dtype=dtype)

    @staticmethod
    def zeros(blk_partition, dtype=np.float32):
        """
        Build a BlkDiagMatrix zeros matrix.

        :param blk_partition: The matrix block partition in the form of a
        K-element list storing all shapes of K diagonal matrix blocks,
        where `blk_partition[i]` corresponds to the shape (number of rows and
        columns) of the `i` diagonal matrix block.
        :param dtype: The data type to set precision of diagonal matrix block.
        :return: A BlkDiagMatrix instance consisting of `K` zero blocks.
        """

        A = BlkDiagMatrix(blk_partition, dtype=dtype)

        for i, blk_sz in enumerate(blk_partition):
            A[i] = np.zeros(blk_sz, dtype=dtype)

        return A

    @staticmethod
    def ones(blk_partition, dtype=np.float32):
        """
        Build a BlkDiagMatrix ones matrix.

        :param blk_partition: The matrix block partition in the form of a
        K-element list storing all shapes of K diagonal matrix blocks,
        where `blk_partition[i]` corresponds to the shape (number of rows and
        columns) of the `i` diagonal matrix block.
        :param dtype: The data type to set precision of diagonal matrix block.
        :return: A BlkDiagMatrix instance consisting of `K` ones blocks.
        """

        A = BlkDiagMatrix(blk_partition, dtype=dtype)

        for i, blk_sz in enumerate(blk_partition):
            A[i] = np.ones(blk_sz, dtype=dtype)

        return A

    @staticmethod
    def eye(blk_partition, dtype=np.float32):
        """
        Build a BlkDiagMatrix eye (identity) matrix

        :param blk_partition: The matrix block partition in the form of
        a K-element list storing all shapes of K diagonal matrix blocks,
        where `blk_partition[i]` corresponds to the shape (number of rows
        and columns) of the `i` diagonal matrix block.
        :param dtype: The data type of the diagonal matrix blocks.
        :return: A BlkDiagMatrix instance consisting of `K` eye (identity)
        blocks.
        """

        A = BlkDiagMatrix(blk_partition, dtype=dtype)

        for i, blk_sz in enumerate(blk_partition):
            rows, cols = blk_sz
            A[i] = np.eye(N=rows, M=cols, dtype=dtype)

        return A

    @staticmethod
    def eye_like(A, dtype=None):
        """
        Build a BlkDiagMatrix eye (identity) matrix with the partition
        structure of BlkDiagMatrix A.  Defaults to dtype of A.

        :param A: BlkDiagMatrix instance.
        :param dtype: Optional, data type of the new diagonal matrix blocks.
        :return: BlkDiagMatrix instance consisting of `K` eye (identity)
        blocks.
        """

        if dtype is None:
            dtype = A.dtype

        return BlkDiagMatrix.eye(A.partition, dtype=dtype)

    @staticmethod
    def zeros_like(A, dtype=None):
        """
        Build a BlkDiagMatrix zeros matrix with the partition
        structure of BlkDiagMatrix A.  Defaults to dtype of A.

        :param A: BlkDiagMatrix instance.
        :param dtype: Optional, data type of the new diagonal matrix blocks.
        :return: BlkDiagMatrix instance consisting of `K` zeros blocks.
        """

        if dtype is None:
            dtype = A.dtype

        return BlkDiagMatrix.zeros(A.partition, dtype=dtype)

    @staticmethod
    def from_list(blk_diag, dtype=np.float32):
        """
        Convert full from python list representation into BlkDiagMatrix.

        This is to facilitate integration with code that may not be
        using the BlkDiagMatrix class yet.

        :param blk_diag; The blk_diag representation in the form of a
        K-element list storing all shapes of K diagonal matrix blocks,
        where `blk_partition[i]` corresponds to the shape (number of rows
        and columns) of the `i` diagonal matrix block.

        :return: The BlkDiagMatrix instance.
        """

        # get the partition (just sizes)
        blk_partition = [None] * len(blk_diag)
        for i, mat in enumerate(blk_diag):
            blk_partition[i] = np.shape(mat)

        # instantiate an empty BlkDiagMatrix with that structure
        A = BlkDiagMatrix(blk_partition, dtype=dtype)

        # set the data
        for i in range(A.nblocks):
            A.data[i] = np.array(blk_diag[i], dtype=dtype)

        return A
