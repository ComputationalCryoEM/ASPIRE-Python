"""
Implements operations for diagonal matrices as used by ASPIRE.
"""

import numpy as np

from .blk_diag_matrix import BlkDiagMatrix


class DiagMatrix:
    """
    Implements operations for diagonal matrices as used by ASPIRE.

    Currently `DiagMatrix` is implemented as the equivalent of square
    matrices.  In the future it can be extended to support
    applications with rectangular shapes.
    """

    def __init__(self, data, dtype=None):
        """
        Instantiate a `DiagMatrix` with Numpy `data` shaped (...., self.count),
        where `self.count` is the length of one diagonal vector.

        Slower axes are taken to be stack axes.
        Inputs of zero or one dimension are taken to be a stack of one.

        :param data: Diagonal matrix entries.
        :param dtype: Datatype. Default of `None` will attempt
        passthrough of `data.dtype`.  When explicitly provided, will
        attempt casting `data` as needed.
        :return: `DiagMatrix` instance.
        """

        # Assign the datatype.
        if dtype is None:
            dtype = data.dtype
        self.dtype = np.dtype(dtype)

        self._data = data.astype(self.dtype, copy=False)
        self.count = self._data.shape[-1]
        self.stack_shape = self._data.shape[:-1]
        self.shape = self._data.shape

        # Numpy interop
        # https://numpy.org/devdocs/user/basics.interoperability.html#the-array-interface-protocol
        self.__array_interface__ = self.asnumpy().__array_interface__
        self.__array__ = self.asnumpy()

    def stack_reshape(self, *args):
        """
        Reshape the stack axis.

        :*args: Integer(s) or tuple describing the intended shape.

        :returns: DiagMatrix instance.
        """

        # If we're passed a tuple, use that
        if len(args) == 1 and isinstance(args[0], tuple):
            shape = args[0]
        else:
            # Otherwise use the variadic args
            shape = args

        # Sanity check the size
        if shape != (-1,) and np.prod(shape) != self.n_stack:
            raise ValueError(
                f"Number of images {self.n_stack} cannot be reshaped to {shape}."
            )

        return DiagMatrix(self._data.reshape(*shape, self._data.shape[-1]))

    def asnumpy(self):
        view = self._data.view()
        view.flags.writeable = False
        return view

    def __repr__(self):
        """
        String represention describing instance.
        """
        return "DiagMatrix({}, {})".format(repr(self._data), repr(self.dtype))

    def copy(self):
        """
        Returns new DiagMatrix which is a copy of `self`.

        :return DiagMatrix like self
        """

        return DiagMatrix(self._data.copy())

    def __getitem__(self, key):
        """
        Convenience wrapper, getter on self._data.
        """

        return self._data[key]

    def __len__(self):
        """
        Convenience function for getting `n`.
        """

        return self.count

    def _is_scalar_type(self, x):
        """
        Internal helper function checking scalar-ness for elementwise ops.

        Essentially we are checking for a single numeric object, as opposed to
        something like an `ndarray` or `DiagMatrix`. We do this by
        checking `numpy.isscalar(x)`.

        In the future this check may require extension to include ASPIRE or
        other third party types beyond what is provided by numpy, so we
        implement it now as a class method.

        :param x: Value to check

        :return: bool.
        """

        return np.isscalar(x)

    def __check_size_compatible(self, other):
        """
        Sanity check two DiagMatrix instances are compatible in size
        for addition operators. (Same size)

        :param other: The DiagMatrix to compare with self.
        """

        if self.count != other.count:
            raise RuntimeError(
                "DiagMatrix instances are "
                f"not same dimension {self.count} != {other.count}"
            )

    def __check_dtype_compatible(self, other):
        """
        Sanity check two DiagMatrix instances are compatible in dtype.

        :param other: The DiagMatrix to compare with self.
        """

        if self.dtype != other.dtype:
            raise RuntimeError(
                "DiagMatrix received different types,"
                f"self: {self.dtype} and other: {other.dtype}.  Please validate and cast"
                " as appropriate."
            )

    def add(self, other, inplace=False):
        """
        Define elementwise addition of DiagMatrix instances

        :param other: The rhs DiagMatrix instance.
        :param inplace: Boolean, when set to True change values in place,
            otherwise return a new instance (default).
        :return:  DiagMatrix instance with elementwise sum equal
            to self + other.
        """
        return self._data + other._data

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

    def sub(self, other, inplace=False):
        """
        Define the element subtraction of DiagMatrix instance.

        :param other: The rhs DiagMatrix instance.
        :param inplace: Boolean, when set to True change values in place,
            otherwise return a new instance (default).
        :return: A DiagMatrix instance with elementwise subraction equal to
            self - other.
        """
        return self._data - other._data

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

        # Note, the case of DiagMatrix_L - DiagMatrix_R would be
        #   evaluated as L.sub(R), so this is only for other
        #   Object - DiagMatrix situations, namely scalars.

        return -(self - other)

    def matmul(self, other, inplace=False):
        """
        Compute the matrix multiplication of two DiagMatrix instances.

        :param other: The rhs `DiagMatrix`, `BlkDiagMatrix` or 2d dense Numpy array.
        :param inplace: Boolean, when set to True change values in place,
            otherwise return a new instance (default).
        :return: Returns `self` @ `other`, as type of `other`
        """
        if isinstance(other, DiagMatrix):
            res = self.mul(other, inplace=inplace)

        elif isinstance(other, BlkDiagMatrix):
            # inplace infeasable, as the result will not be DiagMatrix.
            if inplace:
                raise RuntimeError(
                    "Inplace infeasable between DiagMatrix and BlkDiagMatrix matrix."
                )
            res = BlkDiagMatrix.zeros_like(other)

            ind = 0
            for b, blk in enumerate(other):
                i = len(blk)

                res[b] = np.diag(self._data[ind : ind + i]) * blk
                ind += i

        elif isinstance(other, np.ndarray):
            # inplace infeasable, as the result will not be DiagMatrix.
            if inplace:
                raise RuntimeError(
                    "Inplace infeasable between DiagMatrix and dense ndarray matrix."
                )

            # For now, lets just interop with 2D `other` arrays,
            # assumed to represent matrices.
            if other.ndim != 2:
                raise ValueError(
                    f"DiagMatrix.matmul of ndarray only supports 2D, received {other.shape}."
                )

            # Then the mathematical `DA` for diag D and matrix A,
            # is expressed in code as `D@A`.
            #
            # The operation `DA` is known as Left Scaling,
            # and means to scale the row A_i by d_i in D.
            # This can be accomplished by broadcasting an elemental multiply.

            res = self._data[..., np.newaxis] * other

        return res

    def __matmul__(self, other):
        """
        Operator overload for matrix multiply of DiagMatrix instances.
        """

        return self.matmul(other)

    def __rmatmul__(self, lhs):
        """
        Compute the right matrix multiplication with a DiagMatrix instance,
        and a numpy array, `lhs` @ `self`.

        :param other: The lhs Numpy instance.
        :return: Returns numpy array representing `other @ self`.
        """

        # Note, we should only hit this method when mixing DiagMatrix with numpy.
        #   This is because if both a and b are DiagMatrix,
        #   then a@b would be handled first by a.__matmul__(b), never reaching here.
        if not isinstance(lhs, np.ndarray):
            raise RuntimeError("__rmatmul__ only defined for np.ndarray @ DiagMatrix.")

        # For now, lets just interop with 2D `other` arrays,
        # assumed to represent matrices.
        if lhs.ndim != 2:
            raise ValueError(
                f"DiagMatrix.matmul of ndarray only supports 2D, received {lhs.shape}."
            )

        # Then the mathematical `AD` for diag D and matrix A,
        # is expressed in code as `A@D`.
        #
        # The operation `AD` is known as Right Scaling,
        # and means to scale the column A_j by d_j in D.
        # This can be accomplished by broadcasting an elemental multiply.
        # In this case A is the variable `lhs`

        return lhs * self._data[np.newaxis]

    def __imatmul__(self, other):
        """
        Operator overload for in-place matrix multiply of DiagMatrix
         instances.
        """

        return self.matmul(other, inplace=True)

    def mul(self, other, inplace=False):
        """
        Compute the elementwise multiplication of a DiagMatrix instance and a
        scalar or another DiagMatrix.

        :param other: The rhs in the multiplication..
        :param inplace: Boolean, when set to True change values in place,
            otherwise return a new instance (default).
        :return: A DiagMatrix of self * other.
        """
        if isinstance(other, DiagMatrix):
            if inplace:
                self._data *= other._data
                res = self
            else:
                res = DiagMatrix(self._data * other._data)
        # elif isinstance(other, np.ndarray):
        #     res = self * DiagMatrix(other)
        elif isinstance(other, BlkDiagMatrix):
            raise NotImplementedError("not yet")
        else:  # scalar
            res = DiagMatrix(self._data * other)

        return res

    def __mul__(self, val):
        """
        Operator overload for DiagMatrix scalar multiply.
        """

        return self.mul(val)

    def __imul__(self, val):
        """
        Operator overload for in-place DiagMatrix scalar multiply.
        """

        return self.mul(val, inplace=True)

    def __rmul__(self, other):
        """
        Convenience function, elementwise rmul commutes to mul.
        """
        if isinstance(other, BlkDiagMatrix):
            raise NotImplementedError("todo")
        else:
            return self.mul(other)

    def neg(self):
        """
        Compute the unary negation of DiagMatrix instance.

        :return: A DiagMatrix like self.
        """
        return DiagMatrix(-self._data)

    def __neg__(self):
        """
        Operator overload for unary negation of DiagMatrix instance.
        """

        return self.neg()

    def abs(self):
        """
        Compute the elementwise absolute value of DiagMatrix instance.

        :return: A DiagMatrix like self.
        """
        pass

    def __abs__(self):
        """
        Operator overload for absolute value of DiagMatrix instance.
        """

        return DiagMatrix(self.abs(self._data))

    def pow(self, val, inplace=False):
        """
        Compute the elementwise power of DiagMatrix instance.

        :param inplace: Boolean, when set to True change values in place,
            otherwise return a new instance (default).
        :return: A DiagMatrix like self.
        """

        if inplace:
            self._data[:] = self._data**val
            res = self
        else:
            res = DiagMatrix(self._data**val)

        return res

    def __pow__(self, val):
        """
        Operator overload for inplace pow of DiagMatrix instance.
        """

        return self.pow(val)

    def __ipow__(self, val):
        """
        Compute the in-place elementwise power of DiagMatrix instance.

        :return: self raised to power, elementwise.
        """

        return self.pow(val, inplace=True)

    @property
    def norm(self):
        """
        Compute the norm of a DiagMatrix instance.

        :return: The norm of the DiagMatrix instance.
        """
        # Elements of a diag matrix are its singular values,
        #   and the norm is equal to the largest singular value.
        return np.abs(self._data).max(axis=-1)

    # Transpose methods are provided for reasons of interoperability
    # in code that also uses with BlkDiagMatrix.
    def transpose(self):
        """
        Get the transpose matrix of a DiagMatrix instance.

        :return: The corresponding transpose form as a DiagMatrix.
        """
        return self.copy()

    @property
    def T(self):
        """
        Syntactic sugar for self.transpose().
        """

        return self.transpose()

    @property
    def dense(self):
        """
        Convert DiagMatrix instance into full matrix.

        :return: The DiagMatrix instance including the zero elements of
        non-diagonal elements.
        """

        if self._data.ndim == 1:
            return np.diag(self._data)

        original_stack_shape = self.stack_shape
        diags = self.stack_reshape(-1)  # Flatten stack

        dense = []
        for d in diags.asnumpy():
            dense.append(np.diag(d))
        # Convert to np.array
        dense = np.concatenate(dense)
        # Reshape to original stack shape.
        return dense.reshape(*original_stack_shape, self.count, self.count)

    def apply(self, X):
        """
        Define the apply option of a diagonal matrix with a matrix of
        coefficient vectors.

        :param X: Coefficient matrix, each column is a coefficient vector.
        :return: A matrix with new coefficient vectors.
        """

        return self * X

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

    # `eigval` method is provided for reasons of interoperability
    # in code that also uses with `BlkDiagMatrix`.
    def eigvals(self):
        """
        Compute the eigenvalues of a `DiagMatrix`.

        :return: Array of eigvals, with length equal to the fully expanded matrix diagonal.
        """
        return self._data.asnumpy()

    @staticmethod
    def empty(n, dtype=np.float32):
        """
        Instantiate an empty DiagMatrix with length `n`.
        This corresponds to the diag(A) where A is (n,n).
        Note, like Numpy, empty values are uninitialized.

        :param n: Length of diagonal.
        :param dtype: Datatype, defaults to np.float32.
        :return: DiagMatrix instance.
        """

        return DiagMatrix(np.empty(n, dtype=dtype))

    @staticmethod
    def zeros(n, dtype=np.float32):
        """
        Instantiate a zero intialized DiagMatrix with length `n`.
        This corresponds to the diag(A) where A is (n,n).

        :param n: Length of diagonal.
        :param dtype: Datatype, defaults to np.float32.
        :return: DiagMatrix instance.
        """

        return DiagMatrix(np.zeros(n, dtype=dtype))

    @staticmethod
    def ones(n, dtype=np.float32):
        """
        Instantiate ones intialized DiagMatrix with length `n`.
        This corresponds to the diag(A) where A is (n,n).

        :param n: Length of diagonal.
        :param dtype: Datatype, defaults to np.float32.
        :return: DiagMatrix instance.
        """

        return DiagMatrix(np.ones(n, dtype=dtype))

    @staticmethod
    def eye(n, dtype=np.float32):
        """
        Build a DiagMatrix eye (identity) matrix.
        This is simply an alias for `ones`.

        :param n: Length of diagonal.
        :param dtype: Datatype, defaults to np.float32.
        :return: DiagMatrix instance.
        """

        return DiagMatrix.ones(n, dtype=dtype)

    def as_blk_diag(self, partition):
        """
        Express `DiagMatrix` as a `BlkDiagMatrix` using `partition`.

        :return: `BlkDiagMatrix`
        """
        B = BlkDiagMatrix(partition, dtype=self.dtype)
        ind = 0
        for b, p in enumerate(partition):
            assert p[0] == p[1]
            j = p[0]
            B[b] = np.diag(self._data[ind : ind + j])
            ind += j

        return B

    def yunpeng(self, partition, weights=None):
        if weights is None:
            weights = np.ones(self.count, dtype=self.dtype)

        B = BlkDiagMatrix(partition, dtype=self.dtype)

        ind = 0
        for block, p in enumerate(partition):
            assert p[0] == p[1]  # squareness
            j = p[0]
            Ai = self._data[ind : ind + j].reshape(-1, 1)
            wi = weights[ind : ind + j]
            B[block] = wi * Ai * Ai.T
            ind += j

        return B

    def solve(self, b):
        """
        Solve a x = b for `x`, given diagonal matrix `b`.

        :param b: `DiagMatrix`, right hand side.
        :return: `DiagMatrix`, solution.
        """

        return b / self[:, None]
