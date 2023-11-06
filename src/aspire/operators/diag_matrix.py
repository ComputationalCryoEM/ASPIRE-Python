"""
Implements operations for diagonal matrices as used by ASPIRE.
"""

import numpy as np

from .blk_diag_matrix import BlkDiagMatrix, is_scalar_type


class DiagMatrix:
    """
    Implements operations for diagonal matrices as used by ASPIRE.

    Currently `DiagMatrix` is implemented as the equivalent of square
    matrices.  In the future it can be extended to support
    applications with rectangular shapes.
    """

    # Developers' Note:
    # All instances of this class should have priority over ndarray ops
    #   because we implement them here ourselves.
    # This is a more np current implementation of __array_priority__
    #   operator precedence schedule.
    # Mainly this effects rmul, eg:
    # np_ary @ diag_mat
    __array_ufunc__ = None

    def __init__(self, data, dtype=None):
        """
        Instantiate a `DiagMatrix` with Numpy `data` shaped (...., self.count),
        where `self.count` is the length of one diagonal vector.

        Slower axes (if present) are taken to be stack axes.

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

        # Assign the `data`
        self._data = data.astype(self.dtype, copy=False)

        # Assign shapes from `data`
        self.count = self._data.shape[-1]
        self.stack_shape = self._data.shape[:-1]
        # Total number of stack elements
        self.size = np.prod(self.stack_shape)

        # Numpy interop
        # https://numpy.org/devdocs/user/basics.interoperability.html#the-array-interface-protocol
        self.__array_interface__ = self.asnumpy().__array_interface__
        self.__array__ = self.asnumpy()

    def stack_reshape(self, *args):
        """
        Reshape the stack axis.

        :*args: Integer(s) or tuple describing the intended shape.

        :returns: `DiagMatrix` instance.
        """

        # If we're passed a tuple, use that
        if len(args) == 1 and isinstance(args[0], tuple):
            shape = args[0]
        else:
            # Otherwise use the variadic args
            shape = args

        # Sanity check the size
        if shape != (-1,) and np.prod(shape) != self.size:
            raise ValueError(
                f"Number of images {self.size} cannot be reshaped to {shape}."
            )

        return DiagMatrix(self._data.reshape(*shape, self._data.shape[-1]))

    def asnumpy(self):
        """
        Return data as Numpy array.

        Note this is a read-only view.

        :return: Numpy array of `self.dtype`.
        """

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
        Returns new `DiagMatrix` which is a copy of `self`.

        :return `DiagMatrix` like self
        """

        return DiagMatrix(self._data.copy())

    def __getitem__(self, key):
        """
        Convenience wrapper, getter on self._data.
        """

        return self._data[key]

    def __len__(self):
        """
        Convenience function for getting length of slowest stack axis.
        """
        return self.stack_shape[0]

    def __check_compatible(self, other):
        """
        Sanity check two `DiagMatrix` instances are compatible
        """

        self.__check_size_compatible(other)
        self.__check_dtype_compatible(other)

    def __check_size_compatible(self, other):
        """
        Sanity check two `DiagMatrix` instances are compatible in size
        for addition operators. (Same size)

        :param other: The `DiagMatrix` to compare with self.
        """

        if self.count != other.count:
            raise RuntimeError(
                "DiagMatrix instances are "
                f"not same dimension {self.count} != {other.count}"
            )
        try:
            _ = np.broadcast_shapes(self.stack_shape, other.stack_shape)
        except ValueError:
            raise ValueError(
                f"Attempting to broadcast incompatible shapes: {self.stack_shape} with {other.stack_shape}."
            )

    def __check_dtype_compatible(self, other):
        """
        Sanity check two `DiagMatrix` instances are compatible in dtype.

        :param other: The `DiagMatrix` to compare with self.
        """

        if self.dtype != other.dtype:
            raise RuntimeError(
                "DiagMatrix received different types,"
                f"self: {self.dtype} and other: {other.dtype}.  Please validate and cast"
                " as appropriate."
            )

    def add(self, other):
        """
        Define elementwise addition of `DiagMatrix` instances with
        scalars, `BlkDiagMatrix`, and `DiagMatrix`.

        :param other: scalar, `BlkDiagMatrix` or `DiagMatrix.
        :return:  `DiagMatrix` instance with elementwise sum equal
            to self + other. In the case of `BlkDiagMatrix` a `BlkDiagMatrix` is returned.
        """

        if is_scalar_type(other):
            res = DiagMatrix(self._data + other)
        elif isinstance(other, DiagMatrix):
            self.__check_compatible(other)
            res = DiagMatrix(self._data + other._data)
        elif isinstance(other, BlkDiagMatrix):
            # res is blk_diag_matrix
            res = self.as_blk_diag(other.partition) + other
        else:
            raise NotImplementedError(
                f"`add` operation not implemented for {type(other)}"
            )

        return res

    def __add__(self, other):
        """
        Operator overloading for addition.
        """

        return self.add(other)

    def __radd__(self, other):
        """
        Convenience function for elementwise scalar addition.
        """

        return self.add(other)

    def sub(self, other):
        """
        Define elementwise subtraction of `DiagMatrix` instances by
        scalars, `BlkDiagMatrix`, and `DiagMatrix`.

        :param other: scalar, `BlkDiagMatrix` or `DiagMatrix.
        :return:  `DiagMatrix` instance with elementwise sub equal
            to self - other. In the case of `BlkDiagMatrix` a `BlkDiagMatrix` is returned.
        """

        if is_scalar_type(other):
            res = DiagMatrix(self._data - other)
        elif isinstance(other, DiagMatrix):
            self.__check_compatible(other)
            res = DiagMatrix(self._data - other._data)
        elif isinstance(other, BlkDiagMatrix):
            # res is blk_diag_matrix
            res = self.as_blk_diag(other.partition) - other
        else:
            raise NotImplementedError(
                f"`add` operation not implemented for {type(other)}"
            )

        return res

    def __sub__(self, other):
        """
        Operator overloading for subtraction.
        """

        return self.sub(other)

    def __rsub__(self, other):
        """
        Convenience function for elementwise scalar subtraction.
        """

        # Note, the case of DiagMatrix_L - DiagMatrix_R would be
        #   evaluated as L.sub(R), so this is only for other
        #   Object - DiagMatrix situations, namely scalars.

        return -(self - other)

    def matmul(self, other):
        """
        Compute the matrix multiplication of two `DiagMatrix` instances.

        :param other: The rhs `DiagMatrix`, `BlkDiagMatrix` or 2d dense Numpy array.
        :return: Returns `self` @ `other`, as type of `other`
        """
        if isinstance(other, DiagMatrix):
            res = self.mul(other)

        elif isinstance(other, BlkDiagMatrix):
            if self.stack_shape != ():
                raise RuntimeError(
                    f"Mixed `matmul` only implemented for singletons at this time, received {self.stack_shape}."
                )

            res = BlkDiagMatrix.zeros_like(other)

            ind = 0
            for b, blk in enumerate(other):
                i = len(blk)

                res[b] = np.diag(self._data[ind : ind + i]) @ blk
                ind += i

        elif isinstance(other, np.ndarray):
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
        else:
            raise RuntimeError(f"__matmul__ not implemented for {type(other)}.")

        return res

    def __matmul__(self, other):
        """
        Operator overload for matrix multiply of `DiagMatrix` instances.
        """

        return self.matmul(other)

    def __rmatmul__(self, lhs):
        """
        Compute the right matrix multiplication with a `DiagMatrix` instance,
        and a numpy array, `lhs` @ `self`.

        :param other: The lhs Numpy instance.
        :return: Returns numpy array representing `other @ self`.
        """

        # Note, we should only hit this method when mixing DiagMatrix with numpy.
        #   This is because if both a and b are DiagMatrix,
        #   then a@b would be handled first by a.__matmul__(b), never reaching here.
        #   If a is BlkDiagMatrix, then it will handle the conversion of DiagMatrix b.

        if not isinstance(lhs, np.ndarray):
            raise RuntimeError(f"__rmatmul__ not implemented for {type(lhs)}.")

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

    def mul(self, other):
        """
        Compute the elementwise multiplication of a `DiagMatrix` instance and a
        scalar or `DiagMatrix`.

        :param other: The rhs in the multiplication..
        :return: A `self` * `other` as type of `other`.
        """
        if isinstance(other, DiagMatrix):
            self.__check_compatible(other)
            res = DiagMatrix(self._data * other._data)
        elif is_scalar_type(other):  # scalar
            res = DiagMatrix(self._data * other)
        else:
            raise NotImplementedError(f"mul not implemented for {type(other)}.")

        return res

    def __mul__(self, val):
        """
        Operator overload for `DiagMatrix` scalar multiply.
        """

        return self.mul(val)

    def __rmul__(self, other):
        """
        Convenience function, elementwise rmul commutes to mul.
        """
        # `mul` will attempt all compatible multiplications.
        return self.mul(other)

    def neg(self):
        """
        Compute the unary negation of `DiagMatrix` instance.

        :return: A `DiagMatrix` like self.
        """
        return DiagMatrix(-self._data)

    def __neg__(self):
        """
        Operator overload for unary negation of `DiagMatrix` instance.
        """

        return self.neg()

    def abs(self):
        """
        Compute the elementwise absolute value of `DiagMatrix` instance.

        :return: A `DiagMatrix` like self.
        """

        return DiagMatrix(np.abs(self._data))

    def __abs__(self):
        """
        Operator overload for absolute value of `DiagMatrix` instance.
        """

        return self.abs()

    def pow(self, val):
        """
        Compute the elementwise power of `DiagMatrix` instance.

        :param val: Value to exponentiate by.
        :return: A `DiagMatrix` like self.
        """

        return DiagMatrix(self._data**val)

    def __pow__(self, val):
        """
        Operator overload pow of `DiagMatrix` instance.
        """

        return self.pow(val)

    @property
    def norm(self):
        """
        Compute the 2-norm of a `DiagMatrix` instance.

        :return: The 2-norm of the `DiagMatrix` instance.
        """
        # Elements of a diag matrix are its singular values,
        #   and the norm is equal to the largest singular value.
        return np.abs(self._data).max(axis=-1)

    # Transpose methods are provided for reasons of interoperability
    # in code that also uses with BlkDiagMatrix.
    def transpose(self):
        """
        Get the transpose matrix of a `DiagMatrix` instance.

        :return: The corresponding transpose form as a `DiagMatrix`.
        """
        return self.copy()

    @property
    def T(self):
        """
        Syntactic sugar for self.transpose().
        """

        return self.transpose()

    def dense(self):
        """
        Convert `DiagMatrix` instance into full matrix.

        :return: The `DiagMatrix` instance including the zero elements of
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
        coefficient column vectors.

        :param X: Coefficient matrix (ndarray), each column is a coefficient vector.
        :return: A matrix with new coefficient column vectors.
        """

        # Transpose X to become row major because,
        # X is a coefficient matrix (ndarray), each column is a coefficient vector.
        # Transpose the row major multiplication result back to column major, to
        # return a matrix with new coefficient column vectors.
        return (self * DiagMatrix(X.T)).asnumpy().T

    def rapply(self, X):
        """
        Right apply.  Given a matrix of coefficient vectors,
        applies the diagonal matrix on the right hand side.
        Example, X @ self.

        This is the right hand side equivalent to `apply`,
        which due to being diagonal, is the same as `apply`.
        This method exists purely for interoperability with code
        originally targeting `BlkDiagMatrix`.

        :param X: Coefficient matrix, each column is a coefficient vector.

        :return: A matrix with new coefficient vectors.
        """

        return self.apply(X)

    # `eigval` method is provided for reasons of interoperability
    # in code that also uses with `BlkDiagMatrix`.
    def eigvals(self):
        """
        Compute the eigenvalues of a `DiagMatrix`.

        :return: Array of eigvals, with length equal to the fully expanded matrix diagonal.
        """
        return self.asnumpy()

    @staticmethod
    def empty(shape, dtype=np.float32):
        """
        Instantiate an empty `DiagMatrix` with `shape`.
        When shape is an integer, this corresponds to
        the diag(A) where A is (n,n).
        Note, like Numpy, empty values are uninitialized.

        :param shape: Shape of matrix.
            When integer, corresponds to len(diag(A)).  Otherwise,
            when a tuple (..., n), the last dimension `n` defines the
            length of diagonal, while prior dimensions define any
            stack axes.
        :param dtype: Datatype, defaults to np.float32.
        :return: `DiagMatrix` instance.
        """

        return DiagMatrix(np.empty(shape, dtype=dtype))

    @staticmethod
    def zeros(shape, dtype=np.float32):
        """
        Instantiate a zero intialized `DiagMatrix`.
        When `shape` is an integer, this corresponds to
        the diag(A) where A is (n,n).

        :param shape: Shape of matrix.
            When integer, corresponds to len(diag(A)).  Otherwise,
            when a tuple (..., n), the last dimension `n` defines the
            length of diagonal, while prior dimensions define any
            stack axes.
        :param dtype: Datatype, defaults to np.float32.
        :return: `DiagMatrix` instance.
        """

        return DiagMatrix(np.zeros(shape, dtype=dtype))

    @staticmethod
    def ones(shape, dtype=np.float32):
        """
        Instantiate ones intialized `DiagMatrix`.
        When `shape` is an integer, this corresponds to
        the diag(A) where A is (n,n).

        :param shape: Shape of matrix.
            When integer, corresponds to len(diag(A)).  Otherwise,
            when a tuple (..., n), the last dimension `n` defines the
            length of diagonal, while prior dimensions define any
            stack axes.
        :param dtype: Datatype, defaults to np.float32.
        :return: `DiagMatrix` instance.
        """

        return DiagMatrix(np.ones(shape, dtype=dtype))

    @staticmethod
    def eye(shape, dtype=np.float32):
        """
        Build a `DiagMatrix` eye (identity) matrix.
        This is simply an alias for `ones`.

        :param shape: Shape of matrix.
            When integer, corresponds to len(diag(A)).  Otherwise,
            when a tuple (..., n), the last dimension `n` defines the
            length of diagonal, while prior dimensions define any
            stack axes.
        :param dtype: Datatype, defaults to np.float32.
        :return: `DiagMatrix` instance.
        """

        return DiagMatrix.ones(shape, dtype=dtype)

    def as_blk_diag(self, partition):
        """
        Express `DiagMatrix` as a `BlkDiagMatrix` using `partition`.

        :param partition: BlkDiagMatrix partition.
        :return: `BlkDiagMatrix`
        """
        if self.stack_shape != ():
            raise RuntimeError(
                f"as_blk_diag only implemented for singletons at this time, received {self.stack_shape}."
            )

        B = BlkDiagMatrix(partition, dtype=self.dtype)
        ind = 0
        for b, p in enumerate(partition):
            if p[0] != p[1]:
                raise RuntimeError(
                    f"Block {b} of partition is not square, {p}."
                    "Currently DiagMatrix.as_blk_diag only supports square blocks."
                )

            j = p[0]
            B[b] = np.diag(self._data[ind : ind + j])
            ind += j

        return B

    def solve(self, b):
        """
        For this `DiagMatrix` `a` and vector `b`.
        solve a x = b for `x`.

        :param b: Right hand side, Numpy array.
        :return: `DiagMatrix`, solution `x`.
        """

        if self.stack_shape != ():
            raise RuntimeError(
                f"`solve` only implemented for singletons at this time, received {self.stack_shape}."
            )

        return DiagMatrix(b / self._data)
