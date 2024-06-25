import logging

import numpy as np
from scipy.sparse.linalg import LinearOperator

from aspire.image import Image
from aspire.numeric.scipy import cg
from aspire.utils import mdim_mat_fun_conj
from aspire.volume import Volume

logger = logging.getLogger(__name__)


class Coef:
    """
    Numpy interoperable container for stacks of real coefficient vectors.
    Each `Coef` instance has an associated `Basis`.
    """

    _allowed_dtypes = (np.float32, np.float64)

    def __init__(self, basis, data, dtype=None):
        """
        A stack of one or more coefficient arrays.

        The stack can be multidimensional with `stack_size` equal
        to the product of the stack dimensions.  Singletons will be
        expanded into a 1D stack of length one.

        The last axes always represents the coefficient `count`.

        :param basis: `Basis` associated with `data` coefficients.
        :param data: Numpy array containing image data with shape
            `(..., count)`.
        :param dtype: Optionally cast `data` to this dtype.
            Defaults to `data.dtype`.

        :return: `Coef` instance holding `data`.
        """

        if not isinstance(data, np.ndarray):
            raise ValueError("Coef should be instantiated with an ndarray")

        if data.ndim < 1:
            raise ValueError(
                "Coef data should be ndarray with shape (N1...) x count or (count)."
            )
        elif data.ndim == 1:
            data = np.expand_dims(data, axis=0)

        if dtype is None:
            self.dtype = data.dtype
        else:
            self.dtype = np.dtype(dtype)

        # Check real/complex dtype based on class.
        self._check_dtype()

        if not isinstance(basis, Basis):
            raise TypeError(
                f"`basis` is required to be a `Basis` instance, received {type(basis)}"
            )
        self.basis = basis

        self._data = data.astype(self.dtype, copy=False)
        self.ndim = self._data.ndim
        self.shape = self._data.shape
        self.stack_ndim = self._data.ndim - 1
        self.stack_shape = self._data.shape[:-1]
        self.stack_size = np.prod(self.stack_shape)
        self.count = self._data.shape[-1]

        # Derive count from basis.
        basis_count = self._get_basis_count()

        if self.count != basis_count:
            raise RuntimeError(
                f"Provided data count of {self.count} does not match basis count of {basis_count}."
            )

        # Numpy interop
        # https://numpy.org/devdocs/user/basics.interoperability.html#the-array-interface-protocol
        self.__array_interface__ = self.asnumpy().__array_interface__
        self.__array__ = self.asnumpy()

    def _check_dtype(self):
        """
        Private helper method to check real/complex dtype based on class `_allowed_dtypes`.

        Raises on mismatch.
        """

        if self.dtype not in self._allowed_dtypes:
            raise TypeError(
                f"{self.__class__.__name__} requires {self._allowed_dtypes} coefficients, attempted {self.dtype}."
            )

    def _get_basis_count(self):
        """
        Private helper method to return coefficient count from basis.

        :return: Basis count (integer).
        """
        return int(self.basis.count)

    def __len__(self):
        """
        Return length of slowest stack axis.
        """
        return self.stack_shape[0]

    def asnumpy(self):
        """
        Return image data as a (<stack>, count)
        read-only array view.

        :return: read-only ndarray view
        """

        view = self._data.view()
        view.flags.writeable = False
        return view

    def _check_key_dims(self, key):
        if isinstance(key, tuple) and (len(key) > self._data.ndim):
            raise ValueError(
                f"Coef stack_dim is {self.stack_ndim}, slice length must be =< {self.ndim}"
            )

    def __getitem__(self, key):
        self._check_key_dims(key)
        return self.__class__(self.basis, self._data[key])

    def __setitem__(self, key, value):
        self._check_key_dims(key)
        self._data[key] = value

    def stack_reshape(self, *args):
        """
        Reshape the stack axis.

        :*args: Integer(s) or tuple describing the intended shape.

        :returns: Coef instance
        """

        # If we're passed a tuple, use that
        if len(args) == 1 and isinstance(args[0], tuple):
            shape = args[0]
        else:
            # Otherwise use the variadic args
            shape = args

        # Sanity check the size
        if shape != (-1,) and np.prod(shape) != self.stack_size:
            raise ValueError(
                f"Number of coefficient vectors {self.stack_size} cannot be reshaped to {shape}."
            )

        return self.__class__(
            self.basis, self._data.reshape(*shape, self._data.shape[-1])
        )

    def copy(self):
        """
        Return a new `Coef` instance with a deep copy of the data.
        """
        return self.__class__(self.basis, self._data.copy())

    def evaluate(self):
        """
        Return the evaluation of coefficients in the associated `basis`.
        """
        return self.basis.evaluate(self)

    def rotate(self, radians, refl=None):
        """
        Returns coefs rotated counter-clockwise by `radians`.

        Raises error if underlying coef basis does not support rotations.

        :param radians: Rotation in radians.
        :param refl: Optional reflect image (about y=0) (bool)
        :return: rotated coefs.
        """

        if not callable(getattr(self.basis, "rotate", None)):
            raise RuntimeError(
                f"self.basis={self.basis} does not provide `rotate` method."
            )

        return self.basis.rotate(self, radians, refl)

    def shift(self, shifts):
        """
        Returns coefs shifted by `shifts`.

        This will transform to real cartesian space, shift,
        and transform back to basis space.

        :param coef: Basis coefs.
        :param shifts: Shifts in pixels (x,y). Shape (1,2) or (len(coef), 2).
        :return: coefs of shifted images.
        """

        if not callable(getattr(self.basis, "shift", None)):
            raise RuntimeError(
                f"self.basis={self.basis} does not provide `shift` method."
            )

        return self.basis.shift(self, shifts)

    def __mul__(self, other):
        """
        Overload operator for multiplication.

        :param other: `Coef` instance to multiply with.
            Also allows for multiplication by Numpy arrays and scalars.
        :return: `Coef` instance.
        """

        if isinstance(other, Coef):
            other = other._data

        return self.__class__(self.basis, self._data * other)

    def __add__(self, other):
        """
        Overload operator for addition.

        :param other: `Coef` instance to add.
            Also allows for addition by Numpy arrays and scalars.
        :return: `Coef` instance.
        """

        if isinstance(other, Coef):
            other = other._data

        return self.__class__(self.basis, self._data + other)

    def __sub__(self, other):
        """
        Overload operator for subtraction.

        :param other: `Coef` instance to subtract.
            Also allows for subtraction by Numpy arrays and scalars.
        :return: `Coef` instance.
        """

        if isinstance(other, Coef):
            other = other._data

        return self.__class__(self.basis, self._data - other)

    def __neg__(self):
        """
        Overload operator for negation.

        :return: `Coef` instance.
        """

        return self.__class__(self.basis, -self._data)

    @property
    def size(self):
        """
        Return np.size of underlying data.

        This should be `stack_size * count`,
        or `len(self) * count`.
        """
        return np.size(self._data)

    # This is included for completion, but is not being adopted yet.
    def by_indices(self, **kwargs):
        """
        Select coefficients by indices (`radial`, `angular`).

        See `SteerableBasis.indices_mask` for argument details.

        :return: Numpy array.
        """

        mask = self.basis.indices_mask(**kwargs)
        return self._data[..., mask]

    def to_complex(self):
        """
        Convert and return real coefficients as `ComplexCoef`.
        """
        return self.basis.to_complex(self)

    def to_real(self):
        """
        Not implemented for real Coef.
        """
        raise TypeError("Coef already real.")


class ComplexCoef(Coef):
    """
    Numpy interoperable container for stacks of complex coefficient vectors.
    Each `ComplexCoef` instance has an associated `Basis`.
    """

    _allowed_dtypes = (np.complex64, np.complex128)

    def _get_basis_count(self):
        """
        Private helper method to return coefficient complex count from basis.

        :return: Basis complex count (integer).
        """

        return int(self.basis.complex_count)

    def evaluate(self):
        """
        Return the evaluation of coefficients in the associated `basis`.
        """
        return self.to_real().evaluate()

    def rotate(self, radians, refl=None):
        """
        Returns coefs rotated counter-clockwise by `radians`.

        Raises error if underlying coef basis does not support rotations.

        :param radians: Rotation in radians.
        :param refl: Optional reflect image (about y=0) (bool)
        :return: Rotated ComplexCoefs.
        """

        return self.to_real().rotate(radians, refl).to_complex()

    def shift(self, shifts):
        """
        Returns complex coefs shifted by `shifts`.

        This will transform to real cartesian space, shift,
        and transform back to basis space.

        :param coef: Basis coefs.
        :param shifts: Shifts in pixels (x,y). Shape (1,2) or (len(coef), 2).
        :return: Complex coefs of shifted images.
        """

        return self.to_real().shift(shifts).to_complex()

    def to_real(self):
        """
        Convert and return complex coefficients as `Coef`.
        """
        return self.basis.to_real(self)

    def to_complex(self):
        """
        Not implemented for ComplexCoef.
        """
        raise TypeError("ComplexCoef already complex.")


class Basis:
    """
    Define a base class for expanding 2D particle images and 3D structure volumes

    """

    def __init__(self, size, ell_max=None, dtype=np.float32):
        """
        Initialize an object for the base of basis class

        :param size: The size of the vectors for which to define the basis.
            Currently only square images and cubic volumes are supported.
        :param ell_max: The maximum order ell of the basis elements. If no input
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
        if self.ndim == 2:
            self._cls = Image
        elif self.ndim == 3:
            self._cls = Volume
        else:
            raise RuntimeError("Basis ndim must be 2 or 3")
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.float32, np.float64):
            raise NotImplementedError(
                "Currently only implemented for float32 and float64 types"
            )
        # dtype of coefficients is the same as self.dtype for real bases
        # subclasses with complex coefficients override this attribute
        self.coefficient_dtype = self.dtype

        self._build()

    def _build(self):
        """
        Build the internal data structure to represent basis
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

        :param v: `Coef` instance containing the coefficients to be
            evaluated. The first dimension must correspond to the
            number of coefficient vectors, while the second must
            correspond to `self.count`.
        :return: The evaluation of the coefficient vector(s) `v` for this basis.
            This is an Image or a Volume object containing one image/volume for each
            coefficient vector, and of size `self.sz`.
        """

        if v.dtype != self.coefficient_dtype:
            logger.warning(
                f"{self.__class__.__name__}::evaluate"
                f" Inconsistent dtypes v: {v.dtype} self coefficient dtype: {self.coefficient_dtype}"
            )

        if not isinstance(v, Coef):
            raise TypeError(f"`evaluate` should be passed a `Coef`, received {type(v)}")

        # Flatten stack
        stack_shape = v.stack_shape
        v = v.stack_reshape(-1).asnumpy()

        # Compute the transform
        x = self._evaluate(v)
        # Restore stack shape
        x = x.reshape(*stack_shape, *self.sz)

        # Return the appropriate class
        return self._cls(x)

    def _evaluate(self, v):
        raise NotImplementedError("subclasses must implement this")

    def evaluate_t(self, v):
        """
        Evaluate coefficient in dual basis

        :param v: An Image or Volume object whose size matches `self.sz`.
        :return: The evaluation of the Image or Volume object `v` in the dual
            basis of `basis`.
            This is an array of vectors whose first dimension equals the number of
            images/volumes in `v`. and whose second dimension is `self.count`.
        """
        if v.dtype != self.dtype:
            logger.warning(
                f"{self.__class__.__name__}::evaluate_t"
                f" Inconsistent dtypes v: {v.dtype} self: {self.dtype}"
            )

        if not isinstance(v, self._cls):
            logger.warning(
                f"{self.__class__.__name__}::evaluate_t"
                f" passed numpy array instead of {self._cls}."
            )
        else:
            v = v.asnumpy()

        # Flatten stack, ndim is wrt Basis (2 or 3)
        stack_shape = v.shape[: -self.ndim]
        v = v.reshape(-1, *v.shape[-self.ndim :])
        # Compute the adjoint
        x = self._evaluate_t(v)
        # Restore stack shape
        x = x.reshape(*stack_shape, self.count)

        return Coef(self, x)

    def _evaluate_t(self, v):
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
        return mdim_mat_fun_conj(V, 1, len(self.sz), self._evaluate)

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
        return mdim_mat_fun_conj(X, len(self.sz), 1, self._evaluate_t)

    def expand(self, x, tol=None, atol=0):
        """
        Obtain coefficients in the basis from those in standard coordinate basis

        This is a similar function to evaluate_t but with more accuracy by using
        the cg optimizing of linear equation, Ax=b.

        :param x: An array whose last two or three dimensions are to be expanded
            the desired basis. These dimensions must equal `self.sz`.
        :param tol: Relative tolerance for convergence, `norm(residual) <= max(tol*norm(b), atol)`.
            Deafult `None` sets to dtype's `eps`*10.
        :param atol: Absolute tolerance for convergence, `norm(residual) <= max(tol*norm(b), atol)`.
        :return: The coefficients of `v` expanded in the desired basis.
            The last dimension of `v` is with size of `count` and the
            first dimensions of the return value correspond to
            those first dimensions of `x`.

        """

        if isinstance(x, Image) or isinstance(x, Volume):
            x = x.asnumpy()

        if x.dtype != self.dtype:
            logger.warning(
                f"{self.__class__.__name__}::expand"
                f" Inconsistent dtypes x: {x.dtype} self: {self.dtype}"
            )

        # TODO: We should  only need to do this block when we are not passed Image/Volume.
        # check that last ndim values of input shape match
        # the shape of this basis
        assert (
            x.shape[-self.ndim :] == self.sz
        ), f"Last {self.ndim} dimensions of x must match {self.sz}."
        # extract number of images/volumes, or () if only one
        sz_roll = x.shape[: -self.ndim]
        # convert to standardized shape e.g. (L,L) to (1,L,L)
        x = x.reshape((-1, *self.sz))

        operator = LinearOperator(
            shape=(self.count, self.count),
            matvec=lambda v: self.evaluate_t(self.evaluate(Coef(self, v))),
            dtype=self.dtype,
        )

        if tol is None:
            # TODO: (from MATLAB implementation) - Check that this tolerance make sense for multiple columns in v
            tol = 10 * np.finfo(x.dtype).eps
        logger.info(f"Expanding array in basis with tol={tol} atol={atol}")

        # number of image samples
        n_data = x.shape[0]
        v = np.zeros((n_data, self.count), dtype=self.coefficient_dtype)

        for isample in range(0, n_data):
            b = self.evaluate_t(self._cls(x[isample])).asnumpy().T
            # TODO: need check the initial condition x0 can improve the results or not.
            v[isample], info = cg(operator, b, rtol=tol, atol=atol)
            if info != 0:
                raise RuntimeError(f"Unable to converge! cg info={info}")

        # return v coefficients with the last dimension of self.count
        v = v.reshape((*sz_roll, self.count))

        return Coef(self, v)
