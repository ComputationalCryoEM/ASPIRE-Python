import logging

import numpy as np

from .basis import Basis
from .steerable import SteerableBasis2D

logger = logging.getLogger(__name__)


class Coef:
    """
    Numpy interoperable basis stacks.
    """

    def __init__(self, basis, data, dtype=None):
        """
        A stack of one or more coefficient arrays.

        The stack can be multidimensional with `n_stack` equal
        to the product of the stack dimensions.  Singletons will be
        expanded into a stack of one entry.

        The last axes represents the coefficient count

        :param data: Numpy array containing image data with shape
            `(..., count)`.
        :param dtype: Optionally cast `data` to this dtype.
            Defaults to `data.dtype`.

        :return: Image instance holding `data`.
        """

        if isinstance(data, Coef):
            data = data.asnumpy()
        elif not isinstance(data, np.ndarray):
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
        self.n_stack = np.prod(self.stack_shape)
        self.count = self._data.shape[-1]

        if self.count != self.basis.count:
            raise RuntimeError(
                f"Provided data count of {self.count} does not match basis count of {self.basis.count}."
            )

        # Numpy interop
        # https://numpy.org/devdocs/user/basics.interoperability.html#the-array-interface-protocol
        self.__array_interface__ = self.asnumpy().__array_interface__
        self.__array__ = self.asnumpy()

    def __len__(self):
        """
        Return stack length.

        Note this is product of all stack dimensions.
        """
        return self.n_stack

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
        if shape != (-1,) and np.prod(shape) != self.n_stack:
            raise ValueError(
                f"Number of images {self.n_stack} cannot be reshaped to {shape}."
            )

        return self.__class__(
            self.basis, self._data.reshape(*shape, self._data.shape[-1])
        )

    def copy(self):
        return self.__class__(self.basis, self._data.copy())

    def evaluate(self):
        return self.basis.evaluate(self)

    def rotate(self, radians, refl=None):
        """
        Returns coefs rotated counter-clockwise by `radians`.

        Raises error if underlying coef basis does not support rotations.

        :param radians: Rotation in radians.
        :param refl: Optional reflect image (about y=0) (bool)
        :return: rotated coefs.
        """
        if not isinstance(self.basis, SteerableBasis2D):
            raise RuntimeError(f"self.basis={self.basis} is not SteerableBasis.")

        return self.basis.rotate(self, radians, refl)

    def shift(self, shifts):
        """
        Returns coefs shifted by `shifts`.

        This will transform to real cartesian space, shift,
        and transform back to Polar Fourier-Bessel space.

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
        if isinstance(other, Coef):
            other = other._data

        return self.__class__(self.basis, self._data * other)

    def __add__(self, other):
        if isinstance(other, Coef):
            other = other._data

        return self.__class__(self.basis, self._data + other)

    def __sub__(self, other):
        if isinstance(other, Coef):
            other = other._data

        return self.__class__(self.basis, self._data - other)

    def __neg__(self):
        return self.__class__(self.basis, -self._data)

    def size(self):
        """
        Return np.size of underlying data.
        """
        return np.size(self._data)

    def by_indices(self, **kwargs):
        """
        Select coefficients by indices (`radial`, `angular`).

        See `SteerableBasis.indices_mask` for argument details.

        :return: `Coef` vector.
        """

        mask = self.basis.indices_mask(**kwargs)
        return self._data[:, mask]
