import logging

import numpy as np

from .basis import Basis
from .steerable import SteerableBasis2D

logger = logging.getLogger(__name__)


class Coef:
    """
    Numpy interoperable basis stacks.
    """

    def __init__(self, basis, data, dtype=np.float64):
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

        if not isinstance(basis, Basis):
            raise TypeError(f"`basis` is required to be a `Basis` instance.")
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
                "Provided data count of {self.count} does not match basis count of {self.basis.count}."
            )

        # Numpy interop
        # https://numpy.org/devdocs/user/basics.interoperability.html#the-array-interface-protocol
        self.__array_interface__ = self.asnumpy().__array_interface__
        self.__array__ = self.asnumpy()

    def asnumpy(self):
        """
        Return image data as a (<stack>, count)
        read-only array view.

        :return: read-only ndarray view
        """

        view = self._data.view()
        view.flags.writeable = False
        return view

    def copy(self):
        return self.__class__(self._data.copy())

    def evaluate(self):
        return self.basis.evaluate(self.asnumpy())

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

        return self.basis.rotate(self.asnumpy(), radians, refl)

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

        return self.basis.shift(self.asnumpy(), shifts)
