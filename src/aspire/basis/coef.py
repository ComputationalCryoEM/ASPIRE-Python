import logging

import numpy as np

from .basis import Basis

logger = logging.getLogger(__name__)


class Coef:
    """
    Numpy interoperable basis stacks.
    """
    def __init__(self, basis, data, dtype=np.float64):
        """
        A stack of one or more coefficient arrays.

        The stack can be multidimensional with `n_coefs` equal
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
        self.n_coefs = np.prod(self.stack_shape)
        self.count = self._data.shape[-1]
