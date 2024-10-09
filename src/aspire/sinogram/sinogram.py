import logging

import numpy as np

import aspire.image
from aspire.nufft import anufft
from aspire.numeric import fft, xp

logger = logging.getLogger(__name__)


class Sinogram:
    def __init__(self, data, dtype=None):
        """
        Initialize a Sinogram Object. This is a stack of one or more line projections or sinograms.

        The stack can be multidimensional with 'self.n' equal to the product
        of the stack dimensions. Singletons will be expanded into a stack
        with one entry.

        :param data: Numpy array containing image data with shape
            `(..., angles, radial points)`.
        :param dtype: Optionally cast `data` to this dtype.
            Defaults to `data.dtype`.

        :return: Sinogram instance holding `data`.
        """
        if dtype is None:
            self.dtype = data.dtype
        else:
            self.dtype = np.dtype(dtype)

        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        if data.ndim < 3:
            raise ValueError(
                f"Invalid data shape: {data.shape}. Expected shape: (..., angles, radial_points), where '...' is the stack number."
            )

        self._data = data.astype(self.dtype, copy=False)
        self.ndim = self._data.ndim
        self.shape = self._data.shape
        self.stack_shape = self._data.shape[:-2]
        self.stack_n_dim = self._data.ndim - 2
        self.n = np.prod(self.stack_shape)
        self.n_angles = self._data.shape[-2]
        self.n_radial_points = self._data.shape[-1]

        # Numpy interop
        # https://numpy.org/devdocs/user/basics.interoperability.html#the-array-interface-protocol
        self.__array_interface__ = self._data.__array_interface__
        self.__array__ = self._data

    def _check_key_dims(self, key):
        if isinstance(key, tuple) and (len(key) > self._data.ndim):
            raise ValueError(
                f"Sinogram stack_dim is {self.stack_n_dim}, slice length must be =< {self.n_dim}"
            )

    def __getitem__(self, key):
        self._check_key_dims(key)
        return self.__class__(self._data[key])

    def __setitem__(self, key, value):
        self._check_key_dims(key)
        self._data[key] = value

    def stack_reshape(self, *args):
        """
        Reshape the stack axis.

        :*args: Integer(s) or tuple describing the intended shape.

        :return: Sinogram instance
        """

        # If we're passed a tuple, use that
        if len(args) == 1 and isinstance(args[0], tuple):
            shape = args[0]
        else:
            # Otherwise use the variadic args
            shape = args

        # Sanity check the size
        if shape != (-1,) and np.prod(shape) != self.n:
            raise ValueError(
                f"Number of sinogram images {self.n} cannot be reshaped to {shape}."
            )

        return self.__class__(self._data.reshape(*shape, *self._data.shape[-2:]))

    def asnumpy(self):
        """
        Return image data as a (<stack>, angles, radians)
        read-only array view.

        :return: read-only ndarray view
        """

        view = self._data.view()
        view.flags.writeable = False
        return view

    def copy(self):
        return self.__class__(self._data.copy())

    def __str__(self):
        return f"Sinogram(n_images = {self.n}, n_angles = {self.n_angles}, n_radial_points = {self.n_radial_points})"

    def __repr__(self):
        msg = f"Sinogram: {self.n} images of dtype {self.dtype}, "
        msg += f"arranged as a stack with shape {self.stack_shape}. "
        msg += f"Each image has {self.n_angles} angles and {self.n_radial_points} radial points."
        return msg

    def backproject(self, angles):
        """
        Backprojection method for a single stack of lines.

        :param angles: np.ndarray
            1D array of angles in radians. Each entry in the array
            corresponds to different angles which are used to
            reconstruct the image.
        :return: An Image object containing the original stack size
            with a newly reconstructed numpy array of the images.
            Expected return shape should be (..., n_radial_points, n_radial_points)
        """
        if len(angles) != self.n_angles:
            raise ValueError("Number of angles must match the number of projections.")

        original_stack_shape = self.stack_shape
        sinogram = xp.asarray(self.stack_reshape(-1)._data)
        L = self.n_radial_points
        sinogram = fft.ifftshift(sinogram, axes=-1)
        sinogram_ft = fft.rfft(sinogram, axis=-1)
        sinogram_ft *= xp.pi  # Fix scale to match
        sinogram_ft[..., 0] /= 2  # Fix DC
        angles = xp.asarray(angles)

        # grid generation with real points
        y_idx = fft.rfftfreq(self.n_radial_points) * xp.pi * 2
        n_real_points = len(y_idx)
        pts = xp.empty((2, len(angles), n_real_points), dtype=self.dtype)
        pts[0] = y_idx[xp.newaxis, :] * xp.sin(angles)[:, xp.newaxis]
        pts[1] = y_idx[xp.newaxis, :] * xp.cos(angles)[:, xp.newaxis]

        imgs = anufft(
            sinogram_ft.reshape(self.n, -1),
            pts.reshape(2, n_real_points * len(angles)),
            sz=(L, L),
            real=True,
        ).reshape(self.n, L, L)

        imgs = imgs / (self.n_radial_points * len(angles))
        return aspire.image.Image(xp.asnumpy(imgs)).stack_reshape(original_stack_shape)
