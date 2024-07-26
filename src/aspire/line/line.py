import logging
import os
from warnings import catch_warnings, filterwarnings, simplefilter, warn

import mrcfile
import numpy as np

import aspire.image
from aspire.nufft import anufft

logger = logging.getLogger(__name__)


class Line:
    def __init__(self, data, dtype=np.float64):
        """
        Initialize a Line Object. Change later (similar intuition from Image class)

        :param data: Numpy array containing image data with shape
            `(..., resolution, resolution)`.
        :param dtype: Optionally cast `data` to this dtype.
            Defaults to `data.dtype`.
        """
        self.dtype = np.dtype(dtype)
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        if data.ndim < 3:
            assert "Projection Dimensions should be more than Three-Dimensions"
        self._data = data.astype(self.dtype, copy=False)
        self.ndim = self._data.ndim
        self.shape = self._data.shape
        self.stack_shape = self._data.shape[:-2]
        self.stack_n_dim = self._data.ndim - 2  # fix
        self.n = np.product(self.stack_shape)  # stack number
        self.n_angles = self._data.shape[-1]  # fix
        self.n_radial_points = self._data.shape[-1]

        # Numpy interop
        # https://numpy.org/devdocs/user/basics.interoperability.html#the-array-interface-protocol
        self.__array_interface__ = self._data.__array_interface__
        self.__array__ = self._data

    def _check_key_dims(self, key):
        if isinstance(key, tuple) and (len(key) > self._data.ndim):
            raise ValueError(
                f"Line stack_dim is {self.stack_n_dim}, slice length must be =< {self.n_dim}"
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

        :returns: Line instance
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
                f"Number of sinogram images {self.n_images} cannot be reshaped to {shape}."
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

    # fix later
    def save(self, mrcs_filepath, overwrite=False):
        if self.stack_ndim > 1:
            raise NotImplementedError("`save` is currently limited to 1D image stacks.")

        with mrcfile.new(mrcs_filepath, overwrite=overwrite) as mrc:
            # original input format (the image index first)
            mrc.set_data(self._data.astype(np.float32))

    # fix later
    @staticmethod
    def load(filepath, dtype=None):
        """
        Load raw data from supported files.

        Currently MRC and TIFF are supported.

        :param filepath: File path (string).
        :param dtype: Optionally force cast to `dtype`.
             Default dtype is inferred from the file contents.
        :return: numpy array of image data.
        """

        # Get the file extension
        ext = os.path.splitext(filepath)[1]

        # On unsupported extension, raise with suggested file types
        if ext not in Image.extensions:
            raise RuntimeError(
                f"Attempting to open unsupported file extension '{ext}', try {list(Image.extensions.keys())}."
            )

        # Call the appropriate file reader
        im = Image.extensions[ext](filepath)

        # Attempt casting when user provides dtype
        if dtype is not None:
            im = im.astype(dtype, copy=False)

        # Return as Image instance
        return Image(im)

    def __str__(self):
        # fix later
        return f"Line(n_images = {self.n}, n_angles = {self.n_points}, n_radial_points = {self.n_radial_points})"

    @property
    def stack(self):
        return self.n_images

    def back_project(self, angles):
        """
        Back Projection Method for a single stack of lines.

        :param filter_name: string, optional
            Filter used in frequency domain filtering. Assign None to use no filter.
        :param angles: array
            assuming not perfectly radial angles
        :return: stack of reconstructed
        """
        sinogram = self._data
        n_img, n_angles, n_rad = sinogram.shape
        assert n_angles == len(
            angles
        ), "Number of angles must match the number of projections"

        L = n_rad
        sinogram = np.fft.ifftshift(sinogram, axes=-1)
        sinogram_ft = np.fft.rfft(sinogram, axis=-1)

        # grid generation with real points
        y_idx = np.fft.rfftfreq(n_rad) * np.pi * 2
        n_real_points = len(y_idx)
        pts = np.empty((2, len(angles), n_real_points), dtype=self.dtype)
        pts[0] = y_idx[np.newaxis, :] * np.sin(angles)[:, np.newaxis]
        pts[1] = y_idx[np.newaxis, :] * np.cos(angles)[:, np.newaxis]

        imgs = anufft(
            sinogram_ft.reshape(n_img, -1),
            pts.reshape(2, n_real_points * len(angles)),
            sz=(L, L),
            real=True,
        ).reshape(n_img, L, L)

        # normalization which gives us roughly the same error regardless of angles
        imgs = imgs / (n_real_points * len(angles))
        return aspire.image.Image(imgs)
