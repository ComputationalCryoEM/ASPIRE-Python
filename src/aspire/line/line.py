import logging

import numpy as np

import aspire.image
from aspire.nufft import anufft

logger = logging.getLogger(__name__)


class Line:
    def __init__(self, data, dtype=np.float64):
        """
        Initialize a Line Object. Change later (similar intuition from Image class)
        Question: Is it a line or collection of line object?

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

    #   self.stack_shape = self._data.shape[:-2]
    #  self.n_images = self._data.shape[0] #broken for higher dimensional stacks
    # self.n_lines = self._data.shape[-2]
    # self.n_points = self._data.shape[-1]
    # self.n_dim = (self._data.shape[1], self._data.shape[2])

    def __str__(self):
        return f"Line(n_images = {self.n_images}, n_points = {self.n_points})"

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
