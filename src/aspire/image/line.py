import aspire
import numpy as np


class Line:
    def __init__(self, data, dtype = np.float64):
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
            raise('Projection Dimensions should be more than Three-Dimensions')
        self._data = data.astype(self.dtype, copy=False)
        self.stack_shape = self._data.shape[:-2]
        self.n_images = self._data.shape[0] #broken for higher dimensional stacks
        self.n_lines = self._data.shape[-2] 
        self.n_points = self._data.shape[-1]
        # self.n_dim = (self._data.shape[1], self._data.shape[2])

    def __str__(self):
        return f"Line(n_images = {self.n_images}, n_points = {self.n_points})"

    @property
    def stack(self):
        return self.n_images

    # talk about angles and if they're supposed to be a certain input/output
    # why this method and not another (explain design choices)
    def back_project(self, angles):
        """
        Back Projection Method for a single stack of lines.
        
        :param filter_name: string, optional
            Filter used in frequency domain filtering. Assign None to use no filter.
        :param angles: array
            assuming not perfectly radial angles
        :return: stack of reconstructed
        """
        assert len(angles) == self.n_lines, "Angles must match the number of lines."
        original_stack = self.stack_shape
        n_angles = len(angles)

        ## our implementation
        n_img, n_angles, n_rad = sinogram.shape
        assert n_angles == len(angles), "gonna have a bad time"
        L = n_rad
        sinogram = np.fft.ifftshift(self.n_images, axes= -1)
        sinogram_ft = np.fft.rfft(self.n_images, axis= -1)

        #grid generation
        y_idx = np.fft.rfftfreq(n_rad) * np.pi * 2
        n_real_points = len(y_idx)
        pts = np.empty((2, len(angles), n_real_points), dtype=self.dtype)
        pts[0] = y_idx[np.newaxis, :] * np.sin(angles)[:, np.newaxis]
        pts[1] = y_idx[np.newaxis, :] * np.cos(angles)[:, np.newaxis]
        
        imgs = aspire.nufft.anufft(
            sinogram_ft.reshape(n_img, -1),
            pts.reshape(2, n_real_points * len(angles)),
            sz=(L, L),
            real=True
        ).reshape(n_img, L, L)
        
        return aspire.image.Image(imgs)
    
    def image_filter(self, filter_name, projections):
        """
        Filter Method for projections. Will apply filter to line projection to get collection of projections (ramp, cosine, ... , etc.)
        :param projections: Collection of line projections that need to be filtered.
        :return: Filtered Projections.
        """
        if projections is None:
            raise ValueError('The input projections must not be None')

        filter_types = ('ramp', 'shepp-i logan', 'cosine', 'hamming', 'hann', None)
        if filter_name is not filter_type:
            raise ValueError(f"Unknown filter: {filter_name}")

        # skimage filter 
        fourier_filter = _get_fourier_filter(projection_size_padded, filter_name)
        projection = fft(img, axis=0) * fourier_filter
        radon_filtered = np.real(ifft(projection, axis=0)[:img_shape, :])

        """
        step 0: Look more into filter function from skimage
        thoughts: apply filter to each point
        """
        pass
