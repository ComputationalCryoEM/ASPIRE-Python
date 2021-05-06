import logging

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import lstsq

import aspire.volume
from aspire.nufft import anufft
from aspire.numeric import fft, xp
from aspire.utils import ensure
from aspire.utils.coor_trans import grid_2d
from aspire.utils.matlab_compat import m_reshape
from aspire.utils.matrix import anorm

logger = logging.getLogger(__name__)


def _im_translate2(im, shifts):
    """
    Translate image by shifts
    :param im: An Image instance to be translated.
    :param shifts: An array of size n-by-2 specifying the shifts in pixels.
        Alternatively, it can be a row vector of length 2, in which case the same shifts is applied to each image.
    :return: An Image instance translated by the shifts.

    TODO: This implementation has been moved here from aspire.aspire.abinitio and is faster than _im_translate.
    """

    if not isinstance(im, Image):
        logger.warning(
            "_im_translate2 expects an Image, attempting to convert array."
            "Expects array of size n-by-L-by-L."
        )
        im = Image(im)

    if shifts.ndim == 1:
        shifts = shifts[np.newaxis, :]

    n_shifts = shifts.shape[0]

    if shifts.shape[1] != 2:
        raise ValueError("Input `shifts` must be of size n-by-2")

    if n_shifts != 1 and n_shifts != im.n_images:
        raise ValueError("The number of shifts must be 1 or match the number of images")

    resolution = im.res
    grid = xp.asnumpy(
        fft.ifftshift(xp.asarray(np.ceil(np.arange(-resolution / 2, resolution / 2))))
    )
    om_y, om_x = np.meshgrid(grid, grid)
    phase_shifts = np.einsum("ij, k -> ijk", om_x, shifts[:, 0]) + np.einsum(
        "ij, k -> ijk", om_y, shifts[:, 1]
    )
    # TODO: figure out how why the result of einsum requires reshape
    phase_shifts = phase_shifts.reshape(n_shifts, resolution, resolution)
    phase_shifts /= resolution

    mult_f = np.exp(-2 * np.pi * 1j * phase_shifts)
    im_f = xp.asnumpy(fft.fft2(xp.asarray(im.asnumpy())))
    im_translated_f = im_f * mult_f
    im_translated = np.real(xp.asnumpy(fft.ifft2(xp.asarray(im_translated_f))))

    return Image(im_translated)


def normalize_bg(imgs, bg_radius=1.0, do_ramp=True):
    """
    Normalize backgrounds and apply to a stack of images

    :param imgs: A stack of images in N-by-L-by-L array
    :param bg_radius: Radius cutoff to be considered as background (in image size)
    :param do_ramp: When it is `True`, fit a ramping background to the data
            and subtract. Namely perform normalization based on values from each image.
            Otherwise, a constant background level from all images is used.
    :return: The modified images
    """
    L = imgs.shape[-1]
    grid = grid_2d(L)
    mask = grid["r"] > bg_radius

    if do_ramp:
        # Create matrices and reshape the background mask
        # for fitting a ramping background
        ramp_mask = np.vstack(
            (
                grid["x"][mask].flatten(),
                grid["y"][mask].flatten(),
                np.ones(grid["y"][mask].flatten().size),
            )
        ).T
        ramp_all = np.vstack(
            (grid["x"].flatten(), grid["y"].flatten(), np.ones(L * L))
        ).T
        mask_reshape = mask.reshape((L * L))
        imgs = imgs.reshape((-1, L * L))

        # Fit a ramping background and apply to images
        coeff = lstsq(ramp_mask, imgs[:, mask_reshape].T)[0]  # RCOPT
        imgs = imgs - (ramp_all @ coeff).T  # RCOPT
        imgs = imgs.reshape((-1, L, L))

    # Apply mask images and calculate mean and std values of background
    imgs_masked = imgs * mask
    denominator = np.sum(mask)
    first_moment = np.sum(imgs_masked, axis=(1, 2)) / denominator
    second_moment = np.sum(imgs_masked ** 2, axis=(1, 2)) / denominator
    mean = first_moment.reshape(-1, 1, 1)
    variance = second_moment.reshape(-1, 1, 1) - mean ** 2
    std = np.sqrt(variance)

    return (imgs - mean) / std


class Image:
    def __init__(self, data, dtype=None):
        """
        A stack of one or more images.

        This is a wrapper of numpy.ndarray which provides methods
        for common processing tasks.

        :param data: Numpy array containing image data with shape `(n_images, res, res)`.
        :param dtype: Optionally cast `data` to this dtype. Defaults to `data.dtype`.
        :return: Image instance storing `data`.
        """

        assert isinstance(
            data, np.ndarray
        ), "Image should be instantiated with an ndarray"

        if data.ndim == 2:
            data = data[np.newaxis, :, :]

        if dtype is None:
            self.dtype = data.dtype
        else:
            self.dtype = np.dtype(dtype)

        self.data = data.astype(self.dtype, copy=False)
        self.ndim = self.data.ndim
        self.shape = self.data.shape
        self.n_images = self.shape[0]
        self.res = self.shape[1]

        ensure(data.shape[1] == data.shape[2], "Only square ndarrays are supported.")

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __add__(self, other):
        if isinstance(other, Image):
            other = other.data

        return Image(self.data + other)

    def __sub__(self, other):
        if isinstance(other, Image):
            other = other.data

        return Image(self.data - other)

    def __mul__(self, other):
        if isinstance(other, Image):
            other = other.data

        return Image(self.data * other)

    def __neg__(self):
        return Image(-self.data)

    def sqrt(self):
        return Image(np.sqrt(self.data))

    def flip_axes(self):
        return Image(np.transpose(self.data, (0, 2, 1)))

    def __repr__(self):
        return f"{self.n_images} images of size {self.res}x{self.res}"

    def asnumpy(self):
        return self.data

    def copy(self):
        return Image(self.data.copy())

    def shift(self, shifts):
        """
        Translate image by shifts. This method returns a new Image.

        :param shifts: An array of size n-by-2 specifying the shifts in pixels.
            Alternatively, it can be a column vector of length 2, in which case
            the same shifts is applied to each image.
        :return: The Image translated by the shifts, with periodic boundaries.
        """
        if shifts.ndim == 1:
            shifts = shifts[np.newaxis, :]

        return self._im_translate(shifts)

    def downsample(self, ds_res):
        """
        Downsample Image to a specific resolution. This method returns a new Image.

        :param ds_res: int - new resolution, should be <= the current resolution
            of this Image
        :return: The downsampled Image object.
        """
        grid = grid_2d(self.res)
        grid_ds = grid_2d(ds_res)

        im_ds = np.zeros((self.n_images, ds_res, ds_res), dtype=self.dtype)

        # x, y values corresponding to 'grid'. This is what scipy interpolator needs to function.
        res_by_2 = self.res / 2
        x = y = np.ceil(np.arange(-res_by_2, res_by_2)) / res_by_2

        mask = (np.abs(grid["x"]) < ds_res / self.res) & (
            np.abs(grid["y"]) < ds_res / self.res
        )
        im_shifted = fft.centered_ifft2(
            fft.centered_fft2(xp.asarray(self.data)) * xp.asarray(mask)
        )
        im = np.real(xp.asnumpy(im_shifted))

        for s in range(im_ds.shape[0]):
            interpolator = RegularGridInterpolator(
                (x, y), im[s], bounds_error=False, fill_value=0
            )
            im_ds[s] = interpolator(np.dstack([grid_ds["x"], grid_ds["y"]]))

        return Image(im_ds)

    def filter(self, filter):
        """
        Apply a `Filter` object to the Image and returns a new Image.

        :param filter: An object of type `Filter`.
        :return: A new filtered `Image` object.
        """
        filter_values = filter.evaluate_grid(self.res)

        im_f = xp.asnumpy(fft.centered_fft2(xp.asarray(self.data)))

        if im_f.ndim > filter_values.ndim:
            im_f *= filter_values
        else:
            im_f = filter_values * im_f
        im = xp.asnumpy(fft.centered_ifft2(xp.asarray(im_f)))
        im = np.real(im)

        return Image(im)

    def rotate(self):
        raise NotImplementedError

    def save(self, mrcs_filepath, overwrite=False):
        with mrcfile.new(mrcs_filepath, overwrite=overwrite) as mrc:
            # original input format (the image index first)
            mrc.set_data(self.data.astype(np.float32))

    def _im_translate(self, shifts):
        """
        Translate image by shifts
        :param im: An array of size n-by-L-by-L containing images to be translated.
        :param shifts: An array of size n-by-2 specifying the shifts in pixels.
            Alternatively, it can be a row vector of length 2, in which case the same shifts is applied to each image.
        :return: The images translated by the shifts, with periodic boundaries.

        TODO: This implementation is slower than _im_translate2
        """
        im = self.data

        if shifts.ndim == 1:
            shifts = shifts[np.newaxis, :]
        n_shifts = shifts.shape[0]

        ensure(shifts.shape[-1] == 2, "shifts must be nx2")

        ensure(
            n_shifts == 1 or n_shifts == self.n_images,
            "number of shifts must be 1 or match the number of images",
        )
        # Cast shifts to this instance's internal dtype
        shifts = shifts.astype(self.dtype)

        L = self.res
        im_f = xp.asnumpy(fft.fft2(xp.asarray(im)))
        grid_shifted = fft.ifftshift(
            xp.asarray(np.ceil(np.arange(-L / 2, L / 2, dtype=self.dtype)))
        )
        grid_1d = xp.asnumpy(grid_shifted) * 2 * np.pi / L
        om_x, om_y = np.meshgrid(grid_1d, grid_1d, indexing="ij")

        phase_shifts_x = -shifts[:, 0].reshape((n_shifts, 1, 1))
        phase_shifts_y = -shifts[:, 1].reshape((n_shifts, 1, 1))

        phase_shifts = (
            om_x[np.newaxis, :, :] * phase_shifts_x
            + om_y[np.newaxis, :, :] * phase_shifts_y
        )
        mult_f = np.exp(-1j * phase_shifts)
        im_translated_f = im_f * mult_f
        im_translated = xp.asnumpy(fft.ifft2(xp.asarray(im_translated_f)))
        im_translated = np.real(im_translated)

        return Image(im_translated)

    def norm(self):
        return anorm(self.data)

    @property
    def size(self):
        # probably not needed, transition
        return np.size(self.data)

    def backproject(self, rot_matrices):
        """
        Backproject images along rotation
        :param im: An Image (stack) to backproject.
        :param rot_matrices: An n-by-3-by-3 array of rotation matrices \
        corresponding to viewing directions.

        :return: Volume instance corresonding to the backprojected images.
        """

        L = self.res

        ensure(
            self.n_images == rot_matrices.shape[0],
            "Number of rotation matrices must match the number of images",
        )

        # TODO: rotated_grids might as well give us correctly shaped array in the first place
        pts_rot = aspire.volume.rotated_grids(L, rot_matrices)
        pts_rot = np.moveaxis(pts_rot, 1, 2)
        pts_rot = m_reshape(pts_rot, (3, -1))

        im_f = xp.asnumpy(fft.centered_fft2(xp.asarray(self.data))) / (L ** 2)
        if L % 2 == 0:
            im_f[:, 0, :] = 0
            im_f[:, :, 0] = 0

        im_f = im_f.flatten()

        vol = anufft(im_f, pts_rot, (L, L, L), real=True) / L

        return aspire.volume.Volume(vol)

    def show(self, columns=5, figsize=(20, 10)):
        """
        Plotting Utility Function.

        :param columns: Number of columns in a row of plots.
        :param figsize: Figure size in inches, consult `matplotlib.figure`.
        """

        plt.figure(figsize=figsize)
        for i, im in enumerate(self):
            plt.subplot(self.n_images // columns + 1, columns, i + 1)
            plt.imshow(im, cmap="gray")
        plt.show()


class CartesianImage(Image):
    def expand(self, basis):
        return BasisImage(basis)


class PolarImage(Image):
    def expand(self, basis):
        return BasisImage(basis)


class BispecImage(Image):
    def expand(self, basis):
        return BasisImage(basis)


class BasisImage(Image):
    def __init__(self, basis):
        self.basis = basis

    def evaluate(self):
        return CartesianImage()


class FBBasisImage(BasisImage):
    pass
