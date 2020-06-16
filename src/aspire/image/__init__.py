import mrcfile
import numpy as np
from scipy.fftpack import fft2, ifft2, ifftshift
from scipy.interpolate import RegularGridInterpolator

from aspire.utils import ensure
from aspire.utils.coor_trans import grid_2d
from aspire.utils.fft import centered_fft2, centered_ifft2


# TODO: The implementation of these functions should move directly inside the appropriate Image methods that call them.
def _im_translate(im, shifts):
    """
    Translate image by shifts
    :param im: An array of size L-by-L-by-n containing images to be translated.
    :param shifts: An array of size n-by-2 specifying the shifts in pixels.
        Alternatively, it can be a column vector of length 2, in which case the same shifts is applied to each image.
    :return: The images translated by the shifts, with periodic boundaries.

    TODO: This implementation is slower than _im_translate2
    """
    n_im = im.shape[-1]
    n_shifts = shifts.shape[0]

    ensure(shifts.shape[-1] == 2, "shifts must be nx2")
    ensure(n_shifts == 1 or n_shifts == n_im, "number of shifts must be 1 or match the number of images")
    ensure(im.shape[0] == im.shape[1], "images must be square")

    L = im.shape[0]
    im_f = fft2(im, axes=(0, 1))
    grid_1d = ifftshift(np.ceil(np.arange(-L/2, L/2))) * 2 * np.pi / L
    om_x, om_y = np.meshgrid(grid_1d, grid_1d, indexing='ij')

    phase_shifts_x = np.broadcast_to(-shifts[:, 0], (L, L, n_shifts))
    phase_shifts_y = np.broadcast_to(-shifts[:, 1], (L, L, n_shifts))
    phase_shifts = (om_x[:, :, np.newaxis] * phase_shifts_x) + (om_y[:, :, np.newaxis] * phase_shifts_y)

    mult_f = np.exp(-1j * phase_shifts)
    im_translated_f = im_f * mult_f
    im_translated = ifft2(im_translated_f, axes=(0, 1))
    im_translated = np.real(im_translated)

    return im_translated


def _im_translate2(im, shifts):
    """
    Translate image by shifts
    :param im: An array of size L-by-L-by-n containing images to be translated.
    :param shifts: An array of size 2-by-n specifying the shifts in pixels.
        Alternatively, it can be a column vector of length 2, in which case the same shifts is applied to each image.
    :return: The images translated by the shifts

    TODO: This implementation has been moved here from aspire.aspire.abinitio and is faster than _im_translate.
    """
    n_im = im.shape[2]
    n_shifts = shifts.shape[1]

    if shifts.shape[0] != 2:
        raise ValueError('Input `shifts` must be of size 2-by-n')

    if n_shifts != 1 and n_shifts != n_im:
        raise ValueError('The number of shifts must be 1 or match the number of images')

    if im.shape[0] != im.shape[1]:
        raise ValueError('Images must be square')

    resolution = im.shape[1]
    grid = np.fft.ifftshift(np.ceil(np.arange(-resolution / 2, resolution / 2)))
    om_y, om_x = np.meshgrid(grid, grid)
    phase_shifts = np.einsum('ij, k -> ijk', om_x, shifts[0]) + np.einsum('ij, k -> ijk', om_y, shifts[1])
    phase_shifts /= resolution

    mult_f = np.exp(-2 * np.pi * 1j * phase_shifts)
    im_f = np.fft.fft2(im, axes=(0, 1))
    im_translated_f = im_f * mult_f
    im_translated = np.real(np.fft.ifft2(im_translated_f, axes=(0, 1)))
    return im_translated


class Image:
    def __init__(self, data):
        ensure(data.shape[0] == data.shape[1], 'Only square ndarrays are supported.')
        if data.ndim == 2:
            data = data[:, :, np.newaxis]

        self.data = data
        self.dtype = self.data.dtype
        self.shape = self.data.shape
        self.n_images = self.shape[-1]
        self.res = self.shape[0]

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __add__(self, other):
        return Image(self.data + other.data)

    def __sub__(self, other):
        return Image(self.data - other.data)

    def __mul__(self, amplitudes):
        return Image(self.data * amplitudes)

    def __repr__(self):
        return f'{self.n_images} images of size {self.res}x{self.res}'

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

        im_translated = _im_translate(self.data, shifts)
        return Image(im_translated)

    def downsample(self, ds_res):
        """
        Downsample Image to a specific resolution. This method returns a new Image.

        :param ds_res: int - new resolution, should be <= the current resolution
            of this Image
        :return: The downsampled Image object.
        """
        grid = grid_2d(self.res)
        grid_ds = grid_2d(ds_res)

        im_ds = np.zeros((ds_res, ds_res, self.n_images)).astype(self.dtype)

        # x, y values corresponding to 'grid'. This is what scipy interpolator needs to function.
        res_by_2 = self.res / 2
        x = y = np.ceil(np.arange(-res_by_2, res_by_2)) / res_by_2

        mask = (np.abs(grid['x']) < ds_res / self.res) & (np.abs(grid['y']) < ds_res / self.res)
        im = np.real(centered_ifft2(centered_fft2(self.data) * np.expand_dims(mask, 2)))

        for s in range(im_ds.shape[-1]):
            interpolator = RegularGridInterpolator(
                (x, y),
                im[:, :, s],
                bounds_error=False,
                fill_value=0
            )
            im_ds[:, :, s] = interpolator(np.dstack([grid_ds['x'], grid_ds['y']]))

        return Image(im_ds)

    def filter(self, filter):
        """
        Apply a `Filter` object to the Image and returns a new Image.

        :param filter: An object of type `Filter`.
        :return: A new filtered `Image` object.
        """
        filter_values = filter.evaluate_grid(self.res)

        im_f = centered_fft2(self.data)
        if im_f.ndim > filter_values.ndim:
            im_f = np.expand_dims(filter_values, 2) * im_f
        else:
            im_f = filter_values * im_f
        im = centered_ifft2(im_f)
        im = np.real(im)

        return Image(im)

    def phase_flip(self, filter):
        """
        Perform phase flip to images by applying the sign of a `Filter` object.

        :param filter: An object of type `Filter`.
        :return: A new filtered `Image` object.
        """
        sign_values = np.sign(filter.evaluate_grid(self.res))

        im_f = centered_fft2(self.data)
        if im_f.ndim > sign_values.ndim:
            im_f = np.expand_dims(sign_values, 2) * im_f
        else:
            im_f = sign_values * im_f
        im = centered_ifft2(im_f)
        im = np.real(im)

        return Image(im)

    def rotate(self):
        raise NotImplementedError

    def save(self, mrcs_filepath, overwrite=False):
        with mrcfile.new(mrcs_filepath, overwrite=overwrite) as mrc:
            # change back to the original input format (the image index first)
            # for the consistency with reading
            mrc.set_data(np.swapaxes(self.data.astype('float32'), 0, 2))


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
