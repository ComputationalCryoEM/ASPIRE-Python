import logging
import os
from warnings import catch_warnings, filterwarnings, simplefilter, warn

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
from PIL import Image as PILImage
from scipy.linalg import lstsq

import aspire.sinogram
import aspire.volume
from aspire.nufft import anufft, nufft
from aspire.numeric import fft, xp
from aspire.utils import FourierRingCorrelation, anorm, crop_pad_2d, grid_2d
from aspire.volume import SymmetryGroup

logger = logging.getLogger(__name__)


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
    if imgs.ndim > 3:
        raise NotImplementedError(
            "`normalize_bg` is currently limited to 1D image stacks."
        )

    L = imgs.shape[-1]
    input_dtype = imgs.dtype
    grid = grid_2d(L, indexing="yx", dtype=input_dtype)
    mask = grid["r"] > bg_radius

    if do_ramp:
        # Create matrices and reshape the background mask
        # for fitting a ramping background
        ramp_mask = np.vstack(
            (
                grid["x"][mask].flatten(),
                grid["y"][mask].flatten(),
                np.ones(grid["y"][mask].flatten().size, dtype=input_dtype),
            )
        ).T
        ramp_all = np.vstack(
            (
                grid["x"].flatten(),
                grid["y"].flatten(),
                np.ones(L * L, dtype=input_dtype),
            )
        ).T
        mask_reshape = mask.reshape((L * L))
        imgs = imgs.reshape((-1, L * L))

        # Fit a ramping background and apply to images
        coef = lstsq(ramp_mask, imgs[:, mask_reshape].T)[0]  # RCOPT
        imgs = imgs - (ramp_all @ coef).T  # RCOPT
        imgs = imgs.reshape((-1, L, L))

    # Apply mask images and calculate mean and std values of background
    imgs_masked = imgs * mask
    denominator = np.sum(mask)
    first_moment = np.sum(imgs_masked, axis=(1, 2)) / denominator
    second_moment = np.sum(imgs_masked**2, axis=(1, 2)) / denominator
    mean = first_moment.reshape(-1, 1, 1)
    variance = second_moment.reshape(-1, 1, 1) - mean**2
    std = np.sqrt(variance)

    return (imgs - mean) / std


def load_mrc(filepath):
    """
    Load raw data from `.mrc` into an array.

    :param filepath: File path (string).
    :return: (numpy array of image data, pixel_size)
    """

    # mrcfile tends to yield many warnings about EMPIAR datasets being corrupt
    # These warnings generally seem benign, and the message could be clearer
    # The following code handles the warnings via ASPIRE's logger
    with catch_warnings(record=True) as ws:
        # Cause all warnings to always be triggered in this context
        simplefilter("always")

        with mrcfile.open(filepath, mode="r", permissive=True) as mrc:
            im = mrc.data
            pixel_size = Image._vx_array_to_size(mrc.voxel_size)

        # Log each mrcfile warning to debug log, noting the associated file
        for w in ws:
            logger.debug(
                "In `Image.load` mrcfile.open reports corruption for"
                f" {filepath} warning: {w.message}"
            )

        # Log a single warning to user
        # Give context and note assocated filepath
        if len(ws) > 0:
            logger.warning(
                f"Image.load of {filepath} reporting {len(ws)} corruptions."
                " Most likely this is a problem with the header contents."
                " Details written to debug log."
                f" Will attempt to continue processing {filepath}"
            )

    return im, pixel_size


def load_tiff(filepath):
    """
    Load raw data from `.tiff` into an array.

    Note, TIFF does not natively provide equivalent to pixel/voxel_size,
    so users of TIFF files may need to manually assign `pixel_size` to
    `Image` instances when required. Defaults to `pixel_size=None`.

    :param filepath: File path (string).
    :return: (numpy array of image data, pixel_size=None)
    """
    # Use PIL to open `filepath`
    img = PILImage.open(filepath)

    # Future todo, extract `voxel_size` if available in TIFF tags (custom tag?)
    # For now, default to `None`.
    pixel_size = None

    # Cast image data as numpy array
    return np.array(img), pixel_size


class Image:
    # Map file extensions to their respective readers
    extensions = {
        ".mrc": load_mrc,
        ".tif": load_tiff,
        ".tiff": load_tiff,
    }

    def __init__(self, data, pixel_size=None, dtype=None):
        """
        A stack of one or more images.

        This is a wrapper of numpy.ndarray which provides methods
        for common processing tasks.

        The stack can be multidimensional with `n_images` equal
        to the product of the stack dimensions.  Singletons will be
        expanded into a stack with one entry.

        The last two axes represent the image size,
        and are checked to be square.

        :param data: Numpy array containing image data with shape
            `(..., resolution, resolution)`.
        :param pixel_size: Optional pixel size in angstroms.
            When provided will be saved with `mrc` metadata.
            Default of `None` will not write to file,
            but will be considered unit pixels (1) for FSC.
        :param dtype: Optionally cast `data` to this dtype.
            Defaults to `data.dtype`.

        :return: Image instance holding `data`.
        """

        if not isinstance(data, np.ndarray):
            raise ValueError("Image should be instantiated with an ndarray")

        if data.ndim < 2:
            raise ValueError(
                "Image data should be ndarray with shape (N1...)xLxL or LxL."
            )
        elif data.ndim == 2:
            data = np.expand_dims(data, axis=0)

        if dtype is None:
            self.dtype = data.dtype
        else:
            self.dtype = np.dtype(dtype)

        if not data.shape[-1] == data.shape[-2]:
            raise ValueError("Only square ndarrays are supported.")

        self._data = data.astype(self.dtype, copy=False)
        self.ndim = self._data.ndim
        self.shape = self._data.shape
        self.stack_ndim = self._data.ndim - 2
        self.stack_shape = self._data.shape[:-2]
        self.n_images = np.prod(self.stack_shape)
        self.resolution = self._data.shape[-1]
        self.pixel_size = None
        if pixel_size is not None:
            self.pixel_size = float(pixel_size)

        # Numpy interop
        # https://numpy.org/devdocs/user/basics.interoperability.html#the-array-interface-protocol
        self.__array_interface__ = self._data.__array_interface__
        self.__array__ = self._data

    def project(self, angles):
        """
        Computes the Radon Transform on an Image Stack using
        Non-Uniform Fast Fourier Transforms. This method projects the
        Image stack along different angles and returns the Radon
        Transform.

        :param angles: A 1-D Numpy Array of angles in Radians.
            This is used to compute the Radon Transform at different angles.
        :return: Radon transform of the Image Stack.
        :rtype: Ndarray (stack size, number of angles, image resolution)
        """
        # number of points to sample on radial line in polar grid
        n_points = self.resolution
        original_stack = self.stack_shape

        # 2-D grid
        radial_idx = fft.rfftfreq(n_points) * xp.pi * 2
        n_real_points = len(radial_idx)
        n_angles = len(angles)
        angles = xp.asarray(angles)

        pts = xp.empty((2, n_angles, n_real_points), dtype=self.dtype)
        pts[0] = radial_idx[xp.newaxis, :] * xp.sin(angles)[:, xp.newaxis]
        pts[1] = radial_idx[xp.newaxis, :] * xp.cos(angles)[:, xp.newaxis]
        pts = pts.reshape(2, n_real_points * n_angles)

        # compute the polar nufft (NUFFT)
        image_ft = nufft(xp.asarray(self.stack_reshape(-1)._data), pts).reshape(
            self.n_images, n_angles, n_real_points
        )

        # Radon transform, output: (stack size, angles, points)
        image_rt = fft.fftshift(fft.irfft(image_ft, n=n_points, axis=-1), axes=-1)
        image_rt = image_rt.reshape(*original_stack, n_angles, n_points)
        return aspire.sinogram.Sinogram(xp.asnumpy(image_rt))

    @property
    def res(self):
        warn(
            "`Image.res` will be deprecated in favor or Image.resolution in an upcoming release.",
            DeprecationWarning,
        )
        return self.resolution

    def _check_key_dims(self, key):
        if isinstance(key, tuple) and (len(key) > self._data.ndim):
            raise ValueError(
                f"Image stack_dim is {self.stack_ndim}, slice length must be =< {self.ndim}"
            )

    def __getitem__(self, key):
        self._check_key_dims(key)
        return self.__class__(self._data[key], pixel_size=self.pixel_size)

    def __setitem__(self, key, value):
        self._check_key_dims(key)
        self._data[key] = value

    def stack_reshape(self, *args):
        """
        Reshape the stack axis.

        :*args: Integer(s) or tuple describing the intended shape.

        :returns: Image instance
        """

        # If we're passed a tuple, use that
        if len(args) == 1 and isinstance(args[0], tuple):
            shape = args[0]
        else:
            # Otherwise use the variadic args
            shape = args

        # Sanity check the size
        if shape != (-1,) and np.prod(shape) != self.n_images:
            raise ValueError(
                f"Number of images {self.n_images} cannot be reshaped to {shape}."
            )

        return self.__class__(
            self._data.reshape(*shape, *self._data.shape[-2:]),
            pixel_size=self.pixel_size,
        )

    def __add__(self, other):
        if isinstance(other, Image):
            other = other._data

        return self.__class__(self._data + other, pixel_size=self.pixel_size)

    def __sub__(self, other):
        if isinstance(other, Image):
            other = other._data

        return self.__class__(self._data - other, pixel_size=self.pixel_size)

    def __mul__(self, other):
        if isinstance(other, Image):
            other = other._data

        return self.__class__(self._data * other, pixel_size=self.pixel_size)

    def __neg__(self):
        return self.__class__(-self._data, pixel_size=self.pixel_size)

    def sqrt(self):
        return self.__class__(np.sqrt(self._data), pixel_size=self.pixel_size)

    @property
    def T(self):
        """
        Abbreviation for transpose.

        :return: Image instance.
        """
        return self.transpose()

    def transpose(self):
        """
        Returns a new Image instance with image data axes transposed.

        :return: Image instance.
        """
        original_stack_shape = self.stack_shape

        im = self.stack_reshape(-1)
        imt = np.transpose(im._data, (0, -1, -2))

        return self.__class__(imt, pixel_size=self.pixel_size).stack_reshape(
            original_stack_shape
        )

    def flip(self, axis=-2):
        """
        Flip image stack data along axis using numpy.flip().

        :param axis: Optionally specify axis as integer or tuple.
            Defaults to axis=-2.

        :return: Image instance.
        """
        # Convert integer to tuple, so we can always loop.
        if isinstance(axis, int):
            axis = (axis,)

        # Check we are not attempting to flip any stack axis.
        for ax in axis:
            ax = ax % self.ndim  # modulo [0, ndim)
            if ax < self.stack_ndim:
                raise ValueError(
                    f"Cannot flip axis {ax}: stack axis. Did you mean {ax-3}?"
                )

        return self.__class__(np.flip(self._data, axis), pixel_size=self.pixel_size)

    def __repr__(self):
        px_msg = "."
        if self.pixel_size is not None:
            px_msg = f" with pixel_size={self.pixel_size} angstroms."

        msg = f"{self.n_images} {self.dtype} images arranged as a {self.stack_shape} stack"
        msg += f" each of size {self.resolution}x{self.resolution}{px_msg}"
        return msg

    def asnumpy(self):
        """
        Return image data as a (<stack>, resolution, resolution)
        read-only array view.

        :return: read-only ndarray view
        """

        view = self._data.view()
        view.flags.writeable = False
        return view

    def copy(self):
        return self.__class__(self._data.copy(), pixel_size=self.pixel_size)

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

        n_shifts = shifts.shape[0]

        if not shifts.shape[1] == 2:
            raise ValueError("Input shifts must be of shape (n_images, 2) or (1, 2).")
        if not n_shifts == 1 and not n_shifts == self.n_images:
            raise ValueError(
                "The number of shifts must be 1 or equal to self.n_images."
            )

        return self._im_translate(shifts)

    def downsample(self, ds_res, zero_nyquist=True):
        """
        Downsample Image to a specific resolution. This method returns a new Image.

        :param ds_res: int - new resolution, should be <= the current resolution
            of this Image
        :param zero_nyquist: Option to keep or remove Nyquist frequency for even resolution.
            Defaults to zero_nyquist=True, removing the Nyquist frequency.
        :return: The downsampled Image object.
        """

        original_stack_shape = self.stack_shape
        im = self.stack_reshape(-1)

        # Note image data is intentionally migrated via `xp.asarray`
        # because all of the subsequent calls until `asnumpy` are GPU
        # when xp and fft in `cupy` mode.

        # compute FT with centered 0-frequency
        fx = fft.centered_fft2(xp.asarray(im._data))
        # crop 2D Fourier transform for each image
        crop_fx = crop_pad_2d(fx, ds_res)

        # If downsampled resolution is even, optionally zero out the nyquist frequency.
        if ds_res % 2 == 0 and zero_nyquist is True:
            crop_fx[:, 0, :] = 0
            crop_fx[:, :, 0] = 0

        # take back to real space, discard complex part, and scale
        out = fft.centered_ifft2(crop_fx).real * (ds_res**2 / self.resolution**2)
        out = xp.asnumpy(out)

        # Optionally scale pixel size
        ds_pixel_size = self.pixel_size
        if ds_pixel_size is not None:
            ds_pixel_size *= self.resolution / ds_res

        return self.__class__(out, pixel_size=ds_pixel_size).stack_reshape(
            original_stack_shape
        )

    def filter(self, filter):
        """
        Apply a `Filter` object to the Image and returns a new Image.

        :param filter: An object of type `Filter`.
        :return: A new filtered `Image` object.
        """
        original_stack_shape = self.stack_shape

        im = self.stack_reshape(-1)

        # Note image and filter data is intentionally migrated via
        # `xp.asarray` because all of the subsequent calls until
        # `asnumpy` are GPU when xp and fft in `cupy` mode.
        #
        # Second note, filter dtype may not match image dtype.
        filter_values = xp.asarray(
            filter.evaluate_grid(self.resolution), dtype=self.dtype
        )

        # Convolve
        im_f = fft.centered_fft2(xp.asarray(im._data))
        im_f = filter_values * im_f
        im = fft.centered_ifft2(im_f)

        im = xp.asnumpy(im.real)

        return self.__class__(im, pixel_size=self.pixel_size).stack_reshape(
            original_stack_shape
        )

    def rotate(self):
        raise NotImplementedError

    def save(self, mrcs_filepath, overwrite=False):
        if self.stack_ndim > 1:
            raise NotImplementedError("`save` is currently limited to 1D image stacks.")

        with mrcfile.new(mrcs_filepath, overwrite=overwrite) as mrc:
            # original input format (the image index first)
            mrc.set_data(self._data.astype(np.float32))
            # Note assigning voxel_size must come after `set_data`
            if self.pixel_size is not None:
                mrc.voxel_size = self.pixel_size

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
        im, pixel_size = Image.extensions[ext](filepath)

        # Attempt casting when user provides dtype
        if dtype is not None:
            im = im.astype(dtype, copy=False)

        # Return as Image instance
        return Image(im, pixel_size=pixel_size)

    def _im_translate(self, shifts):
        """
        Translate image by shifts

        :param im: An array of size n-by-L-by-L containing images to be translated.
        :param shifts: An array of size n-by-2 specifying the shifts in pixels.
            Alternatively, it can be a row vector of length 2, in which case the same shifts is applied to each image.
        :return: The images translated by the shifts, with periodic boundaries.
        """

        # Note original stack shape and flatten stack
        stack_shape = self.stack_shape
        im = self.stack_reshape(-1)._data

        if shifts.ndim == 1:
            shifts = shifts[np.newaxis, :]
        n_shifts = shifts.shape[0]

        assert shifts.shape[-1] == 2, "shifts must be nx2"

        assert (
            n_shifts == 1 or n_shifts == self.n_images
        ), "number of shifts must be 1 or match the number of images"
        # Cast shifts to this instance's internal dtype
        shifts = xp.asarray(shifts, dtype=self.dtype)

        L = self.resolution
        im_f = fft.fft2(xp.asarray(im))
        grid_shifted = fft.ifftshift(
            xp.ceil(xp.arange(-L / 2, L / 2, dtype=self.dtype))
        )
        grid_1d = grid_shifted * 2 * xp.pi / L

        # Grid indexing changed to "xy" to match Relion shift conventions.
        om_x, om_y = xp.meshgrid(grid_1d, grid_1d, indexing="xy")

        phase_shifts_x = -shifts[:, 0].reshape((n_shifts, 1, 1))
        phase_shifts_y = -shifts[:, 1].reshape((n_shifts, 1, 1))

        phase_shifts = (
            om_x[np.newaxis, :, :] * phase_shifts_x
            + om_y[np.newaxis, :, :] * phase_shifts_y
        )
        mult_f = xp.exp(-1j * phase_shifts)
        im_translated_f = im_f * mult_f
        im_translated = fft.ifft2(im_translated_f)
        im_translated = xp.asnumpy(im_translated.real)

        # Reshape to stack shape
        return self.__class__(im_translated, pixel_size=self.pixel_size).stack_reshape(
            stack_shape
        )

    def norm(self):
        return anorm(self._data)

    @property
    def size(self):
        # probably not needed, transition
        return np.size(self._data)

    def backproject(self, rot_matrices, symmetry_group=None, zero_nyquist=True):
        """
        Backproject images along rotations. If a symmetry group is provided, images
        used in back-projection are duplicated (boosted) for symmetric viewing directions.
        Note, it is assumed that a main axis of symmetry aligns with the z-axis.

        :param rot_matrices: An n-by-3-by-3 array of rotation matrices
            corresponding to viewing directions.
        :param symmetry_group: A SymmetryGroup instance or string indicating symmetry, ie. "C3".
            If supplied, uses symmetry to increase number of images used in back-projection.
        :param zero_nyquist: Option to keep or remove Nyquist frequency for even resolution.
            Defaults to zero_nyquist=True, removing the Nyquist frequency.

        :return: Volume instance corresonding to the backprojected images.
        """

        if self.stack_ndim > 1:
            raise NotImplementedError(
                "`Backprojection` is currently limited to 1D image stacks."
            )

        L = self.resolution

        assert (
            self.n_images == rot_matrices.shape[0]
        ), "Number of rotation matrices must match the number of images"

        # Get symmetry rotations from SymmetryGroup.
        symmetry_rots = SymmetryGroup.parse(symmetry_group, dtype=self.dtype).matrices
        if len(symmetry_rots) > 1:
            logger.info(f"Boosting with {len(symmetry_rots)} rotational symmetries.")

        # Compute Fourier transform of images.
        im_f = xp.asnumpy(fft.centered_fft2(xp.asarray(self._data))) / (L**2)

        # If resolution is even, optionally zero out the nyquist frequency.
        if L % 2 == 0 and zero_nyquist is True:
            im_f[:, 0, :] = 0
            im_f[:, :, 0] = 0

        im_f = im_f.flatten()

        # Backproject. Apply boosting by looping over symmetry rotations.
        vol = np.zeros((L, L, L), dtype=self.dtype)
        for sym_rot in symmetry_rots:
            rotations = sym_rot @ rot_matrices

            # TODO: rotated_grids might as well give us correctly shaped array in the first place
            pts_rot = aspire.volume.rotated_grids(L, rotations).astype(
                self.dtype, copy=False
            )
            pts_rot = pts_rot.reshape((3, -1))

            vol += anufft(im_f, pts_rot, (L, L, L), real=True)

        vol /= L

        return aspire.volume.Volume(
            vol, pixel_size=self.pixel_size, symmetry_group=symmetry_group
        )

    def show(self, columns=5, figsize=(20, 10), colorbar=True):
        """
        Plotting Utility Function.

        :param columns: Number of columns in a row of plots.
        :param figsize: Figure size in inches, consult `matplotlib.figure`.
        :param colorbar: Optionally plot colorbar to show scale.
            Defaults to True. Accepts `bool` or `dictionary`,
            where the dictionary is passed to `matplotlib.pyplot.colorbar`.
        """

        if self.stack_ndim > 1:
            raise NotImplementedError("`show` is currently limited to 1D image stacks.")

        # We never need more columns than images.
        columns = min(columns, self.n_images)
        rows = (self.n_images + columns - 1) // columns  # ceiling divide.

        # Create an empty colorbar options dictionary as needed.
        colorbar_opts = colorbar if isinstance(colorbar, dict) else dict()

        # Create a context manager for altering warnings
        with catch_warnings():
            # Filter off specific warning.
            # sphinx-gallery overrides to `agg` backend, but doesn't handle warning.
            filterwarnings(
                "ignore",
                category=UserWarning,
                message="Matplotlib is currently using agg, which is a"
                " non-GUI backend, so cannot show the figure.",
            )

            plt.figure(figsize=figsize)
            for i, im in enumerate(self.asnumpy()):
                plt.subplot(rows, columns, i + 1)
                plt.imshow(im, cmap="gray")
                if colorbar:
                    plt.colorbar(**colorbar_opts)

            plt.show()

    def frc(self, other, cutoff=None, method="fft", plot=False):
        r"""
        Compute the Fourier ring correlation between two images.

        Images are assumed to be well aligned.

        The FRC is defined as:

        .. math::

           c(i) = \frac{ \operatorname{Re}( \sum_i{ \mathcal{F}_1(i) * {\mathcal{F}^{*}_2(i) } } ) }{\
             \sqrt{ \sum_i { | \mathcal{F}_1(i) |^2 } * \sum_i{| \mathcal{F}^{*}_2}(i) |^2 } }

        :param other: `Image` instance to compare.
        :param cutoff: Cutoff value, traditionally `.143`.
            Default `None` implies `cutoff=1` and excludes
            plotting cutoff line.

        :param method: Selects either 'fft' (on cartesian grid),
            or 'nufft' (on polar grid). Defaults to 'fft'.
        :param plot: Optionally plot to screen or file.
            Defaults to `False`.  `True` plots to screen.
            Passing a filepath as a string will attempt to save to file.

        :return: tuple(estimated_resolution,  FRC),
            where `estimated_resolution` is in angstrom
            and FRC is a Numpy array of correlations.
        """

        if not isinstance(other, Image):
            raise TypeError(
                f"`other` image must be an `Image` instance, received {type(other)}"
            )

        frc = FourierRingCorrelation(
            a=self.asnumpy(),
            b=other.asnumpy(),
            pixel_size=self.pixel_size,
            method=method,
        )

        if plot is True:
            frc.plot(cutoff=cutoff)
        elif plot:
            frc.plot(cutoff=cutoff, save_to_file=plot)

        return frc.analyze_correlations(cutoff), frc.correlations

    @staticmethod
    def _vx_array_to_size(vx):
        """
        Utility to convert from several possible `mrcfile.voxel_size`
        representations to a single (float) value or None.
        """

        # Convert from recarray to single values,
        #   checks uniformity.
        if isinstance(vx, np.recarray):
            if vx.x != vx.y:
                raise ValueError(f"Voxel sizes are not uniform: {vx}")
            vx = vx.x

        # Convert `0` to `None`
        if (
            isinstance(vx, int) or isinstance(vx, float) or isinstance(vx, np.ndarray)
        ) and vx == 0:
            vx = None

        # Consistently return a `float` when not None
        if vx is not None:
            vx = float(vx)

        return vx


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
