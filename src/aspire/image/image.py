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
from aspire.utils import (
    FourierRingCorrelation,
    anorm,
    crop_pad_2d,
    grid_2d,
    rename_with_timestamp,
)
from aspire.volume import SymmetryGroup

logger = logging.getLogger(__name__)


def normalize_bg(imgs, bg_radius=1.0, do_ramp=True, shifted=False, ddof=0):
    """
    Normalize backgrounds and apply to a stack of images.

    To recreate legacy MATLAB workflow results, review parameters used in
    `ImageSource.legacy_normalize_background`.

    :param imgs: A stack of images in N-by-L-by-L array
    :param bg_radius: Radius cutoff to be considered as background (in image size)
    :param do_ramp: When it is `True`, fit a ramping background to the data
        and subtract. Namely perform normalization based on values from each image.
        Otherwise, a constant background level from all images is used.
    :param shifted: Optionally shifts 2d grid by 1/2 pixel for even
        resolution to replicate MATLAB.
    :param ddof: Degrees of freedom for standard deviation.
    :return: The modified images
    """
    if imgs.ndim > 3:
        raise NotImplementedError(
            "`normalize_bg` is currently limited to 1D image stacks."
        )
    L = imgs.shape[-1]
    input_dtype = imgs.dtype

    # Generate background mask
    grid_dtype = np.float64  # Use doubles for accuracy and MATLAB repro
    grid = grid_2d(L, shifted=shifted, indexing="yx", dtype=grid_dtype)
    mask = grid["r"] > bg_radius

    if do_ramp:
        # Create matrices and reshape the background mask
        # for fitting a ramping background
        ramp_mask = np.vstack(
            (
                grid["x"][mask].flatten(),
                grid["y"][mask].flatten(),
                np.ones(grid["y"][mask].flatten().size, dtype=grid_dtype),
            )
        ).T
        ramp_all = np.vstack(
            (
                grid["x"].flatten(),
                grid["y"].flatten(),
                np.ones(L * L, dtype=grid_dtype),
            )
        ).T
        mask_reshape = mask.reshape((L * L))
        imgs = imgs.reshape((-1, L * L))

        # Fit a ramping background and apply to images
        coef = lstsq(ramp_mask, imgs[:, mask_reshape].T)[0]  # RCOPT
        imgs = imgs - (ramp_all @ coef).T  # RCOPT
        imgs = imgs.reshape((-1, L, L))

    # Apply mask images and calculate mean and std values of background
    # These should be computed and normalized as doubles
    bg_pixels = imgs[:, mask].astype(np.float64, copy=False)
    mean = np.mean(bg_pixels, axis=1)
    std = np.std(bg_pixels, ddof=ddof, axis=1)
    imgs = (imgs - mean[:, None, None]) / std[:, None, None]

    # Restore input dtype
    return imgs.astype(input_dtype, copy=False)


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
        ".mrcs": load_mrc,
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
            the same shift is applied to each image.
        :return: The Image translated by the shifts, with periodic boundaries.
        """
        if shifts.ndim == 1:
            shifts = shifts[np.newaxis, :]

        n_shifts = shifts.shape[0]

        if not shifts.shape[1] == 2:
            raise ValueError("Input shifts must be of shape (n_images, 2) or (1, 2).")

        if not (n_shifts == 1 or self.n_images == 1 or n_shifts == self.n_images):
            raise ValueError(
                "The number of shifts must be 1 or equal to self.n_images."
            )

        return self._im_translate(shifts)

    def legacy_whiten(self, psd, delta):
        """
        Apply the legacy MATLAB whitening transformation.

        Note that this legacy method will compute the convolution in
        (complex)double precision regardless of this instances
        `dtype`.  However, the resulting image stack will be cast to
        the instance `dtype`.

        :param psd: PSD as computed by `LegacyNoiseEstimator`.
            `psd` in this case is shape (2 * self.src.L - 1, 2 * self.src.L - 1).
        :param delta: Threshold used to determine which frequencies to whiten
            and which to set to zero. By default all `sqrt(psd)` values
            less than `delta` are zeroed out in the whitening filter.
        """
        n = self.n_images
        L = self.resolution
        L_half = L // 2
        K = psd.shape[-1]
        k = int(np.ceil(K / 2))

        # Check PSD
        shp = (2 * L - 1, 2 * L - 1)
        if psd.shape != shp:
            raise RuntimeError(f"Incorrect PSD shape {psd.shape}, expectect {shp}.")

        # Create result array
        res = np.empty((n, L, L), dtype=self.dtype)

        # The whitening filter is the sqrt of the power spectrum of the noise, normalized to unit energy.
        psd = xp.asarray(psd, dtype=np.float64)
        fltr = xp.sqrt(psd)
        fltr = fltr / xp.linalg.norm(fltr)

        # Error checking
        if (err := xp.linalg.norm(fltr.imag)) > 10 * delta:
            raise RuntimeError(
                f"Whitening filter has non trivial imaginary component {err}."
            )
        err_ud = xp.linalg.norm(fltr - xp.flipud(fltr))
        err_lr = xp.linalg.norm(fltr - xp.fliplr(fltr))
        if (err_ud > 10 * delta) or (err_lr > 10 * delta):
            raise RuntimeError(
                f"Whitening filter has non trivial symmetry lr {err_lr}, ud {err_ud}."
            )

        # Enforce symmetry
        fltr = (fltr + xp.flipud(fltr)) / 2
        fltr = (fltr + xp.fliplr(fltr)) / 2

        # The filter may have very small values or even zeros.
        # We don't want to process these, so make a list of all large entries.
        nzidx = fltr > 100 * delta
        fltr_nz = fltr[nzidx]

        padded_proj = xp.zeros((K, K), dtype=np.float64)
        filtered_fpadded_proj = xp.zeros((K, K), dtype=np.complex128)

        # Precompute the slices
        if L % 2 == 1:
            slc = slice(k - L_half - 1, k + L_half)
        else:
            slc = slice(k - L_half - 1, k + L_half - 1)

        # Note these computations should be in double precision
        for i, proj in enumerate(self.asnumpy()):

            # Zero pad the image to twice the size
            padded_proj[slc, slc] = xp.asarray(proj, dtype=np.float64)

            # Take the Fourier Transform of the padded image.
            fpadded_proj = fft.centered_fft2(padded_proj)

            # Divide the image by the whitening filter but only in
            # places where the filter is large.  In frequencies where
            # the filter is tiny we cannot whiten so we just use
            # zeros.
            filtered_fpadded_proj[nzidx] = fpadded_proj[nzidx] / fltr_nz
            # `filtered_proj` is still padded and complex. Masked and cast below.
            filtered_proj = fft.centered_ifft2(filtered_fpadded_proj)

            # The resulting image should be real.
            if (
                xp.linalg.norm(filtered_proj.imag) / xp.linalg.norm(filtered_proj)
                > 1e-13
            ):
                raise RuntimeError("Whitened image has strong imaginary component.")

            filtered_proj = filtered_proj[slc, slc].real

            # Assign the resulting image, cast if required.
            res[i] = xp.asnumpy(filtered_proj).astype(res.dtype, copy=False)

        return Image(res)

    @staticmethod
    def _downsample(data, ds_res, zero_nyquist=True, centered_fft=True):
        """
        Downsample Image data to a specific resolution.

        :param data: Numpy array of Image data, shape (n_imgs, resolution, resolution).
        :param ds_res: int - new resolution, should be <= the current resolution
            of this Image
        :param zero_nyquist: Option to keep or remove Nyquist frequency for even
            resolution (boolean). Defaults to zero_nyquist=True, removing the Nyquist frequency.
        :param centered_fft: Default of True uses `centered_fft` to
            maintain ASPIRE-Python centering conventions.

        :return: NumPy array of downsampled Image data.
        """

        # Note image data is intentionally migrated via `xp.asarray`
        # because all of the subsequent calls until `asnumpy` are GPU
        # when xp and fft in `cupy` mode.
        resolution = data.shape[-1]

        if centered_fft:
            # compute FT with centered 0-frequency
            fx = fft.centered_fft2(xp.asarray(data))
        else:
            fx = fft.fftshift(fft.fft2(xp.asarray(data)))

        # crop 2D Fourier transform for each image
        crop_fx = crop_pad_2d(fx, ds_res)

        # If downsampled resolution is even, optionally zero out the nyquist frequency.
        if ds_res % 2 == 0 and zero_nyquist:
            crop_fx[:, 0, :] = 0
            crop_fx[:, :, 0] = 0

        # take back to real space, discard complex part, and scale
        if centered_fft:
            out = fft.centered_ifft2(crop_fx)
        else:
            out = fft.ifft2(fft.ifftshift(crop_fx))

        # The parenths are required because dtype casting semantics
        # differs between Numpy 1, 2, and CuPy.
        # At time of writing CuPy is consistent with Numpy1.
        # The additional parenths yield consistent out.dtype.
        # See #1298 for relevant debugger output.
        out = xp.asnumpy(out.real * (ds_res**2 / resolution**2))

        return out

    def downsample(self, ds_res, zero_nyquist=True, centered_fft=True):
        """
        Downsample Image to a specific resolution. This method returns a new Image.

        :param ds_res: int - new resolution, should be <= the current resolution
            of this Image
        :param zero_nyquist: Option to keep or remove Nyquist frequency for even
            resolution (boolean). Defaults to zero_nyquist=True, removing the Nyquist frequency.
        :param centered_fft: Default of True uses `centered_fft` to
            maintain ASPIRE-Python centering conventions.
        :return: The downsampled Image object.
        """
        original_stack_shape = self.stack_shape
        data = self.stack_reshape(-1).asnumpy()

        ims_ds = self._downsample(
            data, ds_res, zero_nyquist=zero_nyquist, centered_fft=centered_fft
        )

        # Optionally scale pixel size
        ds_pixel_size = self.pixel_size
        if ds_pixel_size is not None:
            ds_pixel_size *= self.resolution / ds_res

        return self.__class__(ims_ds, pixel_size=ds_pixel_size).stack_reshape(
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
        # Second note, filter and grid dtype may not match image dtype,
        # upcast both here for most accurate convolution.
        filter_values = xp.asarray(
            filter.evaluate_grid(
                self.resolution, dtype=np.float64, pixel_size=self.pixel_size
            ),
            dtype=np.float64,
        )

        # Convolve
        _im = xp.asarray(im._data, dtype=np.float64)
        im_f = fft.centered_fft2(_im)
        im_f = filter_values * im_f
        im = fft.centered_ifft2(im_f)

        im = xp.asnumpy(im.real).astype(
            self.dtype, copy=False
        )  # restore to original dtype

        return self.__class__(im, pixel_size=self.pixel_size).stack_reshape(
            original_stack_shape
        )

    def rotate(self):
        raise NotImplementedError

    def save(self, mrcs_filepath, overwrite=None):
        """
        Save Image to disk as mrcs file

        :param filename: Filepath where Image will be saved.
        :param overwrite: Options to control overwrite behavior (default is None):
            - True: Overwrites the existing file if it exists.
            - False: Raises an error if the file exists.
            - None: Renames the old file by appending a time/date stamp.
        """
        if self.stack_ndim > 1:
            raise NotImplementedError("`save` is currently limited to 1D image stacks.")

        data = self._data.astype(np.float32)
        if self.n_images == 1:
            data = data[0]

        if overwrite is None and os.path.exists(mrcs_filepath):
            # If the file exists, append a timestamp to the old file and rename it
            _ = rename_with_timestamp(mrcs_filepath)
        elif overwrite is None:
            overwrite = False

        with mrcfile.new(mrcs_filepath, overwrite=overwrite) as mrc:
            # original input format (the image index first)
            mrc.set_data(data)
            # Note assigning voxel_size must come after `set_data`
            if self.pixel_size is not None:
                mrc.voxel_size = self.pixel_size

    @staticmethod
    def _load_raw(filepath, dtype=None):
        """
        Load raw data from supported files.

        Currently MRC and TIFF are supported.

        :param filepath: File path (string).
        :param dtype: Optionally force cast to `dtype`.
             Default dtype is inferred from the file contents.
        :returns:
            - numpy array of image data.
            - pixel size
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

        return im, pixel_size

    @staticmethod
    def load(filepath, dtype=None):
        """
        Load raw data from supported files.

        Currently MRC and TIFF are supported.

        :param filepath: File path (string).
        :param dtype: Optionally force cast to `dtype`.
             Default dtype is inferred from the file contents.
        :return: Image instance
        """
        # Load raw data from filepath with pixel size
        im, pixel_size = Image._load_raw(filepath, dtype=dtype)

        # Return as Image instance
        return Image(im, pixel_size=pixel_size)

    def _im_translate(self, shifts):
        """
        Translate image by `shifts`.

        Note broadcasting special case
        Image shape (n,L,L) x shifts shape (n,2) -> (n,L,L) shifted images
        Image shape (1,L,L) x shifts shape (n,2) -> (n,L,L) shifted images

        :param im: An array of size m-by-L-by-L containing images to be translated.
            m may be 1 or n.
        :param shifts: An array of size n-by-2 specifying the shifts in pixels.
            Alternatively, it can be a row vector of length 2, in which case the same shifts is applied to each image.
        :return: The images translated by the shifts, with periodic boundaries.
        """

        if shifts.ndim == 1:
            shifts = shifts[np.newaxis, :]
        n_shifts = shifts.shape[0]

        assert shifts.shape[-1] == 2, "shifts must be nx2"

        # Note original stack shape and flatten stack
        stack_shape = self.stack_shape
        if self.n_images == 1 and n_shifts > 1:
            # Handle the shift broadcast special case
            stack_shape = n_shifts
        im = self.stack_reshape(-1)._data

        assert (
            n_shifts == 1 or self.n_images == 1 or n_shifts == self.n_images
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
        symmetry_rots = SymmetryGroup.parse(symmetry_group).matrices.astype(
            self.dtype, copy=False
        )
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
                logger.warning(f"Voxel sizes are not uniform: {vx}")
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
