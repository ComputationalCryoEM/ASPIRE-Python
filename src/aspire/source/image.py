import copy
import functools
import logging
import os.path
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterable

import mrcfile
import numpy as np

from aspire.abinitio import CLOrient3D, CLSyncVoting
from aspire.image import Image, normalize_bg
from aspire.image.xform import (
    Downsample,
    FilterXform,
    IndexedXform,
    LambdaXform,
    Multiply,
    Pipeline,
)
from aspire.noise import NoiseEstimator, WhiteNoiseEstimator
from aspire.operators import (
    CTFFilter,
    Filter,
    IdentityFilter,
    MultiplicativeFilter,
    PowerFilter,
)
from aspire.storage import MrcStats, StarFile
from aspire.utils import Rotation, grid_2d, support_mask, trange
from aspire.volume import IdentitySymmetryGroup, SymmetryGroup

logger = logging.getLogger(__name__)


class _ImageAccessor:
    """
    Helper class for accessing images from an ImageSource as slices via the `src.images[start:stop:step]` API.
    """

    def __init__(self, fun, num_imgs):
        """
        :param fun: The private image-accessing method specific to the ImageSource associated with this ImageAccessor.
                    Generally _images() but can be substituted with a custom method.
        :param num_imgs: The max number of images that this ImageAccessor can load (generally ImageSource.n).
        """
        self.fun = fun
        self.num_imgs = num_imgs

    def __getitem__(self, indices):
        """
        ImageAccessor can be indexed via Python slice object, 1-D NumPy array, list, or a single integer,
        corresponding to the indices of the requested images. By default, slices default to a start of 0,
        an end of self.num_imgs, and a step of 1.

        :return: An Image object containing the requested images.
        """
        if isinstance(indices, Iterable) and not isinstance(indices, np.ndarray):
            indices = np.fromiter(indices, int)
        elif isinstance(indices, (int, np.integer)):
            indices = np.array([indices])
        elif isinstance(indices, slice):
            start, stop, step = indices.start, indices.stop, indices.step

            # X[:end], slice(None, e, None) -> slice(0, e, 1)
            if not start:
                start = 0

            # Special case for X[-10:]
            if start < 0 and stop is None:
                # slice(-10, None, None) -> slice(-10, *0* ,1)
                stop = 0
            # All other cases, limit to num_imgs
            #   slice(s, None, None) -> slice(s, num_imgs, 1)
            #   slice(s, 10**10, None) -> slice(0, num_imgs, 1)
            elif not stop or stop > self.num_imgs:
                stop = self.num_imgs

            if not step:
                step = 1

            if not all(isinstance(i, (int, np.integer)) for i in [start, stop, step]):
                raise TypeError("Non-integer slice components.")
            indices = np.arange(start, stop, step)

        if not isinstance(indices, np.ndarray):
            raise KeyError(
                "Key for .images must be a slice, 1-D NumPy array, or iterable yielding integers."
            )
        if not indices.ndim == 1:
            raise KeyError("Only one-dimensional indexing is allowed for images.")

        # final check for out-of-range indices
        out_of_range = indices >= self.num_imgs
        if out_of_range.any():
            raise KeyError(f"Out-of-range indices: {list(indices[out_of_range])}")

        # check for negative indices and flip to positive
        indices = indices % self.num_imgs

        return self.fun(indices)


def _as_copy(func):
    """
    Method decorator that invokes the decorated method on a deepcopy of the object,
    and returns it on exit. The original object is unmodified.
    This allows one to take a mutating method on an object:
       obj.increment(by=2)
    and use it in a functional way:
       another = obj.increment(by=2)  # obj unmodified
    Note that the original return value of the method is lost, so this decorator
    is best used on methods that mutate the object but don't return anything.
    """

    @functools.wraps(func)  # Pass metadata (eg name and doctrings) from `func`
    def wrapper(self, *args, **kwargs):
        obj_copy = copy.deepcopy(self)
        func_copy = copy.deepcopy(func)
        func_copy(obj_copy, *args, **kwargs)
        return obj_copy

    return wrapper


class ImageSource(ABC):
    """
    When creating an `ImageSource` object, a 'metadata' table holds metadata information about all images in the
    `ImageSource`. The number of rows in this metadata table will equal the total number of images supported by this
    `ImageSource` (available as the 'n' attribute), though reading/writing of images is usually done in chunks.

    This metadata table is implemented as a dictionary of numpy arrays.

    The 'values' in this metadata table are usually primitive types (floats/ints/strings) that are suitable
    for being read from STAR files, and being written to STAR files. The columns corresponding to these fields
    begin with a single underscore '_'.

    In addition, the metadata table may also contain references to Python objects.
    `Filter` objects, for example, are stored in this metadata table as references to unique `Filter` objects that
    correspond to images in this `ImageSource`. Several rows of metadata may end up containing a reference to a small
    handful of unique `Filter` objects, depending on the values found in other columns (identical `Filter`
    objects). For example, a smaller number of CTFFilter objects may apply to subsets of particles depending on
    the unique "_rlnDefocusU"/"_rlnDefocusV" Relion parameters.
    """

    # The abstract class starts off _mutable. Conrete classes should
    # disable _mutable as the last step in __init__.
    _mutable = True

    def __init__(
        self,
        L,
        n,
        dtype="double",
        metadata=None,
        memory=None,
        symmetry_group=None,
        pixel_size=None,
    ):
        """
        A cryo-EM ImageSource object that supplies images along with other parameters for image manipulation.

        :param L: resolution of (square) images (int)
        :param n: The total number of images available
            Note that images() may return a different number of images based on its arguments.
        :param metadata: A Dataframe of metadata information corresponding to this ImageSource's images
        :param memory: str or None
            The path of the base directory to use as a data store or None. If None is given, no caching is performed.
        :param symmetry_group: A SymmetryGroup instance or string indicating the underlying symmetry of the molecule.
            Defaults to the `IdentitySymmetryGroup`, which represents an asymmetric particle, if none provided.
        :param pixel_size: Pixel size of the images in angstroms, default `None`.
        """

        # Instantiate the accessor for the `images` property
        self._img_accessor = _ImageAccessor(self._images, n)

        self.L = L
        self._n = None
        self.n = n
        self.dtype = np.dtype(dtype)
        if pixel_size is not None:
            pixel_size = float(pixel_size)
        self.pixel_size = pixel_size

        # The private attribute '_cached_im' can be populated by calling this object's cache() method explicitly
        self._cached_im = None

        # _rotations is assigned non None value
        #  by `rotations` or `angles` setters.
        #  It is potentially used by subclasses to test if we've used setters.
        #  This must come before the Relion/starfile metadata parsing below.
        self._rotations = None

        if metadata is None:
            self._metadata = {}
        else:
            self._metadata = copy.copy(metadata)
            if self.has_metadata(["_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"]):
                self._rotations = Rotation.from_euler(
                    np.deg2rad(
                        self.get_metadata(
                            ["_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"]
                        )
                    )
                )

        self._populate_symmetry_group(symmetry_group)

        self.unique_filters = []
        self.generation_pipeline = Pipeline(xforms=None, memory=memory)

        logger.info(f"Creating {self.__class__.__name__} with {len(self)} images.")

    @property
    def symmetry_group(self):
        """
        A SymmetryGroup instance associated with the symmetry type of the ImageSource object.
        Access rotation matrices of the symmetry_group via `symmetry_group.matrices`.
        """
        return self._symmetry_group

    @symmetry_group.setter
    def symmetry_group(self, value):
        """
        Set the `symmetry_group` for `src`.

        :param value: A `SymmetryGroup` instance or string indicating symmetry, ie. "C5", "D7", "T", etc.
        """
        if not self._mutable:
            raise RuntimeError(
                f"This source is no longer mutable. Try new_source = source.update(symmetry_group='{value}')."
            )

        self._symmetry_group = SymmetryGroup.parse(value, dtype=self.dtype)
        self.set_metadata(["_rlnSymmetryGroup"], str(self.symmetry_group))

    def _populate_symmetry_group(self, symmetry_group):
        """
        Populates the symmetry_group attribute with user provided symmetry_group or metadata.
        If neither exist, defaults to C1 symmetry.
        """
        if self.has_metadata(["_rlnSymmetryGroup"]):
            if symmetry_group:
                logger.warning(
                    f"Overriding metadata with supplied symmetry group {symmetry_group}"
                )
            else:
                symmetry_group = SymmetryGroup.parse(
                    symmetry=self.get_metadata(["_rlnSymmetryGroup"])[0],
                    dtype=self.dtype,
                )

        self.symmetry_group = symmetry_group or IdentitySymmetryGroup(dtype=self.dtype)

    def __getitem__(self, indices):
        """
        Check `indices` and return slice of current Source as a new
        Source.

        Internally uses `IndexedSource`.

        :param indices: Requested indices as a Python slice object,
            1-D NumPy array, list, or a single integer. Slices default
            to a start of 0, an end of self.num_imgs, and a step of 1.
            See _ImageAccessor.
        :return: Source composed of the images and metadata at `indices`.
        """

        return IndexedSource(self, indices)

    def __len__(self):
        """
        Returns total number of images in source.
        """
        return self.n

    def _metadata_as_dict(self, metadata_fields, indices, default_value=None):
        """
        Return a dictionary of selected metadata fields at selected indices.

        :param metadata_fields: An iterable of strings specifying metadata fields.
        :param indices: An ndarray of 0-indexed locations we're interested in.
        :param default_value: A scalar default value to use if a metadata_field is not found.
        :return: A dictionary of numpy arrays of specified metadata fields at specified indices.
        """
        result = {}
        for metadata_field in metadata_fields:
            if metadata_field in self._metadata:
                result[metadata_field] = self._metadata[metadata_field][indices].copy()
            else:
                if default_value is None:
                    raise ValueError(
                        f"Missing metadata field {metadata_field} and no default_value supplied"
                    )
                result[metadata_field] = np.full(len(indices), fill_value=default_value)
        return result

    def _metadata_as_ndarray(self, metadata_fields, indices, default_value=None):
        """
        Return a numpy array of selected metadata fields at selected indices.

        :param metadata_fields: An iterable of strings specifying metadata fields.
        :param indices: An ndarray of 0-indexed locations we're interested in.
        :param default_value: A scalar default value to use if a metadata_field is not found.
        :return: A numpy array of specified metadata fields at specified indices.
        """
        # Start with the most generic type - we'll narrow it later
        result = np.empty((len(indices), len(metadata_fields))).astype("object")
        # Keep track of dtypes of individual metadata fields, so we can narrow the result into a single dtype
        dtypes = [None] * len(metadata_fields)

        for i, metadata_field in enumerate(metadata_fields):
            if metadata_field not in self._metadata:
                if default_value is None:
                    raise ValueError(
                        f"Missing metadata field {metadata_field} and no default_value supplied"
                    )
                result[:, i] = default_value
                dtypes[i] = np.array([default_value]).dtype
            else:
                values = self._metadata[metadata_field][indices]
                result[:, i] = values
                dtypes[i] = values.dtype

        dtype = np.result_type(*dtypes)
        result = result.astype(dtype)

        if result.shape[1] == 1:
            result = result.squeeze(axis=1)

        return result

    def update(self, **kwargs):
        """
        Update certain properties that modify the underlying metadata, and return a new ImageSource
        object with the new properties. The original object is unchanged.
        """
        updateable_props = (
            "states",
            "filter_indices",
            "offsets",
            "amplitudes",
            "angles",
            "rotations",
            "symmetry_group",
        )

        cp = copy.deepcopy(self)
        cp._mutable = True
        for prop in updateable_props:
            if prop in kwargs:
                setattr(cp, prop, kwargs.pop(prop))

        if kwargs:
            logger.warning(f"Unhandled arguments = {kwargs.keys()}")
        cp._mutable = False

        return cp

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        """
        Sets max image index `n` in `src` and associated
        `ImageAccessor`.

        :param n: Number of images.
        """
        self._set_n(n)

    def _set_n(self, n):
        """
        Sets max image index `n` in `src` and associated
        `ImageAccessor`.

        :param n: Number of images.
        """
        # Protect _n by default.
        if self._n is not None:
            raise RuntimeError("Source `n` is already set.")

        # Enforce type, just-in-case.
        if n != int(n):
            raise TypeError("`n` must be an integer")
        n = int(n)

        self._img_accessor.num_imgs = n
        self._n = n

    @property
    def n_ctf_filters(self):
        """
        Return the number of CTFFilters found in this Source.
        """
        return len([f for f in self.unique_filters if isinstance(f, CTFFilter)])

    @property
    def states(self):
        return np.atleast_1d(self.get_metadata("_rlnClassNumber"))

    @states.setter
    def states(self, values):
        return self.set_metadata("_rlnClassNumber", values)

    @property
    def filter_indices(self):
        return np.atleast_1d(self.get_metadata("__filter_indices"))

    @filter_indices.setter
    def filter_indices(self, indices):
        # create metadata of filters for all images
        return self.set_metadata(["__filter_indices"], indices)

    @property
    def offsets(self):
        return np.atleast_2d(
            self.get_metadata(["_rlnOriginX", "_rlnOriginY"], default_value=0.0)
        )

    @offsets.setter
    def offsets(self, values):
        return self.set_metadata(
            ["_rlnOriginX", "_rlnOriginY"], np.array(values, dtype=self.dtype)
        )

    @property
    def amplitudes(self):
        return np.atleast_1d(self.get_metadata("_rlnAmplitude", default_value=1.0))

    @amplitudes.setter
    def amplitudes(self, values):
        return self.set_metadata("_rlnAmplitude", np.array(values, dtype=self.dtype))

    @property
    def angles(self):
        """
        Rotation angles in radians.

        :return: Rotation angles in radians, as a n x 3 array
        """
        # Call a private method. This allows subclasses to efficiently override.
        return self._angles()

    def _angles(self):
        """
        Converts internal _rotations representation to expected matrix form.
        """
        return self._rotations.angles.astype(self.dtype)

    @property
    def rotations(self):
        """
        Returns rotation matrices.

        :return: Rotation matrices as a n x 3 x 3 array
        """
        # Call a private method. This allows sub classes to effeciently override.
        return self._rots()

    def _rots(self):
        """
        Converts internal `_rotations` representation to expected matrix form.

        :return: Rotation matrices as a n x 3 x 3 array
        """
        return self._rotations.matrices.astype(self.dtype)

    @angles.setter
    def angles(self, values):
        """
        Set rotation angles

        :param values: Rotation angles in radians, as a n x 3 array
        :return: None
        """

        values = values.astype(self.dtype)
        self._rotations = Rotation.from_euler(values)
        self.set_metadata(
            ["_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"], np.rad2deg(values)
        )

    @rotations.setter
    def rotations(self, values):
        """
        Set rotation matrices

        :param values: Rotation matrices as a n x 3 x 3 array
        :return: None
        """

        values = values.astype(self.dtype)
        self._rotations = Rotation.from_matrix(values)
        self.set_metadata(
            ["_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"],
            np.rad2deg(self._rotations.angles),
        )

    @property
    def class_indices(self):
        """
        Returns table of class image indices as `(src.n, n_nbors)`
        Numpy array.

        Each row reprsents a class, with the columns ordered by
        smallest `class_distances` from the reference image (zeroth
        columm).

        Note `n_nbors` is managed by `self.classifier` and used here
        for documentation.

        :return: Numpy array, integers.
        """
        res = self.get_metadata(["_class_indices"])
        return np.vstack([np.array(row.split(","), dtype=int) for row in res])

    @property
    def selection_indices(self):
        return self.get_metadata(["_selection_indices"])

    @property
    def class_refl(self):
        """
        Returns table of class image reflections as `(src.n, n_nbors)`
        Numpy array.

        Follows same layout as `class_indices` but holds booleans that
        are True when the image should be reflected before averaging.

        Note `n_nbors` is managed by `self.classifier` and used here
        for documentation.

        :return: Numpy array, boolean.
        """
        res = self.get_metadata(["_class_refl"])
        # Read table of (0, 1) integers, cast to `bool`.
        return np.vstack([np.array(row.split(","), dtype=int) for row in res]).astype(
            bool
        )

    @property
    def class_distances(self):
        """
        Returns table of class image distances as `(src.n, n_nbors)`
        Numpy array.

        Follows same layout as `class_indices` but holds floats
        representing the distance (returned by classifier) to the
        zeroth image in each class.

        Note `n_nbors` is managed by `self.classifier` and used here
        for documentation.

        :return: Numpy array, self.dtype.
        """
        res = self.get_metadata(["_class_distances"])
        return np.vstack([np.array(row.split(","), dtype=self.dtype) for row in res])

    def set_metadata(self, metadata_fields, values, indices=None):
        """
        Modify metadata field information of this ImageSource for selected indices

        :param metadata_fields: A string, or list of strings, representing the metadata field(s) to be modified
        :param values: A scalar or vector (of length |indices|) of replacement values.
        :param indices: A list of 0-based indices indicating the indices for which to modify metadata.
            If indices is None, then all indices in this Source object are modified. In this case,
            values should either be a scalar or a vector of length equal to the total number of images, |self.n|.
        :return: On return, the metadata associated with the specified indices has been modified.
        """

        # This breaks lots of things, maybe not something we want to rush out.
        # # Check if we're in an immutable state.
        # if not self._mutable:
        #     raise RuntimeError("This source is no longer mutable, try using `update` instead of `set_metadata`")

        if isinstance(metadata_fields, str):
            metadata_fields = [metadata_fields]

        if indices is None:
            indices = np.arange(self.n)

        # Check if we're an iterable, and in case we're not, broadcast
        # the single values into a list `indices` long.
        try:
            iter(values)
        except TypeError:
            values = [values] * len(indices)
        else:
            # Special case for single `str`, which are iterable, but
            # need to be broadcast like a singleton.
            if isinstance(values, str):
                values = [values] * len(indices)
        if len(values) != len(indices):
            raise RuntimeError(
                f"Mismatch between len(values) {len(values)} and len(indices) {len(indices)}."
            )

        values = np.array(values)  # make a copy for our use

        # When creating metadata fields that are string, coerce them into python objects to allow string expansion
        # later (replacing 'hello' with 'goodbye', for example).
        # This is in part to conform to legacy implementation of metadata that used pandas,
        # but is probably a good thing to do anyway.
        # We also come up with a sensible fill value while we're at it.
        if np.issubdtype(values.dtype, np.str_):
            values = values.astype("object")
            fill_value = ""
        elif np.issubdtype(values.dtype, np.integer):
            # For integers, we'll use the minimal value.
            # This will be a large negative value when signed,
            # and zero for unsigned integers.
            fill_value = np.iinfo(values.dtype).min
        else:
            fill_value = np.nan

        if values.ndim == 1:
            values = values.reshape(-1, 1)  # convert to column
            values = np.tile(
                values, (1, len(metadata_fields))
            )  # stack columns for each metadata field

        for i, metadata_field in enumerate(metadata_fields):
            if metadata_field not in self._metadata:
                self._metadata[metadata_field] = np.full(self.n, fill_value).astype(
                    values.dtype
                )
            self._metadata[metadata_field][indices] = values[:, i]

    def has_metadata(self, metadata_fields):
        """
        Find out if one or more metadata fields are available for this `ImageSource`.

        :param metadata_fields: A string, of list of strings, representing the metadata field(s) to be queried.
        :return: Boolean value indicating whether the field(s) are available.
        """
        if isinstance(metadata_fields, str):
            metadata_fields = [metadata_fields]
        return all(f in self._metadata for f in metadata_fields)

    def get_metadata(
        self, metadata_fields=None, indices=None, default_value=None, as_dict=False
    ):
        """
        Get metadata field information of this ImageSource for a
        selection of fields of indices.  The default should return the
        entire metadata table.

        :param metadata_fields: A string, or list of strings,
            representing the metadata field(s) to be queried.
            Defaults to None, which yields all populated columns.
        :param indices: A list of 0-based indices indicating the
            indices for which to get metadata.  If indices is None,
            then values corresponding to all indices in this Source
            object are returned.
        :param default_value: Default scalar value to use for any
            fields not found in the metadata. If None, no default
            value is used, and missing field(s) cause a RuntimeError.
        :param as_dict: Boolean indicating whether we want to return
            metadata as a dictionary (True), or as a numpy ndarray (False).
            In the latter case, all returned values are typecast to a common
            numpy dtype, so use with caution.
        :return: An ndarray of values (any valid np types)
            representing metadata info. If is_dict is True, then returns
            a dictionary mapping metadata names to numpy arrays of values.
        """
        if metadata_fields is None:
            metadata_fields = list(self._metadata.keys())
        if isinstance(metadata_fields, str):
            metadata_fields = [metadata_fields]

        if indices is None:
            indices = np.arange(self.n)
        else:
            try:
                iter(indices)
            except TypeError:
                indices = [indices]

        if as_dict:
            return self._metadata_as_dict(
                metadata_fields=metadata_fields,
                indices=indices,
                default_value=default_value,
            )
        else:
            return self._metadata_as_ndarray(
                metadata_fields=metadata_fields,
                indices=indices,
                default_value=default_value,
            )

    def _apply_filters(
        self,
        im_orig,
        filters,
        indices,
    ):
        """
        For images in `im_orig`, `filters` associated with the corresponding
        index in the supplied `indices` are applied. The images are then returned as an `Image` stack.

        :param im_orig: An `Image` object
        :param filters: A list of `Filter` objects
        :param indices: A list of indices indicating the corresponding filter in `filters`
        """
        if not isinstance(im_orig, Image):
            logger.warning(
                f"_apply_filters() passed {type(im_orig)} instead of Image instance"
            )
            # for now just convert it
            im_orig = Image(im_orig, pixel_size=self.pixel_size)

        im = im_orig.copy()

        for i, filt in enumerate(filters):
            idx_k = np.where(indices == i)[0]
            if len(idx_k) > 0:
                im[idx_k] = im[idx_k].filter(filt).asnumpy()

        return im

    def _apply_source_filters(self, im_orig, indices):
        return self._apply_filters(
            im_orig,
            self.unique_filters,
            self.filter_indices[indices],
        )

    @_as_copy
    def cache(self):
        """
        Computes all queued pipeline transformations and stores the
        generated images in an array.  This trades memory for fast
        image access, and is useful when images will be repeatedly
        queried since it avoids recomputing on-the-fly.
        """
        logger.info("Caching source images")
        self._cached_im = self.images[:]
        self.generation_pipeline.reset()

    @property
    def images(self):
        """
        Subscriptable property which returns the images contained in this source
        corresponding to the indices given.
        """
        return self._img_accessor

    @abstractmethod
    def _images(self, indices):
        """
        Subclasses must implement a private _images() method accepting a 1-D NumPy array of indices.
        Subclasses handle cached image check as well as applying transforms in the generation pipeline.
        """

    @_as_copy
    def downsample(self, L):
        if L > self.L:
            raise ValueError(
                "Max desired resolution {L} should be less than the current resolution {self.L}."
            )
        logger.info(f"Setting max. resolution of source = {L}")

        self.generation_pipeline.add_xform(Downsample(resolution=L))

        ds_factor = self.L / L
        self.unique_filters = [f.scale(ds_factor) for f in self.unique_filters]
        self.offsets /= ds_factor

        self.L = L

    @_as_copy
    def whiten(self, noise_estimate=None, epsilon=None):
        """
        Modify the `ImageSource` in-place by appending a whitening filter to the generation pipeline.

        When no `noise_estimate` is provided, a `WhiteNoiseEstimator`
        will be instantiated at this preprocessing stage behind the
        scenes.

        :param noise_estimate: Optional, `NoiseEstimator` or `Filter`. When
            passed a `NoiseEstimator` the `filter` attribute will be
            queried.  Alternatively, the noise PSD may be passed
            directly as a `Filter` object.
        :param epsilon: Threshold used to determine which frequencies to whiten
            and which to set to zero. By default all PSD values in the `noise_estimate`
            less than eps(self.dtype) are zeroed out in the whitening filter.
        :return: On return, the `ImageSource` object has been modified in place.
        """

        if noise_estimate is None:
            noise_filter = WhiteNoiseEstimator(self).filter
        elif isinstance(noise_estimate, NoiseEstimator):
            # Any NoiseEstimator instance should have a `filter`.
            noise_filter = noise_estimate.filter
        elif isinstance(noise_estimate, Filter):
            # We were given a `Filter` object.
            noise_filter = noise_estimate
        else:
            raise TypeError(
                f"Whiten passed {noise_estimate}"
                " instead of `NoiseEstimator` or `Filter`."
            )

        if epsilon is None:
            epsilon = np.finfo(self.dtype).eps

        logger.info("Whitening source object")
        whiten_filter = PowerFilter(noise_filter, power=-0.5, epsilon=epsilon)

        logger.info("Transforming all CTF Filters into Multiplicative Filters")
        self.unique_filters = [
            MultiplicativeFilter(f, whiten_filter) for f in self.unique_filters
        ]
        logger.info("Adding Whitening Filter Xform to end of generation pipeline")
        self.generation_pipeline.add_xform(FilterXform(whiten_filter))

    @_as_copy
    def phase_flip(self):
        """
        Perform phase flip on images in the source object using CTF information.
        If no CTFFilters exist this will emit a warning and otherwise no-op.
        """

        logger.info("Perform phase flip on source object")

        if len(self.unique_filters) >= 1:
            unique_xforms = [FilterXform(f.sign) for f in self.unique_filters]

            logger.info("Adding Phase Flip Xform to end of generation pipeline")
            self.generation_pipeline.add_xform(
                IndexedXform(unique_xforms, self.filter_indices)
            )

        else:
            # No CTF filters found
            logger.warning(
                "No Filters found."
                "  `phase_flip` is a no-op without Filters."
                "  Confirm you have correctly populated CTFFilters."
            )

    @_as_copy
    def invert_contrast(self, batch_size=512):
        """
        invert the global contrast of images

        Check if all images in a stack should be globally phase flipped so that
        the molecule corresponds to brighter pixels and the background corresponds
        to darker pixels. This is done by comparing the mean in a small circle
        around the origin (supposed to correspond to the molecule) with the mean
        of the noise, and making sure that the mean of the molecule is larger.
        From the implementation level, we modify the `ImageSource` in-place by
        appending a `Multiple` filter to the generation pipeline.

        :param batch_size: Batch size of images to query.
        :return: On return, the `ImageSource` object has been modified in place.
        """

        logger.info("Apply contrast inversion on source object")
        L = self.L
        grid = grid_2d(L, indexing="yx", shifted=True)
        # Get mask indices of signal and noise samples assuming molecule
        signal_mask = grid["r"] < 0.5
        noise_mask = grid["r"] > 0.8

        # Calculate mean values in batch_size
        signal_mean = 0.0
        noise_mean = 0.0

        logger.info("Computing signal vs background contrast on source object")
        for i in trange(0, self.n, batch_size):
            images = self.images[i : i + batch_size].asnumpy()
            signal = images * signal_mask
            noise = images * noise_mask
            signal_mean += np.sum(signal)
            noise_mean += np.sum(noise)
        signal_denominator = self.n * np.sum(signal_mask)
        noise_denominator = self.n * np.sum(noise_mask)
        signal_mean /= signal_denominator
        noise_mean /= noise_denominator

        if signal_mean < noise_mean:
            logger.info("Need to invert contrast")
            scale_factor = -1.0
        else:
            logger.info("No need to invert contrast")
            scale_factor = 1.0

        logger.info("Adding Scaling Xform to end of generation pipeline")
        self.generation_pipeline.add_xform(Multiply(scale_factor))

    @_as_copy
    def normalize_background(self, bg_radius=1.0, do_ramp=True):
        """
        Normalize the images by the noise background

        This is done by shifting the image density by the mean value of background
        and scaling the image density by the standard deviation of background.
        From the implementation level, we modify the `ImageSource` in-place by
        appending the `Add` and `Multiple` filters to the generation pipeline.

        :param bg_radius: Radius cutoff to be considered as background (in image size)
        :param do_ramp: When it is `True`, fit a ramping background to the data
            and subtract. Namely perform normalization based on values from each image.
            Otherwise, a constant background level from all images is used.
        :return: On return, the `ImageSource` object has been modified in place.
        """

        logger.info(
            f"Normalize background on source object with radius "
            f"size of {bg_radius} and do_ramp of {do_ramp}"
        )
        self.generation_pipeline.add_xform(
            LambdaXform(normalize_bg, bg_radius=bg_radius, do_ramp=do_ramp)
        )

    def im_backward(self, im, start, weights=None, symmetry_group=None):
        """
        Apply adjoint mapping to set of images

        :param im: An Image instance to which we wish to apply the adjoint of the forward model.
        :param start: Start index of image to consider
        :param weights: Optional vector of weights to apply to images.
            Weights should be length `self.n`.
        :param symmetry_group: A SymmetryGroup instance. If supplied, uses symmetry to increase
             number of images used in back-projectioon.
        :return: An L-by-L-by-L volume containing the sum of the adjoint mappings applied to the start+num-1 images.
        """
        num = im.n_images

        all_idx = np.arange(start, min(start + num, self.n))
        im *= self.amplitudes[all_idx, np.newaxis, np.newaxis]
        im = im.shift(-self.offsets[all_idx, :])
        im = self._apply_source_filters(im, all_idx)

        if weights is not None:
            im *= weights[all_idx, np.newaxis, np.newaxis]

        vol = im.backproject(
            self.rotations[start : start + num, :, :], symmetry_group=symmetry_group
        )[0]

        return vol

    def vol_forward(self, vol, start, num):
        """
        Apply forward image model to volume

        :param vol: A volume instance.
        :param start: Start index of image to consider
        :param num: Number of images to consider
        :return: The images obtained from volume by projecting, applying CTFs, translating, and multiplying by the
            amplitude.
        """
        all_idx = np.arange(start, min(start + num, self.n))
        if vol.n_vols != 1:
            raise ValueError("vol_forward expects a single volume, not a stack.")

        if vol.dtype != self.dtype:
            logger.warning(f"Volume.dtype {vol.dtype} inconsistent with {self.dtype}")

        im = vol.project(self.rotations[all_idx, :, :])
        im = self._apply_source_filters(im, all_idx)
        im = im.shift(self.offsets[all_idx, :])
        im *= self.amplitudes[all_idx, np.newaxis, np.newaxis]
        return im

    def save(
        self,
        starfile_filepath,
        batch_size=512,
        save_mode=None,
        overwrite=False,
    ):
        """
        Save the output metadata to STAR file and/or images to MRCS file.

        :param starfile_filepath: Path to STAR file where we want to
            save metadata of image_source
        :param batch_size: Batch size of images to query.
            Note, `batch_size=1` implies single MRC extension `.mrc`,
            while `batch_size>=1` implies stack MRC extension `.mrcs`.
        :param save_mode: Whether to save all images in a `single` or multiple files in batch size.
            Default is multiple, supply `'single'` for single mode.
        :param overwrite: Option to overwrite the output MRC files.
        :return: A dictionary containing "starfile"--the path to the saved starfile-- and "mrcs", a
            list of the saved particle stack MRC filenames.
        """
        logger.info("save metadata into STAR file")
        filename_indices = self.save_metadata(
            starfile_filepath,
            batch_size=batch_size,
            save_mode=save_mode,
        )
        unique_filenames = list(dict.fromkeys(filename_indices))

        logger.info("save images into MRCS file")
        self.save_images(
            starfile_filepath,
            filename_indices=filename_indices,
            batch_size=batch_size,
            overwrite=overwrite,
        )
        # return some information about the saved files
        info = {"starfile": starfile_filepath, "mrcs": unique_filenames}
        return info

    @staticmethod
    def _populate_common_metadata(
        n,
        meta_dict,
        local_cols,
        starfile_filepath,
        batch_size,
        save_mode,
    ):
        """
        Populate metadata columns common to all `ImageSource` subclasses.
        """
        # Create a new key that we will be populating in the loop below
        meta_dict["_rlnImageName"] = np.full(n, fill_value="").astype("object")

        if save_mode == "single":
            # Save all images into one single mrc file
            fname = os.path.basename(starfile_filepath)
            fstem = os.path.splitext(fname)[0]
            mrcs_filename = f"{fstem}_{0}_{n-1}.mrcs"

            # Then set name in dict for the StarFile
            meta_dict["_rlnImageName"][:] = [
                f"{j + 1:06}@{mrcs_filename}" for j in range(n)
            ]
        else:
            # save all images into multiple mrc files in batch size
            # When batch_size is >1, use the plural extension.
            ext = ".mrcs"
            if batch_size == 1:
                ext = ".mrc"

            for i_start in np.arange(0, n, batch_size):
                i_end = min(n, i_start + batch_size)
                num = i_end - i_start
                mrcs_filename = (
                    os.path.splitext(os.path.basename(starfile_filepath))[0]
                    + f"_{i_start}_{i_end-1}{ext}"
                )
                meta_dict["_rlnImageName"][i_start:i_end] = [
                    "{0:06}@{1}".format(j + 1, mrcs_filename) for j in range(num)
                ]

        # Subclass-specific columns are popped to the end of the dictionary in order:
        # pop() both removes the given column and returns its data as a ndarray,
        # which is then tacked back on to the rightmost side of metadata
        # Note that all dictionaries in py>=3.7 are ordered
        for col in local_cols:
            meta_dict[col] = meta_dict.pop(col)

    def _populate_local_metadata(self):
        """
        Populate metadata columns specific to the `ImageSource` subclass being saved.
        Subclasses optionally override, but must return a list of strings.

        :return: A list of the names of the columns added.
        """
        return []

    def save_metadata(self, starfile_filepath, batch_size=512, save_mode=None):
        """
        Save updated metadata to a STAR file

        :param starfile_filepath: Path to STAR file where we want to
            save image_source
        :param batch_size: Batch size of images to query from the
            `ImageSource` object. Every `batch_size` rows, entries are
            written to STAR file.
        :param save_mode: Whether to save all images in a single or
            multiple files in batch size.
        :return: None
        """

        # Get local metadata columns that were added by subclass
        local_cols = self._populate_local_metadata()

        metadata = self.get_metadata(as_dict=True).copy()
        # Drop any column that doesn't start with a *single* underscore
        metadata = {
            k: v
            for k, v in metadata.items()
            if k.startswith("_") and not k.startswith("__")
        }

        # Populates _rlnImageName column, setting up filepaths to .mrcs stacks
        self._populate_common_metadata(
            self.n, metadata, local_cols, starfile_filepath, batch_size, save_mode
        )

        filename_indices = [
            x[1]
            for x in np.char.split(metadata["_rlnImageName"].astype(np.str_), sep="@")
        ]

        # initialize the star file object and save it
        odict = OrderedDict()
        # since our StarFile only has one block, the convention is to save it with the header "data_", i.e. its name is blank
        # if we had a block called "XYZ" it would be saved as "XYZ"
        # thus we index the metadata block with ""
        odict[""] = metadata
        out_star = StarFile(blocks=odict)
        out_star.write(starfile_filepath)
        return filename_indices

    def save_images(
        self, starfile_filepath, filename_indices=None, batch_size=512, overwrite=False
    ):
        """
        Save an ImageSource to MRCS files

        Note that .mrcs files are saved at the same location as the STAR file.

        :param filename_indices: Filename list for save all images
        :param starfile_filepath: Path to STAR file where we want to save image_source
        :param batch_size: Batch size of images to query from the `ImageSource` object.
            if `save_mode` is not `single`, images in the same batch will save to one MRCS file.
        :param overwrite: Whether to overwrite any .mrcs files found at the target location.
        :return: None
        """

        if filename_indices is None:
            # Generate filenames from metadata
            filename_indices = [
                self._metadata["_rlnImageName"][i].split("@")[1] for i in range(self.n)
            ]

        # get the save_mode from the file names
        unique_filename = set(filename_indices)
        save_mode = None
        if len(unique_filename) == 1:
            save_mode = "single"

        if save_mode == "single":
            # Save all images into one single mrc file

            # First, construct name for mrc file
            fdir = os.path.dirname(starfile_filepath)
            mrcs_filepath = os.path.join(fdir, filename_indices[0])

            # Open new MRC file
            with mrcfile.new_mmap(
                mrcs_filepath,
                shape=(self.n, self.L, self.L),
                mrc_mode=2,
                overwrite=overwrite,
            ) as mrc:
                stats = MrcStats()
                # Loop over source setting data into mrc file
                for i_start in np.arange(0, self.n, batch_size):
                    i_end = min(self.n, i_start + batch_size)
                    logger.info(
                        f"Saving ImageSource[{i_start}-{i_end-1}] to {mrcs_filepath}"
                    )
                    datum = self.images[i_start:i_end].asnumpy().astype("float32")

                    # Assign to mrcfile
                    mrc.data[i_start:i_end] = datum

                    # Accumulate stats
                    stats.push(datum)

                # To be safe, explicitly set the header
                #   before the mrc file context closes.
                mrc.update_header_from_data()

                # Also write out updated statistics for this mrc.
                #   This should be an optimization over mrc.update_header_stats
                #   for large arrays.
                stats.update_header(mrc)

        else:
            # save all images into multiple mrc files in batch size
            for i_start in np.arange(0, self.n, batch_size):
                i_end = min(self.n, i_start + batch_size)

                mrcs_filepath = os.path.join(
                    os.path.dirname(starfile_filepath), filename_indices[i_start]
                )

                logger.info(
                    f"Saving ImageSource[{i_start}-{i_end-1}] to {mrcs_filepath}"
                )
                im = self.images[i_start:i_end]
                im.save(mrcs_filepath, overwrite=overwrite)

    def estimate_signal_mean_energy(
        self,
        sample_n=None,
        support_radius=None,
        batch_size=512,
        image_accessor=None,
    ):
        """
        Estimate the signal mean of `sample_n` projections.

        :param sample_n: Number of images used for estimate.
            Defaults to all images in source.
        :param support_radius: Pixel radius used for masking signal support.
            Default of None will compute inscribed circle, `self.L // 2`.
        :param batch_size: Images per batch, defaults 512.
        :param image_accessor: Optionally override images. Defaults `self.images`.
        :returns: Estimated signal mean
        """

        if sample_n is None:
            sample_n = self.n

        if sample_n > self.n:
            logger.warning(
                f"`estimate_signal_mean_energy` sample_n > Source.n: {sample_n} > {self.n}."
                f" Accuracy may be impaired, setting sample_n=self.n={self.n}"
            )
            sample_n = self.n

        images = image_accessor or self.images

        mask = support_mask(self.L, support_radius, dtype=self.dtype)

        # mean is estimated batch-wise, compare with numpy
        s = 0.0
        _denom = sample_n * np.sum(mask)
        for i in trange(0, sample_n, batch_size):
            # Gather this batch of images and mask off area outside support_radius
            images_masked = images[i : i + batch_size].asnumpy()[..., mask]
            # Accumulate second moments
            s += np.sum(images_masked**2) / _denom

        logger.debug(f"Source estimated signal mean energy: {s}")

        return s

    def estimate_signal_var(
        self, sample_n=None, support_radius=None, batch_size=512, image_accessor=None
    ):
        """
        Estimate the signal variance of `sample_n` projections.

        :param sample_n: Number of images used for estimate.
            Defaults to all images in source.
        :param support_radius: Pixel radius used for masking signal support.
            Default of None will compute inscribed circle, `self.L // 2`.
        :param batch_size: Images per batch, defaults 512.
        :param image_accessor: Optionally override images. Defaults `self.images`.
        :returns: Estimated signal variance.
        """

        if sample_n is None:
            sample_n = self.n

        if sample_n > self.n:
            logger.warning(
                f"`estimate_signal_var` sample_n > Source.n: {sample_n} > {self.n}."
                f" Accuracy may be impaired, setting sample_n=self.n={self.n}"
            )
            sample_n = self.n

        images = image_accessor or self.images

        mask = support_mask(self.L, support_radius, dtype=self.dtype)

        # Var is estimated batch-wise, compare with numpy
        # np_estimated_var = np.var(images[:sample_n].asnumpy()[..., mask])
        first_moment = 0.0
        second_moment = 0.0
        _denom = sample_n * np.sum(mask)
        for i in trange(0, sample_n, batch_size):
            # Gather this batch of images and mask off area outside support_radius
            images_masked = images[i : i + batch_size].asnumpy()[..., mask]
            # Accumulate first and second moments
            first_moment += np.sum(images_masked) / _denom
            second_moment += np.sum(images_masked**2) / _denom

        # E[X**2] - E[X]**2
        estimated_var = second_moment - first_moment**2
        logger.debug(f"Source estimated signal var: {estimated_var}")

        return estimated_var

    def estimate_signal_power(
        self,
        sample_n=None,
        support_radius=None,
        batch_size=512,
        signal_power_method="estimate_signal_mean_energy",
        image_accessor=None,
    ):
        """
        Estimate the signal energy of `sample_n` projections using prescribed method.

        :param sample_n: Number of images used for estimate.
            Defaults to all images in source.
        :param support_radius: Pixel radius used for masking signal support.
            Default of None will compute inscribed circle, `self.L // 2`.
        :param batch_size: Images per batch, defaults 512.
        :param signal_power_method: Method used for computing signal energy.
           Defaults to mean via `estimate_signal_mean_energy`.
           Can use variance method via `estimate_signal_var`.
        :param image_accessor: Optionally override images. Defaults `self.images`.
        :returns: Estimated signal variance.
        """

        try:
            signal_estimate_method = getattr(self, signal_power_method)
        except AttributeError:
            raise ValueError(
                f"Cannot find signal_power_method={signal_power_method}."
                "  Try the default 'estimate_signal_mean_energy' or 'estimate_signal_var'"
            )

        signal_power = signal_estimate_method(
            sample_n=sample_n,
            support_radius=support_radius,
            batch_size=batch_size,
            image_accessor=image_accessor,
        )

        return signal_power

    def estimate_noise_power(
        self,
        sample_n=None,
        support_radius=None,
        batch_size=512,
    ):
        """
        Estimate the noise energy of `sample_n` images using
        `WhiteNoiseEstimator`.

        :param sample_n: Number of images used for estimate.
            Defaults to all images in source.
        :param support_radius: Pixel radius used for masking signal support.
            Default of None will compute inscribed circle, `self.L // 2`.
        :param batch_size: Images per batch, defaults 512.

        :returns: Estimated noise energy (variance).
        """

        if support_radius is None:
            support_radius_proportion = 1
        else:
            # Note, noise_estimator expects radius as proportion.
            support_radius_proportion = support_radius / (self.L // 2)

        est = WhiteNoiseEstimator(
            src=self, bgRadius=support_radius_proportion, batchSize=batch_size
        )

        return est.estimate()

    def estimate_snr(
        self,
        sample_n=None,
        support_radius=None,
        batch_size=512,
        noise_power=None,
        signal_power_method="estimate_signal_mean_energy",
    ):
        """
        Estimate the SNR of the simulated data set using
        estimated signal power / noise power.

        Note signal power depends on choice of `signal_power_method`,
        but differences should be small in practice when background
        noise is zero centered.

        :param sample_n: Number of images used for estimate.
            Defaults to all images in source.
        :param support_radius: Pixel radius used for masking signal support.
            Default of None will compute inscribed circle, `self.L // 2`.
        :param batch_size: Images per batch, defaults 512.
        :param signal_power_method: Method used for computing signal energy.
           Defaults to mean via `estimate_signal_mean_energy`.
           Can use variance method via `estimate_signal_var`.
        :returns: Estimated signal to noise ratio.
        """

        if sample_n is None:
            sample_n = self.n

        if sample_n > self.n:
            logger.warning(
                f"`estimate_snr` sample_n > Source.n: {sample_n} > {self.n}."
                f" Accuracy may be impaired, setting sample_n=self.n={self.n}"
            )
            sample_n = self.n

        if noise_power is None:
            noise_power = self.estimate_noise_power()

        signal_power = self.estimate_signal_power(
            sample_n=sample_n,
            support_radius=support_radius,
            batch_size=batch_size,
            signal_power_method=signal_power_method,
        )

        # For `estimate_signal_mean_energy` we yield: mean(signal**2)
        #     `estimate_signal_var`   we yield: signal_variance
        snr = (signal_power - noise_power) / noise_power

        # Check for extremal values.
        if snr < 0:
            logger.warning(
                "For extremely low SNR, estimation accuracy may be impaired."
                f"  Clamping estimated SNR {snr} to 0."
            )
            snr = 0

        return snr


class IndexedSource(ImageSource):
    """
    Map into another into ImageSource.
    """

    def __init__(self, src, indices, memory=None):
        """
        Instantiates a new source along given `indices`.

        :param src: ImageSource to be used as the source.
        :param index_map: index_map
        :param memory: str or None
            The path of the base directory to use as a data store or
            None. If None is given, no caching is performed.
        """

        self.src = src
        if not isinstance(src, ImageSource):
            raise TypeError(f"Input src {src} must be an ImageSource.")

        # `_ImageAccessor` performs checking and slicing logic.
        # `index_map` sequence forms a natural map from the "new" source -> "self".
        # Example, if request=slice(500,1000),
        #   then new_src[0] ~> old_src[500]; index_map[0] = 500.
        self.index_map = _ImageAccessor(lambda x: x, src.n)[indices]

        # Get all the metadata associated with these indices.
        metadata = self.src.get_metadata(indices=self.index_map, as_dict=True).copy()

        # Construct a fully formed ImageSource with this metadata
        super().__init__(
            L=src.L,
            n=len(self.index_map),
            dtype=src.dtype,
            metadata=metadata,
            memory=memory,
            pixel_size=src.pixel_size,
        )

        # Create filter indices, these are required to pass unharmed through filter eval code
        #   that is potentially called by other methods later.
        self.filter_indices = np.zeros(self.n, dtype=int)
        self.unique_filters = [IdentityFilter()]

        # Any further operations should not mutate this instance.
        self._mutable = False

    def _images(self, indices):
        """
        Returns images from `self.src` corresponding to `indices`
        remapped by `self.index_map`.

        :param indices: A 1-D NumPy array of indices.
        :return: An `Image` object.
        """

        if self._cached_im is not None:
            im = self._cached_im[indices]
        else:
            mapped_indices = self.index_map[indices]
            # Load previous source image data and apply any transforms
            # belonging to this IndexedSource.  Note the previous source
            # requires remapped indices, while the current source uses the
            # `indices` arg directly.
            im = self.src.images[mapped_indices]

        return self.generation_pipeline.forward(im, indices)

    def __repr__(self):
        return f"{self.__class__.__name__} mapping {self.n} of {self.src.n} indices from {self.src.__class__.__name__}."


class OrientedSource(IndexedSource):
    """
    Source for oriented 2D images using orientation estimation methods.
    """

    def __init__(self, src, orientation_estimator=None):
        """
        Constructor of an oriented ImageSource object. Orientation is determined by
        performing orientation estimation using a supplied `orientation_estimator`.

        :param src: Source used for orientation estimation
        :param orientation_estimator: CLOrient3D subclass used for orientation estimation.
            Default uses the CLSyncVoting method.
        """

        self.src = src
        if not isinstance(self.src, ImageSource):
            raise ValueError(
                f"`src` should be subclass of `ImageSource`, found {self.src}."
            )

        # `indices` for IndexedSource.
        indices = np.arange(self.src.n)

        super().__init__(
            src=self.src,
            indices=indices,
        )

        # Remove any orientation information passed in by original source.
        self._reset_orientation()

        # Flag for lazy eval of orientation estimation.
        self._oriented = False

        if orientation_estimator is None:
            orientation_estimator = CLSyncVoting(src)

        self.orientation_estimator = orientation_estimator
        if not isinstance(self.orientation_estimator, CLOrient3D):
            raise ValueError(
                "`orientation_estimator` should be subclass of `CLOrient3D`,"
                f" found {self.orientation_estimator}."
            )

        # Any further operations should not mutate this instance.
        self._mutable = False

    def _orient(self):
        """
        Perform orientation estimation if not already done.
        """

        # Short circuit.
        if self._oriented:
            logger.debug(f"{self.__class__.__name__} already oriented, skipping")
            return

        logger.info(
            f"Estimating rotations for {self.src} using {self.orientation_estimator}."
        )
        self.orientation_estimator.estimate_rotations()

        # Allow mutability to set rotations.
        self._mutable = True
        self.rotations = self.orientation_estimator.rotations
        self._mutable = False

        self._oriented = True

    def _reset_orientation(self):
        """
        Remove orientation information passed in by original source.
        """
        _info_removed = False
        rot_keys = ["_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"]
        for key in rot_keys:
            if self.has_metadata(key):
                del self._metadata[key]
                _info_removed = True
        if _info_removed:
            logger.info(f"Removing orientation information passed by {self.src}.")

    def _rots(self):
        """
        Converts internal `_rotations` representation to expected matrix form.
        If rotations have not been set orientation estimation will be performed
        prior to serving up the rotations.

        :return: Rotation matrices as a n x 3 x 3 array
        """

        self._orient()
        return self._rotations.matrices.astype(self.dtype)

    def _angles(self):
        self._orient()
        return super()._angles()

    def save_metadata(self, starfile_filepath, batch_size=512, save_mode=None):
        self._orient()
        return super().save_metadata(
            starfile_filepath, batch_size=batch_size, save_mode=None
        )

    def get_metadata(
        self, metadata_fields=None, indices=None, default_value=None, as_dict=False
    ):
        # get_metadata is used during ImageSource instantiation, so only perform lazy
        # orientation estimation if the oriented_source is already initialized,
        # ie. when no longer mutable.
        if not self._mutable:
            self._orient()
        return super().get_metadata(
            metadata_fields=metadata_fields,
            indices=indices,
            default_value=default_value,
            as_dict=as_dict,
        )

    def __repr__(self):
        return f"{self.__class__.__name__} for origin source {self.src.__class__.__name__}."


class ArrayImageSource(ImageSource):
    """
    An `ImageSource` object that holds a reference to an underlying `Image` object (a thin wrapper on an ndarray)
    representing images. It does not produce its images on the fly, but keeps them in memory. As such, it should not be
    used where large Image objects are involved, but can be used in situations where API conformity is desired.

    Note that this class does not provide an `_images` method, since it populates the `_cached_im` attribute which,
    if available, is consulted directly by the parent class, bypassing `_images`.
    """

    def __init__(
        self, im, metadata=None, angles=None, symmetry_group=None, pixel_size=None
    ):
        """
        Initialize from an `Image` object.

        :param im: An `Image` or Numpy array object representing image data served up by this `ImageSource`.
            In the case of a Numpy array, attempts to create an 'Image' object.
        :param metadata: A Dataframe of metadata information corresponding to this ImageSource's images
        :param angles: Optional n-by-3 array of rotation angles corresponding to `im`.
        :param symmetry_group: A SymmetryGroup instance or string indicating the underlying symmetry of the molecule.
        :param pixel_size: Pixel size of the images in angstroms, default `None`.
        """

        if not isinstance(im, Image):
            logger.info("Attempting to create an Image object from Numpy array.")
            try:
                im = Image(im, pixel_size=pixel_size)
            except Exception as e:
                raise RuntimeError(
                    "Creating Image object from Numpy array failed."
                    f" Original error: {str(e)}"
                )

        super().__init__(
            L=im.resolution,
            n=im.n_images,
            dtype=im.dtype,
            metadata=metadata,
            memory=None,
            symmetry_group=symmetry_group,
            pixel_size=im.pixel_size,
        )

        self._cached_im = im

        # Create filter indices, these are required to pass unharmed through filter eval code
        #   that is potentially called by other methods later.
        self.filter_indices = np.zeros(self.n, dtype=int)
        self.unique_filters = [IdentityFilter()]

        # Optionally populate angles/rotations.
        if angles is not None:
            if angles.shape != (self.n, 3):
                raise ValueError(f"Angles should be shape {(self.n, 3)}")
            # This will populate `_rotations`,
            #   which is exposed by properties `angles` and `rotations`.
            self.angles = angles

        # Any further operations should not mutate this instance.
        self._mutable = False

    def _images(self, indices):
        """
        Returns images corresponding to `indices` after being accessed via the
        `ImageSource.images` property

        :param indices: A 1-D NumPy array of indices.
        :return: An `Image` object.
        """
        # Load cached data and apply transforms
        return self.generation_pipeline.forward(self._cached_im[indices, :, :], indices)

    def _rots(self):
        """
        Private method, checks if `_rotations` has been set,
        then returns inherited rotations, otherwise raise.
        """

        if self._rotations is not None:
            return super()._rots()
        else:
            raise RuntimeError(
                "Consumer of ArrayImageSource trying to access rotations,"
                " but rotations were not defined for this source."
                "  Try instantiating with angles."
            )

    def _angles(self):
        """
        Private method, checks if `_rotations` has been set,
        then returns inherited angles, otherwise raise.
        """

        if self._rotations is not None:
            return super()._angles()
        else:
            raise RuntimeError(
                "Consumer of ArrayImageSource trying to access angles,"
                " but angles were not defined for this source."
                "  Try instantiating with angles."
            )
