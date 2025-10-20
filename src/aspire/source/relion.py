import logging
import os.path
from itertools import groupby, islice
from operator import itemgetter

import mrcfile
import numpy as np

from aspire.image import Image
from aspire.operators import CTFFilter, IdentityFilter
from aspire.source import ImageSource
from aspire.utils import RelionStarFile

logger = logging.getLogger(__name__)


class RelionSource(ImageSource):
    """
    A RelionSource represents a source of picked and cropped particles stored as slices in a `.mrcs` stack.
    It must be instantiated via a STAR file, which--at a minumum--lists the particles in each `.mrcs` stack in the
    `_rlnImageName` column. The STAR file may also contain Relion-specific metadata columns. This information
    is read into dictionaries containing rows for each particle specifying its location and
    its metadata. The metadata table may be augmented or modified via helper methods found in ImageSource. It may
    store, for example, Filter objects added during preprocessing.
    """

    def __init__(
        self,
        filepath,
        data_folder=None,
        pixel_size=None,
        B=0,
        n_workers=-1,
        max_rows=None,
        symmetry_group=None,
        memory=None,
        dtype=None,
    ):
        """
        Load STAR file at given filepath

        :param filepath: Absolute or relative path to STAR file
        :param data_folder: Path to folder w.r.t which all relative paths to .mrcs files are resolved.
            If None, the folder corresponding to filepath is used.
        :param pixel_size: The pixel size of the images in angstroms. By default, pixel size is
            populated from the STAR file if relevant metadata fields exist. If not found in metadata,
            pixel_size must be provided.
        :param B: the envelope decay of the CTF in inverse square angstrom (Default 0)
        :param n_workers: Number of threads to spawn to read referenced .mrcs files (Default -1 to auto detect)
        :param max_rows: Maximum number of rows in STAR file to read. If None, all rows are read.
            Note that this refers to the max number of images to load, not the max. number of .mrcs files (which may be
            equal to or less than the number of images).
        :param symmetry_group: A `SymmetryGroup` object or string corresponding to the symmetry of the molecule.
        :param memory: str or None
            The path of the base directory to use as a data store or None. If None is given, no caching is performed.
        :param dtype: Optional datatype override.
            Default `None` infers dtype from underlying image (MRC) files.
            Can be used to upcast STAR files for processing in double precision.
        """
        logger.info(f"Creating ImageSource from STAR file at path {filepath}")

        self.filepath = filepath
        self.data_folder = data_folder
        self.B = B
        self.n_workers = n_workers
        self.max_rows = max_rows

        metadata = self.populate_metadata()

        n = len(metadata["__mrc_filepath"])
        if n == 0:
            raise RuntimeError("No mrcs files found for starfile!")

        # Peek into the first image and populate some attributes
        first_mrc_filepath = metadata["__mrc_filepath"][0]
        mrc = mrcfile.open(first_mrc_filepath, mode="r", permissive=True)

        # Get the 'mode' (data type) - TODO: There's probably a more direct way to do this.
        mode = int(mrc.header.mode)
        mrc_dtypes = {0: "int8", 1: "int16", 2: "float32", 6: "uint16"}
        assert (
            mode in mrc_dtypes
        ), f"Only modes={list(mrc_dtypes.keys())} in MRC files are supported for now."

        mrc_dtype = mrc_dtypes[mode]
        # Potentially over ride the inferred data type.
        if dtype is not None and dtype != np.dtype(mrc_dtype):
            logger.warning(
                f"Overriding MRC datatype {mrc_dtype} with user supplied {dtype}."
            )
        elif dtype is None:
            dtype = mrc_dtype

        shape = mrc.data.shape
        # the code below  accounts for the case where the first MRCS image in the STAR file has one image
        # in that case, the shape will be (resolution, resolution), whereas this code expects
        # (1, resolution, resolution). below, the shape is expanded to accomodate this expectation
        if len(shape) == 2:
            shape = (1,) + shape

        assert shape[1] == shape[2], "Only square images are supported"
        L = shape[1]
        logger.debug(f"Image size = {L}x{L}")

        # Save original image resolution that we expect to use when we start reading actual data
        self._original_resolution = L

        ImageSource.__init__(
            self,
            L=L,
            n=n,
            dtype=dtype,
            metadata=metadata,
            symmetry_group=symmetry_group,
            memory=memory,
            pixel_size=pixel_size,
        )

        # Ensure Relion >= 3.1 convention for offsets
        offset_keys = ["_rlnOriginX", "_rlnOriginY"]
        if self.has_metadata(offset_keys):
            # The setter will store offsets as _rlnOriginX(Y)Angst in metadata
            self.offsets = np.atleast_2d(self.get_metadata(offset_keys))
            # Remove old convention from metadata
            for key in offset_keys:
                del self._metadata[key]

        # CTF estimation parameters coming from Relion
        CTF_params = [
            "_rlnVoltage",
            "_rlnDefocusU",
            "_rlnDefocusV",
            "_rlnDefocusAngle",
            "_rlnSphericalAberration",
            "_rlnAmplitudeContrast",
        ]
        # If these all exist in the STAR file, we may create CTF filters for the source
        if set(CTF_params).issubset(metadata.keys()):
            # partition particles according to unique CTF parameters
            ctf_data = np.stack([metadata[k] for k in CTF_params]).T
            filter_params, filter_indices = np.unique(
                ctf_data,
                return_inverse=True,
                axis=0,
            )
            filters = []
            # for each unique CTF configuration, create a CTFFilter object
            for row in filter_params:
                filters.append(
                    CTFFilter(
                        voltage=row[0],
                        defocus_u=row[1],
                        defocus_v=row[2],
                        defocus_ang=row[3] * np.pi / 180,  # degrees to radians
                        Cs=row[4],
                        alpha=row[5],
                        B=B,
                    )
                )
            self.unique_filters = filters
            # filter_indices stores, for each particle index, the index in
            # self.unique_filters of the filter that should be applied
            self.filter_indices = filter_indices

        # We have provided some, but not all the required params
        elif any(param in metadata for param in CTF_params):
            logger.warning(
                f"Found partially populated CTF Params."
                f"  To automatically populate CTFFilters provide {CTF_params}"
            )

        # If no CTF info in STAR, we initialize the filter values of metadata with default values
        else:
            self.unique_filters = [IdentityFilter()]
            self.filter_indices = np.zeros(self.n, dtype=int)

        logger.info(f"Populated {self.n_ctf_filters} CTFFilters from '{filepath}'")

        # Any further operations should not mutate this instance.
        self._mutable = False

    def populate_metadata(self):
        """
        Relion STAR files may contain a large number of metadata columns in addition
        to the locations of particles. We read this into a dict and add some of
        our own columns for convenience.
        """
        if self.data_folder is not None:
            if not os.path.isabs(self.data_folder):
                self.data_folder = os.path.join(
                    os.path.dirname(self.filepath), self.data_folder
                )
        else:
            self.data_folder = os.path.dirname(self.filepath)

        metadata = RelionStarFile(self.filepath).get_merged_data_block()

        # particle locations are stored as e.g. '000001@first_micrograph.mrcs'
        # in the _rlnImageName column. here, we're splitting this information
        # so we can get the particle's index in the .mrcs stack as an int
        indices_filenames = [s.split("@") for s in metadata["_rlnImageName"]]
        # __mrc_index corresponds to the integer index of the particle in the __mrc_filename stack
        # Note that this is 1-based indexing
        metadata["__mrc_index"] = np.array([int(s[0]) for s in indices_filenames])
        metadata["__mrc_filename"] = np.array([s[1] for s in indices_filenames])

        # Adding a full-filepath field to the Dataframe helps us save time later
        # Note that os.path.join works as expected when the second argument is an absolute path itself
        metadata["__mrc_filepath"] = np.array(
            [os.path.join(self.data_folder, p) for p in metadata["__mrc_filename"]]
        )

        # finally, chop off the metadata df at max_rows
        if self.max_rows is None:
            return metadata
        else:
            max_rows = min(self.max_rows, len(metadata["__mrc_filepath"]))
            return {k: v[:max_rows] for k, v in metadata.items()}

    def __str__(self):
        return f"RelionSource ({self.n} images of size {self.L}x{self.L})"

    def _images(self, indices):
        """
        Returns particle images when accessed via the `ImageSource.images` property.
        Loads particle images corresponding to `indices` from StarFile and .mrcs stacks.

        :param indices: A 1-D NumPy array of integer indices.
        :return: An `Image` object.
        """

        # check for cached images first
        if self._cached_im is not None:
            logger.debug("Loading images from cache")
            return self.generation_pipeline.forward(
                self._cached_im[indices, :, :], indices
            )

        logger.debug(f"Loading {len(indices)} images from STAR file")
        # Log the indices in case needed to debug a crash
        logger.debug(f"Indices: {indices}")

        # Array to hold requested data
        im = np.empty(
            (len(indices), self._original_resolution, self._original_resolution),
            dtype=self.dtype,
        )

        # Map the requested source indices to individual filename and
        # the indices inside that file.  Returns an iterable of tuples
        # (f_name, f_idx, req_idx), ...
        file_requests = self._decompose_source_request(indices)

        # sort `file_requests` by (fname, f_idx) for the upcoming groupby
        file_requests = sorted(file_requests, key=itemgetter(0, 1))

        # Group requested images by filename,
        # and for each file, load the requested `images`.
        for fpath, grp in groupby(file_requests, key=itemgetter(0)):
            # unpack from iter of rows to columns,
            #   also dropping fname column via slice.
            f_indices, req_indices = islice(zip(*grp), 1, None)

            # Load the relevant `f_indices` images from `fpath`
            # and assign to batch array `im`.
            #   Converts tuple of tuples `req_indices` to list.
            im[list(req_indices)] = _load_single_mrcs(fpath, f_indices)

        logger.debug(f"Loading {len(indices)} images complete")

        # Finally, apply transforms to resulting Image
        return self.generation_pipeline.forward(
            Image(im, pixel_size=self.pixel_size), indices
        )

    def _decompose_source_request(self, indices):
        """
        Given requested `indices`, lookup the corresponding image
        files and indexing within that file from metadata.
        Return as an iterable of tuples.

        :param indices: Iterable containing source indices.
        :return: [(filepath, file_index, req_index),...]
        """

        return zip(
            self._metadata["__mrc_filepath"][indices],
            self._metadata["__mrc_index"][indices]
            - 1,  # convert from STAR one based to zero based
            range(len(indices)),
        )


def _optimize_contiguous_slice(inds):
    """
    Given an iterable of indices `inds`,
    determine if `inds` is a contiguous slice,
    and in that case return an equivalent `slice` object,
    otherwise return `inds` as an array.

    :param inds: iterable of indices
    :return: slice or array
    """
    inds = np.array(inds, dtype=int)

    head, tail = inds[0], inds[-1]
    # Short circuit incorrect size, then test if a contiguous range
    if (len(inds) == (tail + 1) - head) and (inds == np.arange(head, tail + 1)).all():
        # coniguous range, replace with a slice
        inds = slice(head, tail + 1)
    return inds


def _load_single_mrcs(filepath, mrc_indices):
    """
    Local utility to wrap up loading a slice of MRC data.

    :param filepath: String filepath to MRC file.
    :param indices: Requested indices from MRC file.
    :return: Slice of array data (as mmap).
    """

    # Attempt to optimize `f_indices` as a slice.
    mrc_indices = _optimize_contiguous_slice(mrc_indices)

    with mrcfile.mmap(filepath, mode="r", permissive=True) as fh:
        arr = fh.data
        # if the stack only contains one image, arr will have shape (resolution, resolution)
        # the code below reshapes it to (1, resolution, resolution)
        if len(arr.shape) == 2:
            arr = arr.reshape((1,) + arr.shape)
        data = arr[mrc_indices, :, :]

    return data
