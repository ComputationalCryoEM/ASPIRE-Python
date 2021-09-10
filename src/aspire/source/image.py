import logging
import os.path

import mrcfile
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from aspire.image import Image, normalize_bg
from aspire.image.xform import (
    Downsample,
    FilterXform,
    IndexedXform,
    LambdaXform,
    Multiply,
    Pipeline,
)
from aspire.operators import (
    IdentityFilter,
    LambdaFilter,
    MultiplicativeFilter,
    PowerFilter,
)
from aspire.storage import MrcStats, StarFile, StarFileBlock
from aspire.utils import ensure
from aspire.utils.coor_trans import grid_2d

logger = logging.getLogger(__name__)


class ImageSource:
    """
    When creating an `ImageSource` object, a 'metadata' table holds metadata information about all images in the
    `ImageSource`. The number of rows in this metadata table will equal the total number of images supported by this
    `ImageSource` (available as the 'n' attribute), though reading/writing of images is usually done in chunks.

    This metadata table is implemented as a pandas `DataFrame`.

    The 'values' in this metadata table are usually primitive types (floats/ints/strings) that are suitable
    for being read from STAR files, and being written to STAR files. The columns corresponding to these fields
    begin with a single underscore '_'.

    In addition, the metadata table may also contain references to Python objects.
    `Filter` objects, for example, are stored in this metadata table as references to unique `Filter` objects that
    correspond to images in this `ImageSource`. Several rows of metadata may end up containing a reference to a small
    handful of unique `Filter` objects, depending on the values found in other columns (identical `Filter`
    objects, depending on unique CTF values found for _rlnDefocusU/_rlnDefocusV etc.
    """

    """
    The metadata_fields dictionary below specifies default data types of certain key fields used in the codebase.
    The STAR file used to initialize subclasses of ImageSource may well contain other columns not found below; these
    additional columns are available when read, and they default to the pandas data type 'object'.
    """
    metadata_fields = {
        "_rlnVoltage": float,
        "_rlnDefocusU": float,
        "_rlnDefocusV": float,
        "_rlnDefocusAngle": float,
        "_rlnSphericalAberration": float,
        "_rlnDetectorPixelSize": float,
        "_rlnCtfFigureOfMerit": float,
        "_rlnMagnification": float,
        "_rlnAmplitudeContrast": float,
        "_rlnImageName": str,
        "_rlnOriginalName": str,
        "_rlnCtfImage": str,
        "_rlnCoordinateX": float,
        "_rlnCoordinateY": float,
        "_rlnCoordinateZ": float,
        "_rlnNormCorrection": float,
        "_rlnMicrographName": str,
        "_rlnGroupName": str,
        "_rlnGroupNumber": str,
        "_rlnOriginX": float,
        "_rlnOriginY": float,
        "_rlnAngleRot": float,
        "_rlnAngleTilt": float,
        "_rlnAnglePsi": float,
        "_rlnClassNumber": int,
        "_rlnLogLikeliContribution": float,
        "_rlnRandomSubset": int,
        "_rlnParticleName": str,
        "_rlnOriginalParticleName": str,
        "_rlnNrOfSignificantSamples": float,
        "_rlnNrOfFrames": int,
        "_rlnMaxValueProbDistribution": float,
    }

    def __init__(self, L, n, dtype="double", metadata=None, memory=None):
        """
        A Cryo-EM ImageSource object that supplies images along with other parameters for image manipulation.

        :param L: resolution of (square) images (int)
        :param n: The total number of images available
            Note that images() may return a different number of images based on its arguments.
        :param metadata: A Dataframe of metadata information corresponding to this ImageSource's images
        :param memory: str or None
            The path of the base directory to use as a data store or None. If None is given, no caching is performed.
        """
        self.L = L
        self.n = n
        self.dtype = np.dtype(dtype)

        # The private attribute '_cached_im' can be populated by calling this object's cache() method explicitly
        self._cached_im = None

        if metadata is None:
            self._metadata = pd.DataFrame([], index=pd.RangeIndex(self.n))
        else:
            self._metadata = metadata
            if self.has_metadata(["_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"]):
                self._rotations = R.from_euler(
                    "ZYZ",
                    self.get_metadata(
                        ["_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"]
                    ),
                    degrees=True,
                )

        self.unique_filters = []
        self.generation_pipeline = Pipeline(xforms=None, memory=memory)
        self._metadata_out = None
        # _rotations is assigned non None value
        #  by `rots` or `angles` setters.
        #  It is potentially used by sublasses to test if we've used setters.
        self._rotations = None

    @property
    def states(self):
        return np.atleast_1d(self.get_metadata("_rlnClassNumber"))

    @states.setter
    def states(self, values):
        return self.set_metadata("_rlnClassNumber", values)

    @property
    def filter_indices(self):
        return self.get_metadata("__filter_indices")

    @filter_indices.setter
    def filter_indices(self, indices):
        # create metadata of filters for all images
        if indices is None:
            filter_values = np.nan
        else:
            attribute_list = (
                "voltage",
                "defocus_u",
                "defocus_v",
                "defocus_ang",
                "Cs",
                "alpha",
            )
            filter_values = np.zeros((len(indices), len(attribute_list)))
            for i, filt in enumerate(self.unique_filters):
                filter_values[indices == i] = [
                    getattr(filt, attribute, np.nan) for attribute in attribute_list
                ]

        self.set_metadata(
            [
                "_rlnVoltage",
                "_rlnDefocusU",
                "_rlnDefocusV",
                "_rlnDefocusAngle",
                "_rlnSphericalAberration",
                "_rlnAmplitudeContrast",
            ],
            filter_values,
        )
        return self.set_metadata(["__filter_indices"], indices)

    @property
    def offsets(self):
        return np.atleast_2d(
            self.get_metadata(["_rlnOriginX", "_rlnOriginY"], default_value=0.0)
        )

    @offsets.setter
    def offsets(self, values):
        return self.set_metadata(["_rlnOriginX", "_rlnOriginY"], values)

    @property
    def amplitudes(self):
        return np.atleast_1d(self.get_metadata("_rlnAmplitude", default_value=1.0))

    @amplitudes.setter
    def amplitudes(self, values):
        return self.set_metadata("_rlnAmplitude", values)

    @property
    def angles(self):
        """
        :return: Rotation angles in radians, as a n x 3 array
        """
        # Call a private method. This allows sub classes to effeciently override.
        return self._angles()

    def _angles(self):
        """
        Converts internal _rotations representation to expected matrix form.
        """
        return self._rotations.as_euler("ZYZ", degrees=False).astype(self.dtype)

    @property
    def rots(self):
        """
        :return: Rotation matrices as a n x 3 x 3 array
        """
        # Call a private method. This allows sub classes to effeciently override.
        return self._rots()

    def _rots(self):
        """
        Converts internal `_rotations` representation to expected matrix form.
        :return: Rotation matrices as a n x 3 x 3 array
        """
        return self._rotations.as_matrix().astype(self.dtype)

    @angles.setter
    def angles(self, values):
        """
        Set rotation angles
        :param values: Rotation angles in radians, as a n x 3 array
        :return: None
        """
        self._rotations = R.from_euler("ZYZ", values)
        self.set_metadata(
            ["_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"], np.rad2deg(values)
        )

    @rots.setter
    def rots(self, values):
        """
        Set rotation matrices
        :param values: Rotation matrices as a n x 3 x 3 array
        :return: None
        """
        self._rotations = R.from_matrix(values)
        self.set_metadata(
            ["_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"],
            self._rotations.as_euler("ZYZ", degrees=True),
        )

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
        # Convert a single metadata field into a list of single metadata field, since that's what the 'columns'
        # argument of a DataFrame constructor expects.
        if isinstance(metadata_fields, str):
            metadata_fields = [metadata_fields]

        if indices is None:
            indices = self._metadata.index.values

        df = pd.DataFrame(values, columns=metadata_fields, index=indices)
        for metadata_field in metadata_fields:
            series = df[metadata_field]
            if metadata_field not in self._metadata.columns:
                self._metadata = self._metadata.merge(
                    series, how="left", left_index=True, right_index=True
                )
            else:
                self._metadata[metadata_field] = series

    def has_metadata(self, metadata_fields):
        """
        Find out if one more more metadata fields are available for this `ImageSource`.
        :param metadata_fields: A string, of list of strings, representing the metadata field(s) to be queried.
        :return: Boolean value indicating whether the field(s) are available.
        """
        if isinstance(metadata_fields, str):
            metadata_fields = [metadata_fields]
        return all(f in self._metadata.columns for f in metadata_fields)

    def get_metadata(self, metadata_fields, indices=None, default_value=None):
        """
        Get metadata field information of this ImageSource for selected indices
        :param metadata_fields: A string, of list of strings, representing the metadata field(s) to be queried.
        :param indices: A list of 0-based indices indicating the indices for which to get metadata.
            If indices is None, then values corresponding to all indices in this Source object are returned.
        :param default_value: Default scalar value to use for any fields not found in the metadata. If None,
            no default value is used, and missing field(s) cause a RuntimeError.
        :return: An ndarray of values (any valid np types) representing metadata info.
        """
        if isinstance(metadata_fields, str):
            metadata_fields = [metadata_fields]
        if indices is None:
            indices = self._metadata.index.values

        # The pandas .loc indexer does work with missing columns (as long as not ALL of them are missing)
        # which messes with our logic. This behavior will change in pandas 0.21.0.
        # See https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#indexing-with-list-with-missing-labels-is-deprecated
        # We deal with the situation in a slightly verbose manner as follows.
        missing_columns = [
            col for col in metadata_fields if col not in self._metadata.columns
        ]
        if len(missing_columns) == 0:
            result = self._metadata.loc[indices, metadata_fields]
        else:
            if default_value is not None:
                right = pd.DataFrame(
                    default_value, columns=missing_columns, index=indices
                )
                found_columns = [
                    col for col in metadata_fields if col not in missing_columns
                ]
                if len(found_columns) > 0:
                    left = self._metadata.loc[indices, found_columns]
                    result = left.join(right)
                else:
                    result = right
            else:
                raise RuntimeError("Missing columns and no default value provided")

        return result.to_numpy().squeeze()

    def _images(self, start=0, num=np.inf, indices=None):
        """
        Return images WITHOUT applying any filters/translations/rotations/amplitude corrections/noise
        Subclasses may want to implement their own caching mechanisms.
        :param start: start index of image
        :param num: number of images to return
        :param indices: A numpy array of image indices. If specified, start and num are ignored.
        :return: A 3D volume of images of size L x L x n
        """
        raise NotImplementedError(
            "Subclasses should implement this and return an Image object"
        )

    def eval_filters(self, im_orig, start=0, num=np.inf, indices=None):
        if not isinstance(im_orig, Image):
            logger.warning(
                f"eval_filters passed {type(im_orig)} instead of Image instance"
            )
            # for now just convert it
            im = Image(im_orig)

        im = im_orig.copy()

        if indices is None:
            indices = np.arange(start, min(start + num, self.n))

        for i, filt in enumerate(self.unique_filters):
            idx_k = np.where(self.filter_indices[indices] == i)[0]
            if len(idx_k) > 0:
                im[idx_k] = Image(im[idx_k]).filter(filt).asnumpy()

        return im

    def eval_filter_grid(self, L, power=1):
        grid2d = grid_2d(L, dtype=self.dtype)
        omega = np.pi * np.vstack((grid2d["x"].flatten(), grid2d["y"].flatten()))

        h = np.empty((omega.shape[-1], len(self.filter_indices)), dtype=self.dtype)
        for i, filt in enumerate(self.unique_filters):
            idx_k = np.where(self.filter_indices == i)[0]
            if len(idx_k) > 0:
                filter_values = filt.evaluate(omega)
                if power != 1:
                    filter_values **= power
                h[:, idx_k] = np.column_stack((filter_values,) * len(idx_k))

        h = np.reshape(h, grid2d["x"].shape + (len(self.filter_indices),))

        return h

    def cache(self):
        logger.info("Caching source images")
        self._cached_im = self.images(start=0, num=np.inf)
        self.generation_pipeline.reset()

    def images(self, start, num, *args, **kwargs):
        """
        Return images from this ImageSource as an Image object.
        :param start: The inclusive start index from which to return images.
        :param num: The exclusive end index up to which to return images.
        :param args: Any additional positional arguments to pass on to the `ImageSource`'s underlying `_images` method.
        :param kwargs: Any additional keyword arguments to pass on to the `ImageSource`'s underlying `_images` method.
        :return: an `Image` object.
        """
        indices = np.arange(start, min(start + num, self.n), dtype=int)

        if self._cached_im is not None:
            logger.info("Loading images from cache")
            im = Image(self._cached_im[indices, :, :])
        else:
            im = self._images(indices=indices, *args, **kwargs)

        im = self.generation_pipeline.forward(im, indices=indices)
        logger.info(f"Loaded {len(indices)} images")
        return im

    def downsample(self, L):
        ensure(
            L <= self.L,
            "Max desired resolution should be less than the current resolution",
        )
        logger.info(f"Setting max. resolution of source = {L}")

        self.generation_pipeline.add_xform(Downsample(resolution=L))

        ds_factor = self.L / L
        self.unique_filters = [f.scale(ds_factor) for f in self.unique_filters]
        self.offsets /= ds_factor

        self.L = L

    def whiten(self, noise_filter):
        """
        Modify the `ImageSource` in-place by appending a whitening filter to the generation pipeline.
        :param noise_filter: The noise psd of the images as a `Filter` object. Typically determined by a
            NoiseEstimator class, and available as its `filter` attribute.
        :return: On return, the `ImageSource` object has been modified in place.
        """
        logger.info("Whitening source object")
        whiten_filter = PowerFilter(noise_filter, power=-0.5)

        logger.info("Transforming all CTF Filters into Multiplicative Filters")
        self.unique_filters = [
            MultiplicativeFilter(f, whiten_filter) for f in self.unique_filters
        ]
        logger.info("Adding Whitening Filter Xform to end of generation pipeline")
        self.generation_pipeline.add_xform(FilterXform(whiten_filter))

    def phase_flip(self):
        """
        Perform phase flip to images in the source object using CTF information
        """
        logger.info("Perform phase flip on source object")
        logger.info("Adding Phase Flip Xform to end of generation pipeline")
        unique_xforms = [
            FilterXform(LambdaFilter(f, np.sign)) for f in self.unique_filters
        ]
        self.generation_pipeline.add_xform(
            IndexedXform(unique_xforms, self.filter_indices)
        )

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
        grid = grid_2d(L, shifted=True)
        # Get mask indices of signal and noise samples assuming molecule
        signal_mask = grid["r"] < 0.5
        noise_mask = grid["r"] > 0.8

        # Calculate mean values in batch_size
        signal_mean = 0.0
        noise_mean = 0.0

        for i in range(0, self.n, batch_size):
            images = self.images(i, batch_size).asnumpy()
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

    def im_backward(self, im, start):
        """
        Apply adjoint mapping to set of images
        :param im: An Image instance to which we wish to apply the adjoint of the forward model.
        :param start: Start index of image to consider
        :return: An L-by-L-by-L volume containing the sum of the adjoint mappings applied to the start+num-1 images.
        """
        num = im.n_images

        all_idx = np.arange(start, min(start + num, self.n))
        im *= self.amplitudes[all_idx, np.newaxis, np.newaxis]
        im = im.shift(-self.offsets[all_idx, :])
        im = self.eval_filters(im, start=start, num=num)

        vol = im.backproject(self.rots[start : start + num, :, :])[0]

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
        assert vol.n_vols == 1, "vol_forward expects a single volume, not a stack"

        if vol.dtype != self.dtype:
            logger.warning(f"Volume.dtype {vol.dtype} inconsistent with {self.dtype}")

        im = vol.project(0, self.rots[all_idx, :, :])
        im = self.eval_filters(im, start, num)
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
        Save the output metadata to STAR file and/or images to MRCS file

        :param starfile_filepath: Path to STAR file where we want to
            save metadata of image_source
        :param batch_size: Batch size of images to query.
        :param save_mode: Whether to save all images in a single or multiple files in batch size.
        :param overwrite: Option to overwrite the output MRCS files.
        """
        logger.info("save metadata into STAR file")
        filename_indices = self.save_metadata(
            starfile_filepath,
            new_mrcs=True,
            batch_size=batch_size,
            save_mode=save_mode,
        )

        logger.info("save images into MRCS file")
        self.save_images(
            starfile_filepath,
            filename_indices=filename_indices,
            batch_size=batch_size,
            overwrite=overwrite,
        )

    def save_metadata(
        self, starfile_filepath, new_mrcs=True, batch_size=512, save_mode=None
    ):
        """
        Save updated metadata to a STAR file

        :param starfile_filepath: Path to STAR file where we want to
            save image_source
        :param new_mrcs: Whether to save all images to new MRCS files or not.
            If True, new file names and pathes need to be created.
        :param batch_size: Batch size of images to query from the
            `ImageSource` object. Every `batch_size` rows, entries are
            written to STAR file.
        :param save_mode: Whether to save all images in a single or
            multiple files in batch size.
        :return: None
        """

        df = self._metadata.copy()
        # Drop any column that doesn't start with a *single* underscore
        df = df.drop(
            [
                str(col)
                for col in df.columns
                if not col.startswith("_") or col.startswith("__")
            ],
            axis=1,
        )

        with open(starfile_filepath, "w") as f:
            if new_mrcs:
                # Create a new column that we will be populating in the loop below
                # For
                df["_rlnImageName"] = ""

                if save_mode == "single":
                    # Save all images into one single mrc file
                    fname = os.path.basename(starfile_filepath)
                    fstem = os.path.splitext(fname)[0]
                    mrcs_filename = f"{fstem}_{0}_{self.n-1}.mrcs"

                    # Then set name in dataframe for the StarFile
                    # Note, here the row_indexer is :, representing all rows in this data frame.
                    #   df.loc will be reponsible for dereferencing and assigning values to df.
                    #   Pandas will assert df.shape[0] == self.n
                    df.loc[:, "_rlnImageName"] = [
                        f"{j + 1:06}@{mrcs_filename}" for j in range(self.n)
                    ]
                else:
                    # save all images into multiple mrc files in batch size
                    for i_start in np.arange(0, self.n, batch_size):
                        i_end = min(self.n, i_start + batch_size)
                        num = i_end - i_start
                        mrcs_filename = (
                            os.path.splitext(os.path.basename(starfile_filepath))[0]
                            + f"_{i_start}_{i_end-1}.mrcs"
                        )

                        # Note, here the row_indexer is a slice.
                        #   df.loc will be reponsible for dereferencing and assigning values to df.
                        #   Pandas will assert the lnegth of row_indexer equals num.
                        row_indexer = df[i_start:i_end].index
                        df.loc[row_indexer, "_rlnImageName"] = [
                            "{0:06}@{1}".format(j + 1, mrcs_filename)
                            for j in range(num)
                        ]

            filename_indices = df._rlnImageName.str.split(pat="@", expand=True)[
                1
            ].tolist()

            # initial the star file object and save it
            starfile = StarFile(blocks=[StarFileBlock(loops=[df])])
            starfile.save(f)

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
                    num = i_end - i_start
                    logger.info(
                        f"Saving ImageSource[{i_start}-{i_end-1}] to {mrcs_filepath}"
                    )
                    datum = self.images(start=i_start, num=num).data.astype("float32")

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
                num = i_end - i_start

                mrcs_filepath = os.path.join(
                    os.path.dirname(starfile_filepath), filename_indices[i_start]
                )

                logger.info(
                    f"Saving ImageSource[{i_start}-{i_end-1}] to {mrcs_filepath}"
                )
                im = self.images(start=i_start, num=num)
                im.save(mrcs_filepath, overwrite=overwrite)


class ArrayImageSource(ImageSource):
    """
    An `ImageSource` object that holds a reference to an underlying `Image` object (a thin wrapper on an ndarray)
    representing images. It does not produce its images on the fly, but keeps them in memory. As such, it should not be
    used where large Image objects are involved, but can be used in situations where API conformity is desired.

    Note that this class does not provide an `_images` method, since it populates the `_cached_im` attribute which,
    if available, is consulted directly by the parent class, bypassing `_images`.
    """

    def __init__(self, im, metadata=None, angles=None):
        """
        Initialize from an `Image` object
        :param im: An `Image` or Numpy array object representing image data served up by this `ImageSource`.
        In the case of a Numpy array, attempts to create an 'Image' object.
        :param metadata: A Dataframe of metadata information corresponding to this ImageSource's images
        :param angles: Optional n-by-3 array of rotation angles corresponding to `im`.
        """

        if not isinstance(im, Image):
            logger.info("Attempting to create an Image object from Numpy array.")
            try:
                im = Image(im)
            except Exception as e:
                raise RuntimeError(
                    "Creating Image object from Numpy array failed."
                    f" Original error: {str(e)}"
                )

        super().__init__(
            L=im.res, n=im.n_images, dtype=im.dtype, metadata=metadata, memory=None
        )

        self._cached_im = im

        # Create filter indices, these are required to pass unharmed through filter eval code
        #   that is potentially called by other methods later.
        self.filter_indices = np.zeros(self.n)
        self.unique_filters = [IdentityFilter()]

        # Optionally populate angles/rotations.
        if angles is not None:
            if angles.shape != (self.n, 3):
                raise ValueError(f"Angles should be shape {(self.n, 3)}")
            # This will populate `_rotations`,
            #   which is exposed by properties `angles` and `rots`.
            self.angles = angles

    def _rots(self):
        """
        Private method, checks if `_rotations` has been set,
        then returns inherited rots, otherwise raise.
        """

        if self._rotations is not None:
            return super()._rots()
        else:
            raise RuntimeError(
                "Consumer of ArrayImageSource trying to access rots,"
                " but rots were not defined for this source."
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
