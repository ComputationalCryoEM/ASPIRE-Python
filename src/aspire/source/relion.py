import logging
import os.path
from concurrent import futures
from multiprocessing import cpu_count

import mrcfile
import numpy as np
import pandas as pd

from aspire.image import Image
from aspire.operators import CTFFilter, IdentityFilter
from aspire.source import ImageSource
from aspire.storage import StarFile

logger = logging.getLogger(__name__)


class RelionSource(ImageSource):
    """
    A RelionSource represents a source of picked and cropped particles stored as slices in a `.mrcs` stack.
    It must be instantiated via a STAR file, which--at a minumum--lists the particles in each `.mrcs` stack in the
    `_rlnImageName` column. The STAR file may also contain Relion-specific metadata columns. This information
    is read into a Pandas DataFrame table containing a row for each particle specifying its location and
    its metadata. The metadata table may be augmented or modified via helper methods found in ImageSource. It may
    store, for example, Filter objects added during preprocessing.
    """

    # The metadata_fields dictionary below specifies default data types
    # of certain key fields used in the codebase,
    # which are originally read from Relion STAR files.
    relion_metadata_fields = {
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

    def __init__(
        self,
        filepath,
        data_folder=None,
        pixel_size=1,
        B=0,
        n_workers=-1,
        max_rows=None,
        memory=None,
    ):
        """
        Load STAR file at given filepath
        :param filepath: Absolute or relative path to STAR file
        :param data_folder: Path to folder w.r.t which all relative paths to .mrcs files are resolved.
            If None, the folder corresponding to filepath is used.
        :param pixel_size: the pixel size of the images in angstroms (Default 1)
        :param B: the envelope decay of the CTF in inverse square angstrom (Default 0)
        :param n_workers: Number of threads to spawn to read referenced .mrcs files (Default -1 to auto detect)
        :param max_rows: Maximum number of rows in STAR file to read. If None, all rows are read.
            Note that this refers to the max number of images to load, not the max. number of .mrcs files (which may be
            equal to or less than the number of images).
        :param memory: str or None
            The path of the base directory to use as a data store or None. If None is given, no caching is performed.
        """
        logger.debug(f"Creating ImageSource from STAR file at path {filepath}")

        self.pixel_size = pixel_size
        self.B = B
        self.n_workers = n_workers

        metadata = self.populate_metadata(filepath, data_folder, max_rows)

        n = len(metadata)
        if n == 0:
            raise RuntimeError("No mrcs files found for starfile!")

        # Peek into the first image and populate some attributes
        first_mrc_filepath = metadata.loc[0]["__mrc_filepath"]
        mrc = mrcfile.open(first_mrc_filepath)

        # Get the 'mode' (data type) - TODO: There's probably a more direct way to do this.
        mode = int(mrc.header.mode)
        dtypes = {0: "int8", 1: "int16", 2: "float32", 6: "uint16"}
        assert (
            mode in dtypes
        ), f"Only modes={list(dtypes.keys())} in MRC files are supported for now."

        dtype = dtypes[mode]

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
            self, L=L, n=n, dtype=dtype, metadata=metadata, memory=memory
        )

        # CTF estimation parameters coming from Relion
        CTF_params = [
            "_rlnVoltage",
            "_rlnDefocusU",
            "_rlnDefocusV",
            "_rlnDefocusAngle",
            "_rlnSphericalAberration",
            "_rlnAmplitudeContrast",
        ]
        # If these exist in the STAR file, we may create CTF filters for the source
        if set(CTF_params).issubset(metadata.columns):
            # partition particles according to unique CTF parameters
            filter_params, filter_indices = np.unique(
                metadata[CTF_params].values, return_inverse=True, axis=0
            )
            filters = []
            # for each unique CTF configuration, create a CTFFilter object
            for row in filter_params:
                filters.append(
                    CTFFilter(
                        pixel_size=self.pixel_size,
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
        # If no CTF info in STAR, we initialize the filter values of metadata with default values
        else:
            self.unique_filters = [IdentityFilter()]
            self.filter_indices = np.zeros(self.n, dtype=int)

    def populate_metadata(self, filepath, data_folder=None, max_rows=None):
        """
        Relion STAR files may contain a large number of metadata columns in addition
        to the locations of particles. We read this into a Pandas DataFrame and add some of
        our own columns for convenience.
        """
        if data_folder is not None:
            if not os.path.isabs(data_folder):
                data_folder = os.path.join(os.path.dirname(filepath), data_folder)
        else:
            data_folder = os.path.dirname(filepath)

        # Valid Relion STAR files always have their data in the first loop of the first block.
        # We are getting the first (and only) block in this StarFile object
        df = StarFile(filepath).get_block_by_index(0)
        # convert STAR file strings to data type for each field
        # columns without a specified data type are read as dtype=object
        column_types = {
            name: RelionSource.relion_metadata_fields.get(name, str)
            for name in df.columns
        }
        df = df.astype(column_types)

        # particle locations are stored as e.g. '000001@first_micrograph.mrcs'
        # in the _rlnImageName column. here, we're splitting this information
        # so we can get the particle's index in the .mrcs stack as an int
        df[["__mrc_index", "__mrc_filename"]] = df["_rlnImageName"].str.split(
            "@", 1, expand=True
        )
        # __mrc_index corresponds to the integer index of the particle in the __mrc_filename stack
        # Note that this is 1-based indexing
        df["__mrc_index"] = pd.to_numeric(df["__mrc_index"])

        # Adding a full-filepath field to the Dataframe helps us save time later
        # Note that os.path.join works as expected when the second argument is an absolute path itself
        df["__mrc_filepath"] = df["__mrc_filename"].apply(
            lambda filename: os.path.join(data_folder, filename)
        )

        if max_rows is None:
            return df
        else:
            max_rows = min(max_rows, len(df))
            return df.iloc[:max_rows]

    def __str__(self):
        return f"RelionSource ({self.n} images of size {self.L}x{self.L})"

    def _images(self, start=0, num=np.inf, indices=None):
        if indices is None:
            indices = np.arange(start, min(start + num, self.n))
        else:
            start = indices.min()
        logger.info(f"Loading {len(indices)} images from STAR file")

        def load_single_mrcs(filepath, df):
            arr = mrcfile.open(filepath).data
            # if the stack only contains one image, arr will have shape (resolution, resolution)
            # the code below reshapes it to (1, resolution, resolution)
            if len(arr.shape) == 2:
                arr = arr.reshape((1,) + arr.shape)
            # __mrc_index is the 1-based index of the particle in the stack
            data = arr[df["__mrc_index"] - 1, :, :]

            return df.index, data

        n_workers = self.n_workers
        if n_workers < 0:
            n_workers = cpu_count() - 1

        df = self._metadata.loc[indices]
        im = np.empty(
            (len(indices), self._original_resolution, self._original_resolution),
            dtype=self.dtype,
        )

        groups = df.groupby("__mrc_filepath")
        n_workers = min(n_workers, len(groups))

        with futures.ThreadPoolExecutor(n_workers) as executor:
            to_do = []
            for filepath, _df in groups:
                future = executor.submit(load_single_mrcs, filepath, _df)
                to_do.append(future)

            for future in futures.as_completed(to_do):
                data_indices, data = future.result()
                im[data_indices - start] = data

        logger.info(f"Loading {len(indices)} images complete")

        return Image(im)
