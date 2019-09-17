import os.path
import logging
import pandas as pd
import numpy as np
import mrcfile
from concurrent import futures
from multiprocessing import cpu_count

from aspire.utils import ensure
from aspire.source import ImageSource
from aspire.image import Image
from aspire.io.starfile import StarFile
from aspire.utils.filters import CTFFilter

logger = logging.getLogger(__name__)


class RelionSource(ImageSource):

    _metadata_types = {
        '_rlnVoltage': float,
        '_rlnDefocusU': float,
        '_rlnDefocusV': float,
        '_rlnDefocusAngle': float,
        '_rlnSphericalAberration': float,
        '_rlnDetectorPixelSize': float,
        '_rlnCtfFigureOfMerit': float,
        '_rlnMagnification': float,
        '_rlnAmplitudeContrast': float,
        '_rlnImageName': str,
        '_rlnOriginalName': str,
        '_rlnCtfImage': str,
        '_rlnCoordinateX': float,
        '_rlnCoordinateY': float,
        '_rlnCoordinateZ': float,
        '_rlnNormCorrection': float,
        '_rlnMicrographName': str,
        '_rlnGroupName': str,
        '_rlnGroupNumber': str,
        '_rlnOriginX': float,
        '_rlnOriginY': float,
        '_rlnAngleRot': float,
        '_rlnAngleTilt': float,
        '_rlnAnglePsi': float,
        '_rlnClassNumber': int,
        '_rlnLogLikeliContribution': float,
        '_rlnRandomSubset': int,
        '_rlnParticleName': str,
        '_rlnOriginalParticleName': str,
        '_rlnNrOfSignificantSamples': float,
        '_rlnNrOfFrames': int,
        '_rlnMaxValueProbDistribution': float
    }

    @classmethod
    def starfile2df(cls, filepath, max_rows=np.inf):

        dirpath = os.path.dirname(filepath)

        # Note: Valid Relion image "_data.star" files have to have their data in the first loop of the first block.
        # We thus index our StarFile class with [0][0].
        df = StarFile(filepath)[0][0]
        column_types = {name: cls._metadata_types.get(name, str) for name in df.columns}
        df = df.astype(column_types)

        # Rename fields to standard notation
        reverse_metadata_aliases = {v: k for k, v in cls._metadata_aliases.items()}

        df = df.rename(reverse_metadata_aliases, axis=1)
        _index, df['__mrc_filename'] = df['_image_name'].str.split('@', 1).str
        df['__mrc_index'] = pd.to_numeric(_index)

        # Adding a full-filepath field to the Dataframe helps us save time later
        # Note that os.path.join works as expected when the second argument is an absolute path itself
        df['__mrc_filepath'] = df['__mrc_filename'].apply(lambda filename: os.path.join(dirpath, filename))

        return df.iloc[:max_rows]

    def __init__(self, filepath, pixel_size=1, B=0, n_workers=-1, max_rows=np.inf):
        """
        Load STAR file at given filepath
        :param filepath: Absolute or relative path to STAR file
        :param pixel_size: the pixel size of the images in angstroms (Default 1)
        :param B: the envelope decay of the CTF in inverse square angstrom (Default 0)
        :param n_workers: Number of threads to spawn to read referenced .mrcs files (Default -1 to auto detect)
        :param max_rows: Maximum number of rows in STAR file to read.
            Note that this refers to the max number of images to load, not the max. number of .mrcs files (which may be
            equal to or less than the number of images).
        """
        logger.debug(f'Creating ImageSource from starfile at path {filepath}')

        self.pixel_size = pixel_size
        self.B = B
        self.n_workers = n_workers

        metadata = self.__class__.starfile2df(filepath, max_rows)

        n = len(metadata)
        if n == 0:
            raise RuntimeError('No mrcs files found for starfile!')

        # Peek into the first image and populate some attributes
        first_mrc_filepath = metadata.loc[0]['__mrc_filepath']
        mrc = mrcfile.open(first_mrc_filepath)

        # Get the 'mode' (data type) - TODO: There's probably a more direct way to do this.
        mode = int(mrc.header.mode)
        dtypes = {0: 'int8', 1: 'int16', 2: 'float32', 6: 'uint16'}
        ensure(mode in dtypes, f'Only modes={list(dtypes.keys())} in MRC files are supported for now.')
        dtype = dtypes[mode]

        shape = mrc.data.shape
        ensure(shape[1] == shape[2], "Only square images are supported")
        L = shape[1]
        logger.debug(f'Image size = {L}x{L}')

        # Save original image resolution
        self._L = L

        filter_params, filter_indices = np.unique(
            metadata[[
                '_voltage',
                '_defocus_u',
                '_defocus_v',
                '_defocus_ang',
                '_Cs',
                '_alpha'
            ]].values,
            return_inverse=True,
            axis=0
        )

        filters = []
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
                    B=B
                )
            )

        metadata['filter'] = [filters[i] for i in filter_indices]
        # TODO: Is there an amplitude field in Relion?
        metadata['_amplitude'] = 1.0

        ImageSource.__init__(
            self,
            L=L,
            n=n,
            dtype=dtype,
            metadata=metadata
        )

    def __str__(self):
        return f'RelionSource ({self.n} images of size {self.L}x{self.L})'

    def _images(self, start=0, num=np.inf, indices=None):
        if indices is None:
            indices = np.arange(start, min(start + num, self.n))
        logger.info(f'Loading {len(indices)} images from STAR file')

        def load_single_mrcs(filepath, df):
            arr = mrcfile.open(filepath).data
            data = arr[df['__mrc_index'] - 1, :, :].T

            if self.L < self._L:
                data = Image(data).downsample(self.L).asnumpy()

            return df.index, data

        n_workers = self.n_workers
        if n_workers < 0:
            n_workers = cpu_count() - 1

        df = self._metadata.loc[indices]
        im = np.empty((self.L, self.L, len(indices)))

        groups = df.groupby('__mrc_filepath')
        n_workers = min(n_workers, len(groups))

        with futures.ThreadPoolExecutor(n_workers) as executor:
            to_do = []
            for filepath, _df in groups:
                future = executor.submit(load_single_mrcs, filepath, _df)
                to_do.append(future)

            for future in futures.as_completed(to_do):
                indices, data = future.result()
                im[:, :, indices] = data

        logger.info(f'Loading {len(indices)} images complete')

        return Image(im)
