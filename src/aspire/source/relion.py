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

    @classmethod
    def starfile2df(cls, proj_folder, filepath, max_rows=None):
        star_folder = proj_folder
        if proj_folder is not None:
            if not os.path.isabs(filepath):
                star_folder = os.path.join(proj_folder, os.path.dirname(filepath))
        else:
            star_folder = os.path.dirname(filepath)
        starfile_path = os.path.join(star_folder, os.path.basename(filepath))
        # Note: Valid Relion image "_data.star" files have to have their data in the first loop of the first block.
        # We thus index our StarFile class with [0][0].
        df = StarFile(starfile_path)[0][0]
        column_types = {name: cls.metadata_fields.get(name, str) for name in df.columns}
        df = df.astype(column_types)

        # TODO: The statement need to be reimplemented in future version of pandas
        _index, df['__mrc_filename'] = df['_rlnImageName'].str.split('@', 1).str
        df['__mrc_index'] = pd.to_numeric(_index)

        # Adding a full-filepath field to the Dataframe helps us save time later
        # Note that os.path.join works as expected when the second argument is an absolute path itself
        df['__mrc_filepath'] = df['__mrc_filename'].apply(lambda filename: os.path.join(proj_folder, filename))

        if max_rows is None:
            return df
        else:
            return df.iloc[:max_rows]

    def __init__(self, proj_folder, filepath, pixel_size=1, B=0, n_workers=-1, max_rows=None, memory=None):
        """
        Load STAR file at given filepath
        :param proj_folder: Absolute path to project folder with respect to all relative paths to
            star files for different analysis such as denoising and 2D classification.
            If None, the folder corresponding to filepath is used.
        :param filepath: include relative path to project folder and input STAR file
        :param pixel_size: the pixel size of the images in angstroms (Default 1)
        :param B: the envelope decay of the CTF in inverse square angstrom (Default 0)
        :param n_workers: Number of threads to spawn to read referenced .mrcs files (Default -1 to auto detect)
        :param max_rows: Maximum number of rows in STAR file to read. If None, all rows are read.
            Note that this refers to the max number of images to load, not the max. number of .mrcs files (which may be
            equal to or less than the number of images).
        :param memory: str or None
            The path of the base directory to use as a data store or None. If None is given, no caching is performed.
        """
        logger.debug(f'Creating ImageSource from STAR file at path {filepath}')

        self.pixel_size = pixel_size
        self.B = B
        self.n_workers = n_workers
        self.proj_folder = proj_folder
        self.max_rows = max_rows

        metadata = self.__class__.starfile2df(proj_folder, filepath, max_rows)

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

        # Save original image resolution that we expect to use when we start reading actual data
        self._original_resolution = L

        filter_params, filter_indices = np.unique(
            metadata[[
                '_rlnVoltage',
                '_rlnDefocusU',
                '_rlnDefocusV',
                '_rlnDefocusAngle',
                '_rlnSphericalAberration',
                '_rlnAmplitudeContrast'
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

        metadata['__filter'] = [filters[i] for i in filter_indices]
        metadata['__filter_indices'] = filter_indices

        ImageSource.__init__(
            self,
            L=L,
            n=n,
            dtype=dtype,
            metadata=metadata,
            memory=memory
        )

    def __str__(self):
        return f'RelionSource ({self.n} images of size {self.L}x{self.L})'

    def _images(self, start=0, num=np.inf, indices=None):
        if indices is None:
            indices = np.arange(start, min(start + num, self.n))
        else:
            start = indices.min()
        logger.info(f'Loading {len(indices)} images from STAR file')

        def load_single_mrcs(filepath, df):
            arr = mrcfile.open(filepath).data
            data = arr[df['__mrc_index'] - 1, :, :].T

            return df.index, data

        n_workers = self.n_workers
        if n_workers < 0:
            n_workers = cpu_count() - 1

        df = self._metadata.loc[indices]
        im = np.empty((self._original_resolution, self._original_resolution, len(indices)))

        groups = df.groupby('__mrc_filepath')
        n_workers = min(n_workers, len(groups))

        with futures.ThreadPoolExecutor(n_workers) as executor:
            to_do = []
            for filepath, _df in groups:
                future = executor.submit(load_single_mrcs, filepath, _df)
                to_do.append(future)

            for future in futures.as_completed(to_do):
                data_indices, data = future.result()
                im[:, :, data_indices-start] = data

        logger.info(f'Loading {len(indices)} images complete')

        return Image(im)
