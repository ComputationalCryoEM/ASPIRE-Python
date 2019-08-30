import os.path
import logging
import pandas as pd
import numpy as np
import mrcfile
from tqdm import tqdm
from concurrent import futures
from multiprocessing import cpu_count

from aspire.utils import ensure
from aspire.source import ImageSource
from aspire.image import Image
from aspire.image import im_downsample
from aspire.io.starfile import Starfile
from aspire.utils.filters import CTFFilter

logger = logging.getLogger(__name__)


class StarfileStack(ImageSource):

    @classmethod
    def starfile2df(cls, filepath, block_index_or_name=0, loop_index=0, ignore_missing_files=False, max_rows=None):

        dirpath = os.path.dirname(filepath)
        df = Starfile(filepath)[block_index_or_name][loop_index].data
        column_types = {name: cls._metadata_types.get(name, str) for name in df.columns}
        df = df.astype(column_types)

        # Rename fields to standard notation
        reverse_metadata_aliases = {v: k for k, v in cls._metadata_aliases.items()}

        df = df.rename(reverse_metadata_aliases, axis=1)

        # TODO: Check behavior if this is a single mrc file (no '@')
        _index, df['__mrc_filename'] = df['_image_name'].str.split('@', 1).str
        df['__mrc_index'] = pd.to_numeric(_index)

        # Adding a full-filepath field to the Dataframe helps us save time later
        # Note that os.path.join works as expected when the second argument is an absolute path itself
        df['__mrc_filepath'] = df['__mrc_filename'].apply(lambda filename: os.path.join(dirpath, filename))

        # Helper function to check if a file exists
        def file_exists(filepath):
            return os.path.exists(filepath)

        if max_rows is not None:
            # Keep track of how many mrc files we've found
            df['__mrc_found'] = False
            max_rows = min(max_rows, len(df))
            # Build up our data in chunks of max_rows rows
            for i in range(0, len(df), max_rows):
                _range = np.arange(i, i + max_rows)
                df.loc[_range, '__mrc_found'] = df.loc[_range]['__mrc_filepath'].apply(file_exists)
                # Do we have at least max_rows rows? No need to continue
                if sum(df['__mrc_found'] == True) >= max_rows:  # nopep8
                    break
        else:
            max_rows = len(df)
            df['__mrc_found'] = df['__mrc_filepath'].apply(file_exists)

        missing = df['__mrc_found'] == False  # nopep8
        n_missing = sum(missing)
        if ignore_missing_files:
            if n_missing > 0:
                logger.info(f'Dropping {n_missing} rows with missing mrc files')
                df = df[~missing]
                df.reset_index(inplace=True)
        else:
            ensure(n_missing == 0, f'{n_missing} mrc files missing')

        return df.iloc[:max_rows]

    def __init__(self, filepath, pixel_size=1, B=0, n_workers=-1, block_index_or_name=0, loop_index=0,
                 ignore_missing_files=False, max_rows=None):
        """
        Load starfile at given filepath
        :param filepath: Absolute or relative path to .star file
        :param pixel_size: the pixel size of the images in angstroms (Default 1)
        :param B: the envelope decay of the CTF in inverse square angstrom (Default 0)
        :param n_workers: No. of threads to spawn to read referenced .mrcs files (Default -1 to auto detect)
        :param block_index_or_name: An integer specifying the block index (0-indexed), of a string specifying
            the block name
        :param loop_index: An integer specifying the loop index (0-indexed)
        :param ignore_missing_files: Whether to ignore missing MRC files or not (Default False)
        :param max_rows: Maximum no. of rows in .star file to read. If None (default), all rows are read.
            Note that this refers to the max no. of images to load, not the max. number of .mrcs files (which may be
            equal to or less than the no. of images).
            If ignore_missing_files is False, the first max_rows rows read from the .star file are considered.
            If ignore_missing_files is True, then the first max_rows *available* rows from the .star file are
            considered.
        """
        logger.debug(f'Creating ImageSource from starfile at path {filepath}')

        self.pixel_size = pixel_size
        self.B = B
        self.n_workers = n_workers

        metadata = self.__class__.starfile2df(filepath, block_index_or_name, loop_index, ignore_missing_files, max_rows)

        n = len(metadata)
        if n == 0:
            raise RuntimeError('No mrcs files found for starfile!')

        # Peek into the first image and populate some attributes
        first_mrc_filepath = metadata.iloc[0]['__mrc_filepath']
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

        metadata['__filter'] = [filters[i] for i in filter_indices]

        ImageSource.__init__(
            self,
            L=L,
            n=n,
            dtype=dtype,
            metadata=metadata
        )

    def __str__(self):
        return f'Starfile ({self.n} images of size {self.L}x{self.L})'

    def _images(self, start=0, num=None):

        def load_single_mrcs(filepath, df):
            arr = mrcfile.open(filepath).data
            data = arr[df['__mrc_index'] - 1, :, :].T

            if self.L < self._L:
                data = im_downsample(data, self.L)

            return df.index, data

        n_workers = self.n_workers
        if n_workers < 0:
            n_workers = cpu_count() - 1

        if num is None:
            num = self.n - start
        else:
            num = min(self.n - start, num)

        df = self._metadata[start:num]
        im = np.empty((self.L, self.L, num))

        groups = df.groupby('__mrc_filepath')
        n_workers = min(n_workers, len(groups))

        pbar = tqdm(total=self.n)
        with futures.ThreadPoolExecutor(n_workers) as executor:
            to_do = []
            for filepath, _df in groups:
                future = executor.submit(load_single_mrcs, filepath, _df)
                to_do.append(future)

            for future in futures.as_completed(to_do):
                indices, data = future.result()
                im[:, :, indices] = data
                pbar.update(len(indices))
        pbar.close()

        return im
