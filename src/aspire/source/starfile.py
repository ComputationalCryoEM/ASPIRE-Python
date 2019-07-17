from collections import OrderedDict
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
from aspire.image import ImageStack
from aspire.image import im_downsample

logger = logging.getLogger(__name__)


class StarfileStack(ImageSource):

    """
    Mapping from column name to data type. Leading underscores need not be included here.
    Any column names not found here are defaulted to a string type.
    """
    column_mappings = {}

    @staticmethod
    def star2df(filepath, column_mappings):
        dirpath = os.path.dirname(filepath)
        columns = OrderedDict()
        skiprows = 0
        with open(filepath, 'r') as f:
            for line in f.readlines():
                if line.startswith('_rln'):
                    name = line.split(' ')[0][1:]
                    columns[name] = column_mappings.get(name, str)
                    skiprows += 1
                elif len(columns) > 0:
                    break
                else:
                    skiprows += 1

        if 'rlnImageName' not in columns:
            raise RuntimeError('Valid starfiles at least need a _rlnImageName column specified')

        df = pd.read_csv(filepath, delim_whitespace=True, skiprows=skiprows, names=columns, dtype=columns)

        # TODO: Check behavior if this is a single mrc file (no '@')
        _index, df['_mrc_filename'] = df['rlnImageName'].str.split('@', 1).str
        df['_mrc_index'] = pd.to_numeric(_index)

        # Adding a full-filepath field to the Dataframe helps us save time later
        # Note that os.path.join works as expected when the second argument is an absolute path itself
        df['_mrc_filepath'] = df['_mrc_filename'].apply(lambda filename: os.path.join(dirpath, filename))
        df['_mrc_found'] = df['_mrc_filepath'].apply(lambda filepath: os.path.exists(filepath))

        msg = f'Read starfile with {len(df)} records'
        n_missing = sum(df['_mrc_found'] == False)  # nopep8
        if n_missing > 0:
            msg += f' ({n_missing} files missing)'
            logger.warning(msg)
        else:
            logger.info(msg)

        return df

    def __init__(self, filepath, n_workers=-1, ignore_missing_files=False, max_rows=None):
        """
        Load starfile at given filepath
        :param filepath: Absolute or relative path to .star file
        :param ignore_missing_files: Whether to ignore missing MRC files or not (Default False)
        :param max_rows: Maximum no. of rows in .star file to read. If None (default), all rows are read.
            Note that this refers to the max no. of images to load, not the max. number of .mrcs files (which may be
            equal to or less than the no. of images).
            If ignore_missing_files is False, the first max_rows rows read from the .star file are considered.
            If ignore_missing_files is True, then the first max_rows *available* rows from the .star file are
            considered.
        """
        logger.debug(f'Loading starfile at path {filepath}')

        self.n_workers = n_workers
        self.df = StarfileStack.star2df(filepath, self.column_mappings)
        self.df = self.add_metadata()

        # Handle missing files
        missing = self.df['_mrc_found'] == False  # nopep8
        n_missing = sum(missing)
        if ignore_missing_files:
            if n_missing > 0:
                logger.info(f'Dropping {n_missing} rows with missing mrc files')
                self.df = self.df[~missing]
                self.df.reset_index(inplace=True)
        else:
            ensure(n_missing == 0, f'{n_missing} mrc files missing')

        if max_rows is not None:
            self.df = self.df.iloc[:max_rows]

        n = len(self.df)

        # Peek into the first image and populate some attributes
        first_mrc_filepath = self.df.iloc[0]._mrc_filepath
        mrc = mrcfile.open(first_mrc_filepath)

        # Get the 'mode' (data type) - TODO: There's probably a more direct way to do this.
        mode = int(mrc.header.mode)
        dtypes = {0: 'int8', 1: 'int16', 2: 'float32', 6: 'uint16'}
        ensure(mode in dtypes, f'Only modes={list(dtypes.keys())} in MRC files are supported for now.')
        dtype = dtypes[mode]

        shape = mrc.data.shape
        ensure(shape[1] == shape[2], "Only square images are supported")
        L = shape[1]

        # Save original image resolution
        self._L = L

        logger.debug(f'Image size = {L}x{L}')

        ImageSource.__init__(self, L=L, n=n, dtype=dtype)

    def __str__(self):
        return f'Starfile ({self.n} images of size {self.L}x{self.L})'

    def add_metadata(self):
        """
        Modify the self.df DataFrame to add any calculated columns/change data types.
        This base class implementation of add_metadata() doesn't modify the DataFrame at all.

        :return: A modified DataFrame with calculated columns added.
        """
        return self.df

    def _images(self, start=0, num=None):

        def load_single_mrcs(filepath, df):
            arr = mrcfile.open(filepath).data
            data = arr[df['_mrc_index'] - 1, :, :].T

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

        df = self.df[start:num]
        im = np.empty((self.L, self.L, num))

        groups = df.groupby('_mrc_filepath')
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

        return ImageStack(im)
