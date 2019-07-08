import logging
from collections import OrderedDict
import os.path
import numpy as np
import pandas as pd
import mrcfile
from tqdm import tqdm
from concurrent import futures
from multiprocessing import cpu_count

from aspyre import config
from aspyre.source import SourceFilter
from aspyre.source.micrograph import Micrograph
from aspyre.imaging.filters import CTFFilter
from aspyre.utils import ensure
from aspyre.utils.math import angles_to_rots
from aspyre.imaging import im_downsample

logger = logging.getLogger(__name__)


"""
Mapping from column name to data type. Leading underscores need not be included here.
Any column names not found here are defaulted to a string type.
"""
COLUMN_TYPES = {
    'rlnVoltage': float,
    'rlnDefocusU': float,
    'rlnDefocusV': float,
    'rlnDefocusAngle': float,
    'rlnSphericalAberration': float,
    'rlnDetectorPixelSize': float,
    'rlnCtfFigureOfMerit': float,
    'rlnMagnification': float,
    'rlnAmplitudeContrast': float,
    'rlnImageName': str,
    'rlnOriginalName': str,
    'rlnCtfImage': str,
    'rlnCoordinateX': float,
    'rlnCoordinateY': float,
    'rlnCoordinateZ': float,
    'rlnNormCorrection': float,
    'rlnMicrographName': str,
    'rlnGroupName': str,
    'rlnGroupNumber': str,
    'rlnOriginX': float,
    'rlnOriginY': float,
    'rlnAngleRot': float,
    'rlnAngleTilt': float,
    'rlnAnglePsi': float,
    'rlnClassNumber': int,
    'rlnLogLikeliContribution': float,
    'rlnRandomSubset': int,
    'rlnParticleName': str,
    'rlnOriginalParticleName': str,
    'rlnNrOfSignificantSamples': float,
    'rlnNrOfFrames': int,
    'rlnMaxValueProbDistribution': float
}


class Starfile(Micrograph):
    @staticmethod
    def star2df(filepath):
        dirpath = os.path.dirname(filepath)
        columns = OrderedDict()
        skiprows = 0
        with open(filepath, 'r') as f:
            for line in f.readlines():
                if line.startswith('_rln'):
                    name = line.split(' ')[0][1:]
                    columns[name] = COLUMN_TYPES.get(name, str)
                    skiprows += 1
                elif len(columns) > 0:
                    break
                else:
                    skiprows += 1

        df = pd.read_csv(filepath, delim_whitespace=True, skiprows=skiprows, names=columns, dtype=columns)

        # Add calculated fields
        if 'rlnDefocusU' not in df:
            df['rlnDefocusU'] = np.nan

        if 'rlnDefocusV' not in df:
            df['rlnDefocusV'] = df['rlnDefocusU']
            df['rlnDefocusAngle'] = 0.

        if 'rlnAngleRot' not in df:
            df['rlnAngleRot'] = np.nan
            df['rlnAngleTilt'] = np.nan
            df['rlnAnglePsi'] = np.nan

        if 'rlnOriginX' not in df:
            df['rlnOriginX'] = np.nan
            df['rlnOriginY'] = np.nan

        if 'rlnClassNumber' not in df:
            df['rlnClassNumber'] = np.nan

        # Columns representing angles in radians
        df['_rlnAngleRot_radians'] = (df['rlnAngleRot'] / 180) * np.pi
        df['_rlnAngleTilt_radians'] = (df['rlnAngleTilt'] / 180) * np.pi
        df['_rlnAnglePsi_radians'] = (df['rlnAnglePsi'] / 180) * np.pi
        df['_rlnDefocusAngle_radians'] = (df['rlnDefocusAngle'] / 180) * np.pi

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

    def __init__(self, filepath, pixel_size=1, B=0, ignore_missing_files=False, max_rows=None):
        """
        Load starfile at given filepath
        :param filepath: Absolute or relative path to .star file
        :param pixel_size: the pixel size of the images in angstroms (Default 1)
        :param B: the envelope decay of the CTF in inverse square angstrom (Default 0)
        :param ignore_missing_files: Whether to ignore missing MRC files or not (Default False)
        :param max_rows: Maximum no. of rows in .star file to read. If None (default), all rows are read.
            Note that this refers to the max no. of images to load, not the max. number of .mrcs files (which may be
            equal to or less than the no. of images).
            If ignore_missing_files is False, the first max_rows rows read from the .star file are considered.
            If ignore_missing_files is True, then the first max_rows *available* rows from the .star file are
            considered.
        """
        logger.debug(f'Loading starfile at path {filepath}')

        self.df = Starfile.star2df(filepath)

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

        self.pixel_size = pixel_size
        self.B = B

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

        rots = angles_to_rots(
            self.df[['_rlnAngleRot_radians', '_rlnAngleTilt_radians', '_rlnAnglePsi_radians']].values.T
        )

        filter_params, filter_indices = np.unique(
            self.df[[
                'rlnVoltage',
                'rlnDefocusU',
                'rlnDefocusV',
                '_rlnDefocusAngle_radians',
                'rlnSphericalAberration',
                'rlnAmplitudeContrast'
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
                    defocus_ang=row[3],
                    Cs=row[4],
                    alpha=row[5],
                    B=self.B
                )
            )
        filters = SourceFilter(filters, indices=filter_indices)

        offsets = self.df[['rlnOriginX', 'rlnOriginY']].values.T
        amplitudes = np.ones(n)
        states = self.df['rlnClassNumber'].values

        super().__init__(
            L=L,
            n=n,
            states=states,
            filters=filters,
            offsets=offsets,
            amplitudes=amplitudes,
            rots=rots,
            dtype=dtype
        )

    def __str__(self):
        return f'Starfile ({self.n} images of size {self._L}x{self._L})'

    def _images(self, start=0, num=None):

        def load_single_mrcs(filepath, df):
            arr = mrcfile.open(filepath).data
            data = arr[df['_mrc_index'] - 1, :, :].T

            if self.L < self._L:
                data = im_downsample(data, self.L)

            return df.index, data

        n_workers = config.starfile.n_workers
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

        return im

    def save(self, folder, starfile_name=None):
        """
        Save micrograph files to folder with a given starfile name
        :param folder:
        :return:
        """
        pass
