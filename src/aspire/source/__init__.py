import os.path
from copy import copy
import logging
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from aspire.image import Image
from aspire.volume import im_backproject, vol_project
from aspire.utils import ensure
from aspire.utils.filters import MultiplicativeFilter
from aspire.utils.coor_trans import grid_2d
from aspire.io.starfile import StarFile, StarFileBlock
from aspire.source.xform import Multiply, Shift, DownSample, FilterXform, IndexedXform, Pipeline
from aspire.estimation.noise import WhiteNoiseEstimator

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

    def __init__(self, L, n, dtype='double', metadata=None, memory=None):
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
        self.dtype = dtype

        # The private attribute '_im' can be cached by calling this object's cache() method explicitly
        self._im = None

        if metadata is None:
            self._metadata = pd.DataFrame([], index=pd.RangeIndex(self.n))
        else:
            self._metadata = metadata
            self._rotations = R.from_euler('ZYZ', self.get_metadata(['_rlnAngleRot', '_rlnAngleTilt', '_rlnAnglePsi']), degrees=True)

        self.model_pipeline = Pipeline(xforms=None, memory=memory)
        self.generation_pipeline = Pipeline(xforms=None, memory=memory)

    def init_model_pipeline(self):
        if self.filters is not None:
            unique_filters = list(set(self.filters))
            self.model_pipeline.add_xform(
                IndexedXform(
                    [FilterXform(f, resolution=self.L) for f in unique_filters],
                    indices=[unique_filters.index(f) for f in self.filters]
                )
            )

        self.model_pipeline.add_xforms([
            Shift(self.offsets, resolution=self.L),
            Multiply(self.amplitudes, resolution=self.L)
        ])

    @property
    def states(self):
        return self.get_metadata('_rlnClassNumber')

    @states.setter
    def states(self, values):
        return self.set_metadata('_rlnClassNumber', values)

    @property
    def filters(self):
        return self.get_metadata('__filter')

    @filters.setter
    def filters(self, values):
        self.set_metadata('__filter', values)
        if values is None:
            new_values = np.nan
        else:
            new_values = np.array([(
                getattr(f, 'voltage', np.nan),
                getattr(f, 'defocus_u', np.nan),
                getattr(f, 'defocus_v', np.nan),
                getattr(f, 'defocus_ang', np.nan),
                getattr(f, 'Cs', np.nan),
                getattr(f, 'alpha', np.nan)
            ) for f in values])

        self.set_metadata(
            ['_rlnVoltage', '_rlnDefocusU', '_rlnDefocusV', '_rlnDefocusAngle', '_rlnSphericalAberration', '_rlnAmplitudeContrast'],
            new_values
        )

    @property
    def filter_indices(self):
        return self.get_metadata('__filter_indices')

    @property
    def offsets(self):
        return self.get_metadata(['_rlnOriginX', '_rlnOriginY'], default_value=0.)

    @offsets.setter
    def offsets(self, values):
        return self.set_metadata(['_rlnOriginX', '_rlnOriginY'], values)

    @property
    def amplitudes(self):
        return self.get_metadata('_rlnAmplitude', default_value=1.)

    @amplitudes.setter
    def amplitudes(self, values):
        return self.set_metadata('_rlnAmplitude', values)

    @property
    def angles(self):
        """
        :return: Rotation angles in radians, as a n x 3 array
        """
        return self._rotations.as_euler()

    @property
    def rots(self):
        """
        :return: Rotation matrices as a n x 3 x 3 array
        """
        return self._rotations.as_dcm()

    @angles.setter
    def angles(self, values):
        """
        Set rotation angles
        :param values: Rotation angles in radians, as a n x 3 array
        :return: None
        """
        self._rotations = R.from_euler('ZYZ', values)
        self.set_metadata(['_rlnAngleRot', '_rlnAngleTilt', '_rlnAnglePsi'], np.rad2deg(values))

    @rots.setter
    def rots(self, values):
        """
        Set rotation matrices
        :param values: Rotation matrices as a n x 3 x 3 array
        :return: None
        """
        self._rotations = R.from_dcm(values)
        self.set_metadata(['_rlnAngleRot', '_rlnAngleTilt', '_rlnAnglePsi'], self._rotations.as_euler('ZYZ', degrees=True))

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
                self._metadata = self._metadata.merge(series, how='left', left_index=True, right_index=True)
            else:
                self._metadata[metadata_field] = series

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
        missing_columns = [col for col in metadata_fields if col not in self._metadata.columns]
        if len(missing_columns) == 0:
            result = self._metadata.loc[indices, metadata_fields]
        else:
            if default_value is not None:
                right = pd.DataFrame(default_value, columns=missing_columns, index=indices)
                found_columns = [col for col in metadata_fields if col not in missing_columns]
                if len(found_columns) > 0:
                    left = self._metadata.loc[indices, found_columns]
                    result = left.join(right)
                else:
                    result = right
            else:
                raise RuntimeError('Missing columns and no default value provided')

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
        raise NotImplementedError('Subclasses should implement this and return an Image object')

    def group_by(self, by):
        for by_value, df in self._metadata.groupby(by, sort=False):
            yield by_value, self.images(indices=df.index.values)

    def eval_filters(self, im_orig, start=0, num=np.inf, indices=None):
        im = im_orig.copy()
        if indices is None:
            indices = np.arange(start, min(start + num, self.n))

        unique_filters = set(self.filters)
        for f in unique_filters:
            idx_k = np.where(self.filters[indices] == f)[0]
            if len(idx_k) > 0:
                im[:, :, idx_k] = Image(im[:, :, idx_k]).filter(f).asnumpy()

        return im

    def eval_filter_grid(self, L, power=1):
        grid2d = grid_2d(L)
        omega = np.pi * np.vstack((grid2d['x'].flatten(), grid2d['y'].flatten()))

        h = np.empty((omega.shape[-1], len(self.filters)))
        for f in set(self.filters):
            idx_k = np.where(self.filters == f)[0]
            if len(idx_k) > 0:
                filter_values = f.evaluate(omega)
                if power != 1:
                    filter_values **= power
                h[:, idx_k] = np.column_stack((filter_values,) * len(idx_k))

        h = np.reshape(h, grid2d['x'].shape + (len(self.filters),))

        return h

    def cache(self, im=None):
        logger.info('Caching source images')
        if im is None:
            im = self.images()
        self._im = im

    def images(self, start=0, num=np.inf, indices=None, batch_size=256, *args, **kwargs):
        if indices is None:
            indices = np.arange(start, min(start + num, self.n))

        if self._im is not None:
            logger.info(f'Loading images from cache')
            im = Image(self._im[:, :, indices])
        else:
            resolution = min(self.generation_pipeline.resolution, self.L)
            im = np.empty((resolution, resolution, len(indices)), dtype=self.dtype)
            for i in range(0, len(indices), batch_size):
                chunk_range = np.arange(i, min(i + batch_size, len(indices)))
                chunk_indices = indices[chunk_range]
                _im = self._images(start=np.min(chunk_indices), indices=chunk_indices, *args, **kwargs)
                _im = self.generation_pipeline.forward(_im, indices=chunk_indices)
                im[:, :, chunk_range] = _im.asnumpy()
            im = Image(im)

        logger.info(f'Loaded {len(indices)} images')
        return im

    def downsample(self, L):
        ensure(L <= self.L, "Max desired resolution should be less than the current resolution")
        logger.info(f'Setting max. resolution of source = {L}')

        self.model_pipeline.downsample(resolution=L)
        self.generation_pipeline.add_xform(DownSample(resolution=L))

        self.L = L
        # Invalidate images
        self._im = None

    def whiten(self, whiten_filter=None):
        """
        Modify the Source object in place by whitening + caching all images, and adding the appropriate whitening
            filter to all available filters.
        :param whiten_filter: Whitening filter to apply.
        :return: On return, the Source object has been modified in place.
        """
        logger.info("Whitening source object")

        if whiten_filter is None:
            logger.info('Determining Whitening Filter through a WhiteNoiseEstimator')
            whiten_filter = WhiteNoiseEstimator(self).filter
            whiten_filter.power = -0.5
        else:
            whiten_filter = copy(whiten_filter)
            whiten_filter.power = -0.5
        self.generation_pipeline.add_xform(FilterXform(whiten_filter))

        unique_filters = set(self.filters)
        for f in unique_filters:
            f_new = copy(f)
            f.__class__ = MultiplicativeFilter
            f.__init__(f_new, whiten_filter)

    def im_backward(self, im, start):
        """
        Apply adjoint mapping to set of images
        :param im: An L-by-L-by-n array of images to which we wish to apply the adjoint of the forward model.
        :param start: Start index of image to consider
        :return: An L-by-L-by-L volume containing the sum of the adjoint mappings applied to the start+num-1 images.
        """
        num = im.shape[-1]

        all_idx = np.arange(start, min(start + num, self.n))
        im = self.model_pipeline.adjoint(im, indices=all_idx).asnumpy()
        vol = im_backproject(im, self.rots[start:start+num, :, :])

        return vol

    def vol_forward(self, vol, start, num):
        """
        Apply forward image model to volume
        :param vol: A volume of size L-by-L-by-L.
        :param start: Start index of image to consider
        :param num: Number of images to consider
        :return: The images obtained from volume by projecting, applying CTFs, translating, and multiplying by the
            amplitude.
        """
        all_idx = np.arange(start, min(start + num, self.n))
        im = vol_project(vol, self.rots[all_idx, :, :])

        im = self.model_pipeline.forward(Image(im), indices=all_idx)

        return im

    def save(self, starfile_filepath, batch_size=1024, overwrite=False):

        df = self._metadata.copy()
        # Drop any column that doesn't start with a *single* underscore
        df = df.drop([str(col) for col in df.columns if not col.startswith('_') or col.startswith('__')], axis=1)

        # Create a new column that we will be populating in the loop below
        df['_rlnImageName'] = ''

        with open(starfile_filepath, 'w') as f:
            for i_start in np.arange(0, self.n, batch_size):

                i_end = min(self.n, i_start + batch_size)
                num = i_end - i_start

                mrcs_filename = os.path.splitext(os.path.basename(starfile_filepath))[0] + f'_{i_start}_{i_end}.mrcs'
                mrcs_filepath = os.path.join(
                    os.path.dirname(starfile_filepath),
                    mrcs_filename
                )

                logger.info(f'Saving ImageSource[{i_start}-{i_end}] to {mrcs_filepath}')
                im = self.images(start=i_start, num=num)
                im.save(mrcs_filepath, overwrite=overwrite)

                df['_rlnImageName'][i_start: i_end] = pd.Series(['{0:06}@{1}'.format(j + 1, mrcs_filepath) for j in range(num)])

            starfile = StarFile(blocks=[StarFileBlock(loops=[df])])
            starfile.save(f)


class ArrayImageSource(ImageSource):
    """
    An `ImageSource` object that holds a reference to an underlying `Image` object (a thin wrapper on an ndarray)
    representing images. It does not produce its images on the fly, but keeps them in memory. As such, it should not be
    used where large Image objects are involved, but can be used in situations where API conformity is desired.
    """
    def __init__(self, im, metadata=None):
        """
        Initialize from an `Image` object
        :param im: An `Image` object representing image data served up by this `ImageSource`
        :param metadata: A Dataframe of metadata information corresponding to this ImageSource's images
        """
        super().__init__(L=im.res, n=im.n_images, dtype=im.dtype, metadata=metadata, memory=None)
        self._im = im

    def _images(self, start=0, num=np.inf, indices=None):
        if indices is None:
            indices = np.arange(start, min(start + num, self.n))
        return self._im[indices]
