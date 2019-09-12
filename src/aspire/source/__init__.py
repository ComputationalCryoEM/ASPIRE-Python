import os.path
import logging
import numpy as np
import pandas as pd

from aspire.image import Image
from aspire.volume import im_backproject, vol_project
from aspire.utils.filters import ScalarFilter
from aspire.estimation.noise import WhiteNoiseEstimator
from aspire.utils import ensure
from aspire.utils.coor_trans import grid_2d, angles_to_rots, rots_to_angles
from aspire.utils.matlab_compat import randn
from aspire.io.starfile import Starfile, StarfileBlock

logger = logging.getLogger(__name__)


class ImageSource:

    # ----------------------------------------------------
    # Class Attributes that can be overridden by subclasses
    # ----------------------------------------------------

    # Optional renaming of metadata fields, used
    # These are used ONLY during serialization/deserialization (to/from Starfiles, for example).
    _metadata_aliases = {
        '_image_name':  '_rlnImageName',
        '_offset_x':    '_rlnOriginX',
        '_offset_y':    '_rlnOriginY',
        '_state':       '_rlnClassNumber',
        '_angle_0':     '_rlnAngleRot',
        '_angle_1':     '_rlnAngleTilt',
        '_angle_2':     '_rlnAnglePsi',
        '_amplitude':   '_amplitude',
        '_voltage':     '_rlnVoltage',
        '_defocus_u':   '_rlnDefocusU',
        '_defocus_v':   '_rlnDefocusV',
        '_defocus_ang': '_rlnDefocusAngle',
        '_Cs':          '_rlnSphericalAberration',
        '_alpha':       '_rlnAmplitudeContrast'
    }

    # All metadata fields are strings by default, specify any overrides here.
    # Use the renamed metadata field name (i.e the 'values' in _metadata_aliases) to specify these.
    _metadata_types = {}
    # ----------------------------------------------------

    def __init__(self, L, n, dtype='double', metadata=None):
        """
        A Cryo-EM Source object that supplies images along with other parameters for image manipulation.

        :param L: resolution of (square) images (int)
        :param n: The total number of images available
            Note that images() may return a different number of images based on it's arguments.
        :param metadata: A Dataframe of metadata information corresponding to this ImageSource's images
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

    @property
    def states(self):
        return self.get_metadata('_state')

    @states.setter
    def states(self, values):
        return self.set_metadata('_state', values)

    @property
    def filters(self):
        return self.get_metadata('filter')

    @filters.setter
    def filters(self, values):
        self.set_metadata('filter', values)
        new_values = np.array([(
            getattr(f, 'voltage', np.nan),
            getattr(f, 'defocus_u', np.nan),
            getattr(f, 'defocus_v', np.nan),
            getattr(f, 'defocus_ang', np.nan),
            getattr(f, 'Cs', np.nan),
            getattr(f, 'alpha', np.nan)
        ) for f in values])

        self.set_metadata(
            ['_voltage', '_defocus_u', '_defocus_v', '_defocus_ang', '_Cs', '_alpha'],
            new_values
        )

    @property
    def offsets(self):
        return self.get_metadata(['_offset_x', '_offset_y'])

    @offsets.setter
    def offsets(self, values):
        return self.set_metadata(['_offset_x', '_offset_y'], values)

    @property
    def amplitudes(self):
        return self.get_metadata('_amplitude')

    @amplitudes.setter
    def amplitudes(self, values):
        return self.set_metadata('_amplitude', values)

    @property
    def angles(self):
        return self.get_metadata(['_angle_0', '_angle_1', '_angle_2'])

    @angles.setter
    def angles(self, values):
        self.set_metadata(['_angle_0', '_angle_1', '_angle_2'], values)
        self.rots = angles_to_rots(values)

    @property
    def rots(self):
        return angles_to_rots(self.angles * np.pi / 180)

    @rots.setter
    def rots(self, values):
        angles = rots_to_angles(values) * 180 / np.pi
        self.set_metadata(['_angle_0', '_angle_1', '_angle_2'], angles)

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
                self._metadata.update(df)

    def get_metadata(self, metadata_fields, indices=None):
        """
        Get metadata field information of this ImageSource for selected indices
        :param metadata_fields: A string, of list of strings, representing the metadata field(s) to be queried.
        :param indices: A list of 0-based indices indicating the indices for which to get metadata.
            If indices is None, then
        :return: An ndarray of values (any valid np types) representing metadata info.
        """
        if indices is None:
            indices = self._metadata.index.values
        return self._metadata.loc[indices, metadata_fields].to_numpy()

    def _images(self, start=0, num=np.inf, indices=None):
        """
        Return images WITHOUT applying any filters/translations/rotations/amplitude corrections/noise
        Subclasses may want to implement their own caching mechanisms.
        :param start: start index of image
        :param num: number of images to return
        :param indices: A numpy array of image indices. If specified, start and num are ignored.
        :return: A 3d volume of images of size L x L x n
        """
        raise NotImplementedError('Subclasses should implement this and return a 3d ndarray')

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

    def eval_filter_grid(self, L):
        grid2d = grid_2d(L)
        omega = np.pi * np.vstack((grid2d['x'].flatten(), grid2d['y'].flatten()))

        h = np.empty((omega.shape[-1], len(self.filters)))
        for f in set(self.filters):
            idx_k = np.where(self.filters == f)[0]
            if len(idx_k) > 0:
                h[:, idx_k] = np.column_stack((f.evaluate(omega),) * len(idx_k))

        h = np.reshape(h, grid2d['x'].shape + (len(self.filters),))
        return h

    def cache(self, im=None):
        logger.info('Caching source images')
        if im is None:
            im = self.images()
        self._im = im

    def images(self, start=0, num=np.inf, indices=None, apply_noise=False):
        if indices is None:
            indices = np.arange(start, min(start + num, self.n))

        if self._im is not None:
            im = Image(self._im[:, :, indices])
        else:
            im = Image(self._images(start=start, num=num, indices=indices))

        if apply_noise:
            im += self._noise_images(start=start, num=num, indices=indices)
        return im

    def _noise_images(self, start=0, num=np.inf, indices=None, noise_seed=0, noise_filter=None):
        if indices is None:
            indices = np.arange(start, min(start + num, self.n))

        if noise_filter is None:
            noise_filter = ScalarFilter(value=1, power=0.5)

        im = np.zeros((self.L, self.L, len(indices)), dtype=self.dtype)

        for idx in indices:
            random_seed = noise_seed + 191*(idx+1)
            im_s = randn(2*self.L, 2*self.L, seed=random_seed)
            im_s = Image(im_s).filter(noise_filter)[:, :, 0]
            im_s = im_s[:self.L, :self.L]

            im[:, :, idx-start] = im_s

        return Image(im)

    def set_max_resolution(self, max_L):
        ensure(max_L <= self.L, "Max desired resolution should be less than the current resolution")
        logger.info(f'Setting max. resolution of source = {max_L}')
        self.L = max_L

        ds_factor = self._L / max_L
        self.filters = [f.scale(ds_factor) for f in self.filters]
        self.offsets /= ds_factor

        # Invalidate images
        self._im = None

    def whiten(self, whiten_filter=None):
        """
        Modify the Source object in place by whitening + caching all images, and adding the appropriate whitening
            filter to all available filters.
        :param whiten_filter: Whitening filter to apply. If None, determined automatically.
        :return: On return, the Source object has been modified in place.
        """
        logger.info("Whitening source object")
        if whiten_filter is None:
            logger.info('Determining Whitening Filter')
            whiten_filter = WhiteNoiseEstimator(self).filter
            whiten_filter.power = -0.5

        # Get source images and cache the whitened images
        logger.info('Getting all images')
        images = self.images()
        logger.debug("Applying whitening filter to all images and caching")
        whitened_images = Image(images[:, :, :]).filter(whiten_filter)
        self.cache(whitened_images)

        # TODO: Multiplying every row of self.filters (which may have references to a handful of unique Filter objects)
        # will end up creating self.n unique Filter objects, most of which will be identical !
        self.filters = [f * whiten_filter for f in self.filters]

    def im_backward(self, im, start):
        """
        Apply adjoint mapping to set of images
        :param im: An L-by-L-by-n array of images to which we wish to apply the adjoint of the forward model.
        :param start: Start index of image to consider
        :return: An L-by-L-by-L volume containing the sum of the adjoint mappings applied to the start+num-1 images.
        """
        if im.ndim < 3:
            im = im[:, :, np.newaxis]
        num = im.shape[-1]

        all_idx = np.arange(start, min(start + num, self.n))
        im *= np.broadcast_to(self.amplitudes[all_idx], (self.L, self.L, len(all_idx)))

        im = Image(im).shift(-self.offsets[all_idx, :]).asnumpy()

        im = self.eval_filters(im, start=start, num=num)

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

        im = self.eval_filters(im, start, num)

        im = Image(im).shift(self.offsets[all_idx, :]).asnumpy()

        im *= np.broadcast_to(self.amplitudes[all_idx], (self.L, self.L, len(all_idx)))

        return im

    def to_starfile(self, mrcs_filename):
        df = self._metadata.copy()
        df['_image_name'] = pd.Series(['{0:06}@{1}'.format(i+1, mrcs_filename) for i in range(self.n)])
        df = df.rename(self._metadata_aliases, axis=1)
        df = df.drop([str(col) for col in df.columns if not col.startswith('_')], axis=1)

        return Starfile(blocks=[StarfileBlock(loops=[df])])

    def save(self, starfile_filepath, overwrite=True):
        # TODO: Support optional start/num parameters
        mrcs_filename = os.path.splitext(os.path.basename(starfile_filepath))[0] + '.mrcs'
        mrcs_filepath = os.path.join(
            os.path.dirname(starfile_filepath),
            mrcs_filename
        )

        self.images().save(mrcs_filepath, overwrite=overwrite)
        self.to_starfile(mrcs_filename).save(starfile_filepath, overwrite=overwrite)
