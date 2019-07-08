import logging
import numpy as np

from aspyre.imaging import im_filter, im_translate
from aspyre.imaging.threed import im_backproject, vol_project
from aspyre.imaging.filters import IdentityFilter, ScalarFilter
from aspyre.estimation.noise import WhiteNoiseEstimator
from aspyre.utils import ensure
from aspyre.utils.math import grid_2d
from aspyre.utils.matlab_compat import m_reshape, randn, randi

logger = logging.getLogger(__name__)


# TODO: The SourceFilter class shouldn't exist!
# The assignment of filters to Images should happen inside the ImageSource object!
class SourceFilter:
    """
    An object representing Filter objects and their assignments to a particular Source object
    """
    def __init__(self, filters, indices=None, n=None):
        """
        :param filters: An iterable of Filter objects.
        :param indices: An iterable of indices representing the 0-indexed indices of an image stack
            on which to apply the filters. If unspecified, `n` must be supplied, and individual filters are applied
            randomly.
        :param n: An integer representing the depth of the image stack on which this SourceFilter is applied.
            Not needed if `indices` are supplied.
        """
        if indices is None:
            ensure(n is not None, "Either indices or n must be supplied for a SourceFilter")
            # Assign filters randomly.
            # For Matlab compatibility, randi returns numbers in the range [1, iMax] decrement by one for our purposes
            indices = randi(len(filters), n, seed=0) - 1
        else:
            ensure(n is None, "Cannot supply both indices and n for a SourceFilter")
            n = len(indices)

        self.filters = filters
        self.indices = indices
        self.n = n

    def __str__(self):
        return f'SourceFilter ({len(self.filters)} filters, {self.n} images)'

    def __call__(self, im, start=0, num=None):
        ensure(im.ndim == 3, "A SourceFilter can only be called for a 3d volume representing a stack of images")

        end = self.n
        if num is not None:
            end = min(start + num, self.n)
        all_idx = np.arange(start, end)

        unique_filters = np.unique(self.indices[all_idx]).astype('int')
        for k in unique_filters:
            idx_k = np.where(self.indices[all_idx] == k)[0]
            im[:, :, idx_k] = im_filter(im[:, :, idx_k], self.filters[k])
        return im

    def evaluate(self, omega, *args, **kwargs):
        return np.column_stack([f.evaluate(omega, *args, **kwargs) for f in self.filters])

    def evaluate_grid(self, L, *args, **kwargs):
        # Todo: remove redundancy wrt a single Filter's evaluate_grid
        grid2d = grid_2d(L)
        omega = np.pi * np.vstack((grid2d['x'].flatten('F'), grid2d['y'].flatten('F')))
        h = self.evaluate(omega, *args, **kwargs)

        h = m_reshape(h, grid2d['x'].shape + (len(self.filters),))

        return h

    def scale(self, c):
        logger.info(f'Scaling SourceFilter by factor {c}')
        for f in self.filters:
            f.scale(c)


class ImageSource:
    def __init__(self, L, n, states=None, filters=None, offsets=None, amplitudes=None, rots=None, dtype='double'):
        """
        A Cryo-EM Source object that supplies images along with other parameters for image manipulation.

        :param L: resolution of (square) images (int)
        :param n: The total no. of images available
            Note that images() may return a different no. of images based on it's arguments.
        :param states: A 1-by-n array containing the state (label) for each image (1-indexed)
        :param filters: A SourceFilter object
        :param offsets: ndarray of shape (2, n) specifying the shifts of the images
        :param amplitudes: ndarray of shape (n,) specifying the amplitude multipliers of the images
        :param rots:
        :param dtype: A string representing a valid numpy dtype (typically 'single' or 'double')
            TODO: Elaborate
        """
        if filters is None:
            filters = SourceFilter([IdentityFilter()], n=n)
        self.L = L
        self.n = n
        self.states = states
        self.filters = filters
        self.offsets = offsets
        self.amplitudes = amplitudes
        self.rots = rots
        self.dtype = dtype

        # The private attribute '_im' can be cached by calling this object's cache() method explicitly
        self._im = None

    def _images(self, start=0, num=None):
        """
        Return images WITHOUT applying any filters/translations/rotations/amplitude corrections/noise
        Subclasses may want to implement their own caching mechanisms.
        :param start: start index of image
        :param num: number of images to return
        :return: A 3d volume of images of size L x L x n

        TODO: _images should return and instance of the Image class going forward
        """
        raise NotImplementedError('Subclasses should implement this!')

    def cache(self, im=None):
        logger.info('Caching source images')
        if im is None:
            im = self.images()
        self._im = im

    def images(self, start=0, num=None, apply_noise=False):
        if self._im is not None:
            end = self.n
            if num is not None:
                end = min(start + num, self.n)
            im = self._im[:, :, start:end]
        else:
            im = self._images(start, num)

        if apply_noise:
            im += self._noise_images(start, num)
        return im

    def _noise_images(self, start=0, num=None, noise_seed=0, noise_filter=None):
        # Generate noisy images in interval [start, start+num-1] (a total of 'num' images)

        end = self.n
        if num is not None:
            end = min(start + num, self.n)
        all_idx = np.arange(start, end)

        if noise_filter is None:
            noise_filter = ScalarFilter(value=1, power=0.5)

        im = np.zeros((self.L, self.L, len(all_idx)), dtype=self.dtype)

        for idx in all_idx:
            random_seed = noise_seed + 191*(idx+1)
            im_s = randn(2*self.L, 2*self.L, seed=random_seed)
            im_s = im_filter(im_s, noise_filter)
            im_s = im_s[:self.L, :self.L]

            im[:, :, idx-start] = im_s

        return im

    def set_max_resolution(self, max_L):
        ensure(max_L <= self.L, "Max desired resolution should be less than the current resolution")
        self.L = max_L

        ds_factor = self._L / max_L
        self.filters.scale(ds_factor)
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
        logger.debug("Whitening source object")
        if whiten_filter is None:
            logger.info('Determining Whitening Filter')
            whiten_filter = WhiteNoiseEstimator(self).filter
            whiten_filter.power = -0.5

        # Create a whitening SourceFilter object that applies to all available images
        whiten_source_filter = SourceFilter(
            [whiten_filter],
            indices=np.zeros(self.n).astype('int')
        )
        # Get source images and cache the whitened images
        logger.debug("Getting all images")
        images = self.images()
        logger.debug("Applying whitening filter to all images and caching")
        whitened_images = whiten_source_filter(images)
        self.cache(whitened_images)

        # Modify this Source's SourceFilter
        # TODO: Add ability to multiply a SourceFilter object with a Filter object to avoid attribute access below
        self.filters = SourceFilter(
            [f * whiten_filter for f in self.filters.filters],
            indices=self.filters.indices
        )

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

        im = im_translate(im, -self.offsets[:, all_idx])

        im = self.filters(im, start=start, num=num)

        vol = im_backproject(im, self.rots[:, :, start:start+num])

        return vol

    def vol_forward(self, vol, start, num):
        """
        Apply forward imaging model to volume
        :param vol: A volume of size L-by-L-by-L.
        :param start: Start index of image to consider
        :param num: No. of images to consider
        :return: The images obtained from volume by projecting, applying CTFs, translating, and multiplying by the
            amplitude.
        """
        all_idx = np.arange(start, min(start + num, self.n))
        im = vol_project(vol, self.rots[:, :, all_idx])

        im = self.filters(im, start, num)

        im = im_translate(im, self.offsets[:, all_idx])
        im *= np.broadcast_to(self.amplitudes[all_idx], (self.L, self.L, len(all_idx)))

        return im
