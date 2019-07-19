from scipy import misc, signal
import mrcfile
import numpy as np
from aspire.utils.numeric import xp
from aspire.utils import ensure


class Micrograph:
    def __init__(self, filepath, margin=None, shrink_factor=None, square=False, gauss_filter_size=None, gauss_filter_sigma=None):
        self.filepath = filepath
        self.shrink_factor = shrink_factor
        self.square = square
        self.gauss_filter_size = gauss_filter_size
        self.gauss_filter_sigma = gauss_filter_sigma

        # Attributes populated by the time this constructor returns
        # A 2-D ndarray if loading a MRC file, a 3-D ndarray if loading a MRCS file,
        # with the last dimension indicating the no. of images
        self.im = None

        self._init_margins(margin)
        self._read()

    def _init_margins(self, margin):
        if margin is None:
            t = r = b = l = None
        elif isinstance(margin, (tuple, list)):
            ensure(len(margin)==4, 'If specifying margins a a tuple/list, specify the top/right/bottom/left margins.')
            t, r, b, l = margin
        else:  # assume scalar
            t = r = b = l = int(margin)
        self.margin_top, self.margin_right, self.margin_bottom, self.margin_left = t, r, b, l

    def _read(self):
        with mrcfile.open(self.filepath) as mrc:
            im = mrc.data.astype('double')

        # For multiple mrc files, mrcfile returns an ndarray with (shape n_images, height, width)
        # swap axes 0 and 2 so we get the more natural (height, width, n_images)
        if im.ndim == 3:
            im = np.swapaxes(im, 0, 2)

        self.original_im = im

        # Discard outer pixels
        im = im[
            self.margin_top: -self.margin_bottom if self.margin_bottom is not None else None,
            self.margin_left: -self.margin_right if self.margin_right is not None else None
        ]

        if self.square:
            side_length = min(im.shape)
            im = im[:side_length, :side_length]

        if self.shrink_factor is not None:
            im = misc.imresize(im, 1/self.shrink_factor, mode='F', interp='cubic')

        if self.gauss_filter_size is not None:
            im = signal.correlate(
                im,
                Micrograph.gaussian_filter(self.gauss_filter_size, self.gauss_filter_sigma),
                'same'
            )

        self.im = im.astype('double')
        self.shape = im.shape

    @classmethod
    def gaussian_filter(cls, size_filter, std):
        """Computes low-pass filter.

        Args:
            size_filter: Size of filter (size_filter x size_filter).
            std: sigma value in filter.
        """

        y, x = xp.mgrid[-(size_filter - 1) // 2: (size_filter - 1) // 2 + 1,
               -(size_filter - 1) // 2: (size_filter - 1) // 2 + 1]

        response = xp.exp(-xp.square(x) - xp.square(y) / (2 * (std ** 2))) / (xp.sqrt(2 * xp.pi) * std)
        response[response < xp.finfo('float').eps] = 0

        return xp.asnumpy(response / response.sum())  # Normalize so sum is 1
