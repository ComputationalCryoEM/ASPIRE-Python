import logging

import numpy as np

from aspire.image import Image
from aspire.source import ImageSource

logger = logging.getLogger(__name__)


class DenoisedImageSource(ImageSource):
    """
    Define a derived ImageSource class to perform operations for denoised 2D images
    """

    def __init__(self, src, denoiser):
        """
        Initialize a denoised ImageSource object from original ImageSource of noisy images

        :param src: Original ImageSource object storing noisy images
        :param denoiser: A Denoiser object for specifying a method for denoising
        """

        super().__init__(src.L, src.n, dtype=src.dtype, metadata=src._metadata.copy())
        self._im = None
        self.denoiser = denoiser

    def _images(self, start=0, num=np.inf, indices=None, batch_size=512):
        """
        Internal function to return a set of images after denoising

        :param start: The inclusive start index from which to return images.
        :param num: The exclusive end index up to which to return images.
        :param indices: The indices of images to return.
        :return: an `Image` object after denoisng.
        """
        # start and end (and indices) refer to the indices in the DenoisedImageSource
        # that are being denoised and returned in batches
        if indices is None:
            indices = np.arange(start, min(start + num, self.n))
        else:
            start = indices.min()
        end = indices.max()

        nimgs = len(indices)
        im = np.empty((nimgs, self.L, self.L))

        logger.info(f"Loading {nimgs} images complete")
        for batch_start in range(start, end + 1, batch_size):
            imgs_denoised = self.denoiser.images(batch_start, batch_size)
            batch_end = min(batch_start + batch_size, end + 1)
            # we subtract start here to correct for any offset in the indices
            im[batch_start - start : batch_end - start] = imgs_denoised.asnumpy()

        return Image(im)
