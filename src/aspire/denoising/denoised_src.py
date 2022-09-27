import logging

import numpy as np

from aspire.image import Image
from aspire.source import ImageSource

logger = logging.getLogger(__name__)


class DenoisedImageSource(ImageSource):
    """
    Define a derived ImageSource class to perform operations for denoised 2D images
    """

    def __init__(self, src, denoiser, batch_size=512):
        """
        Initialize a denoised ImageSource object from original ImageSource of noisy images

        :param src: Original ImageSource object storing noisy images
        :param denoiser: A Denoiser object for specifying a method for denoising
        :param batch_size: Batch size for loading denoised images.
        """

        super().__init__(src.L, src.n, dtype=src.dtype, metadata=src._metadata.copy())
        self._im = None
        self.denoiser = denoiser
        self.batch_size = batch_size

    def _images(self, indices):
        """
        Internal function to return a set of images after denoising, when accessed via the
        `ImageSource.images` property.
        :param indices: The indices of images to return as a 1-D NumPy array.
        :return: an `Image` object after denoisng.
        """
        # check for cached images first
        if self._cached_im is not None:
            logger.info("Loading images from cache")
            return self.generation_pipeline.forward(
                Image(self._cached_im[indices, :, :]), indices
            )

        # start and end (and indices) refer to the indices in the DenoisedImageSource
        # that are being denoised and returned in batches
        start = indices.min()
        end = indices.max()

        nimgs = len(indices)
        im = np.empty((nimgs, self.L, self.L))

        # If we request less than a whole batch, don't crash
        batch_size = min(nimgs, self.batch_size)

        logger.info(f"Loading {nimgs} images complete")
        for batch_start in range(start, end + 1, batch_size):
            imgs_denoised = self.denoiser.images(batch_start, batch_size)
            batch_end = min(batch_start + batch_size, end + 1)
            # we subtract start here to correct for any offset in the indices
            im[batch_start - start : batch_end - start] = imgs_denoised.asnumpy()

        # Finally, apply transforms to resulting Image
        return self.generation_pipeline.forward(Image(im), indices)
