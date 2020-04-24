import os.path
import logging
import numpy as np
import pandas as pd

from aspire.image import Image
from aspire.source import ImageSource
from aspire.io.starfile import StarFileBlock, StarFile

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

        super().__init__(src.L, src. n, dtype=src.dtype, metadata=src._metadata.copy())
        self._im = None
        self.denoiser = denoiser

    def _images(self, start=0, num=np.inf, indices=None, batch_size=512):
        """
        Internal function to return a set of images after denoising

        :param start: The inclusive start index from which to return images.
        :param num: The exclusive end index up to which to return images.
        :param num: The indices of images to return.
        :return: an `Image` object after denoisng.
        """
        if indices is None:
            indices = np.arange(start, min(start + num, self.n))
        else:
            start = indices.min()
        end = indices.max()

        im = np.empty((self.L, self.L, len(indices)))

        logger.info(f'Loading {len(indices)} images complete')
        for istart in range(start, end, batch_size):
            imgs_denoised = self.denoiser.images(istart, batch_size)
            im = imgs_denoised.data

        return Image(im)
