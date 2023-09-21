import logging

from aspire.denoising import Denoiser
from aspire.source import ImageSource

logger = logging.getLogger(__name__)


class DenoisedSource(ImageSource):
    """
    `ImageSource` class serving denoised 2D images.
    """

    def __init__(self, denoiser):
        """
        Initialize a denoised `ImageSource` object from an `ImageSource`.

        :param denoiser: A `Denoiser` object for specifying a method for denoising
        """
        self.src = denoiser.src
        super().__init__(
            self.src.L,
            self.src.n,
            dtype=self.src.dtype,
            metadata=self.src._metadata.copy(),
        )
        self.denoiser = denoiser
        if not isinstance(denoiser, Denoiser):
            raise TypeError("`denoiser` must be subclass of `Denoiser`")

        # Any further operations should not mutate this instance.
        self._mutable = False

    def _images(self, indices):
        """
        Internal function to return a set of images after denoising, when accessed via the
        `ImageSource.images` property.

        :param indices: The indices of images to return as a 1-D Numpy array.
        :return: an `Image` object after denoising.
        """

        # check for cached images first
        if self._cached_im is not None:
            logger.info("Loading images from cache")
            return self.generation_pipeline.forward(
                self._cached_im[indices, :, :], indices
            )

        imgs_denoised = self.denoiser.denoise[indices]

        # Finally, apply transforms to resulting Image
        return self.generation_pipeline.forward(imgs_denoised, indices)
