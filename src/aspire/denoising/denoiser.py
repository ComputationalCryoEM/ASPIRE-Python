import logging

logger = logging.getLogger(__name__)


class Denoiser:
    """
    Define a base class for denoising 2D images
    """

    def __init__(self, src):
        """
        Initialize an object for denoising 2D images from the image source

        :param src: The source object of 2D images with metadata
        """
        self.src = src
        self.dtype = src.dtype
        self.nimg = src.n

    def denoise(self):
        """
        Precompute for Denoiser and DenoisedImageSource for 2D images
        """
        raise NotImplementedError("subclasses must implement this")

    def image(self, istart=0, batch_size=512):
        """
        Obtain a batch size of 2D images after denosing by a specified method
        """
        raise NotImplementedError("subclasses must implement this")
