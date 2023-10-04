import logging
from abc import ABC, abstractmethod

from aspire.source.image import _ImageAccessor

logger = logging.getLogger(__name__)


class Denoiser(ABC):
    """
    Base class for 2D image denoisers.
    """

    def __init__(self, src):
        """
        Initialize an object for denoising 2D images from `src`.

        :param src: `ImageSource` providing noisy images.
        """

        self.src = src
        self.dtype = src.dtype
        self.n = src.n
        self._img_accessor = _ImageAccessor(self._denoise, self.n)

    @property
    def denoise(self):
        """
        Subscriptable property returning 2D images after denoising.

        See `_ImageAccessor`.
        """
        return self._img_accessor

    @abstractmethod
    def _denoise(self, indices):
        """
        Subclasses must implement a private `_denoise` method accepting `indices`.
        Subclasses handle any caching as well as denoising.

        See `_ImageAccessor`.
        """
