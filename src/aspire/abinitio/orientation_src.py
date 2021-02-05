import logging

from aspire.source import ImageSource

logger = logging.getLogger(__name__)


class OrientEstSource(ImageSource):
    """
    Derived an ImageSource class for updating orientation information
    """

    def __init__(self, src, orient_method):
        """
        Initialize an Orientation ImageSource object from original ImageSource

        :param src: Original ImageSource object after 2D classification
        :param orient_method: object specifying orientation estimation method
        """

        super().__init__(src.L, src.n, dtype=src.dtype, metadata=src._metadata.copy())
        self._im = None
        self.orient_method = orient_method
        self.rots = orient_method.rotations
