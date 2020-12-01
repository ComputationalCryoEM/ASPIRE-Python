import logging

import numpy as np

from aspire.source import ImageSource
from aspire.storage import StarFile, StarFileBlock

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

    def _images(self, start=0, num=np.inf, indices=None, batch_size=512):
        """
        Internal function to return a set of images

        :param start: The inclusive start index from which to return images.
        :param num: The exclusive end index up to which to return images.
        :param indices: The indices of images to return.
        :return: an `Image` object.
        """
        raise NotImplementedError(
            "Orientation estimation did not change the images."
            "Please use original ImageSource object."
        )

    def save(self, starfile_filepath):
        """
        Save estimated rotation angles to STAR file

        :param starfile_filepath: Path to STAR file for save rotation angles
        :return: None
        """

        df = self._metadata.copy()
        # Drop any column that doesn't start with a *single* underscore
        df = df.drop(
            [
                str(col)
                for col in df.columns
                if not col.startswith("_") or col.startswith("__")
            ],
            axis=1,
        )

        with open(starfile_filepath, "w") as f:
            # initial the star file object and save it
            starfile = StarFile(blocks=[StarFileBlock(loops=[df])])
            starfile.save(f)
