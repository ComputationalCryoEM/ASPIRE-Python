import logging

import numpy as np

from aspire.abinitio import CLOrient3D, CLSyncVoting
from aspire.operators import IdentityFilter
from aspire.source import ImageSource

logger = logging.getLogger(__name__)


class OrientedSource(ImageSource):
    """
    Source for oriented 2D images using orientation estimation methods.
    """

    def __init__(self, src, orientation_estimator=None):
        """
        Constructor of an object for finding the orientations for 2D images
        using orientation estimation methods.

        :param src: Source used for orientation estimation
        :param orientation_estimator: CLOrient3D subclass used for orientation estimation.
        """

        self.src = src
        if not isinstance(self.src, ImageSource):
            raise ValueError(
                f"`src` should be subclass of `ImageSource`, found {self.src}."
            )

        if orientation_estimator is None:
            orientation_estimator = CLSyncVoting(src)

        self.orientation_estimator = orientation_estimator
        if not isinstance(self.orientation_estimator, CLOrient3D):
            raise ValueError(
                f"`orientation_estimator` should be subclass of `CLOrient3D`, found {self.orientation_estimator}."
            )

        # Get metadata from src.
        metadata = self.src._metadata

        super().__init__(
            L=self.src.L,
            n=self.src.n,
            dtype=self.src.dtype,
            metadata=metadata,
        )

        # Create filter indices, these are required to pass unharmed through filter eval code
        #   that is potentially called by other methods later.
        self.filter_indices = np.zeros(self.n, dtype=int)
        self.unique_filters = [IdentityFilter()]

        # Perform orientation estimation.
        self._oriented = False
        self._orient()

    def _orient(self):
        """
        Perform orientation estimation.
        """
        if self._oriented:
            logger.debug(f"{self.__class__.__name__} arleady oriented, skipping")
            return

        logger.info(
            f"Estimating rotations for {self.src} using {self.orientation_estimator}."
        )
        self.orientation_estimator.estimate_rotations()
        self.rotations = self.orientation_estimator.rotations
        self._oriented = True

    def _images(self, indices):
        """
        Returns images from `self.src` corresponding to `indices`.

        :param indices: A 1-D NumPy array of indices.
        :return: An `Image` object.
        """
        return self.src.images[indices]
