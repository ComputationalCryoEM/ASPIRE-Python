import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class Class2D(ABC):
    """
    Base class for 2D Image Classification methods.
    """

    def __init__(
        self,
        src,
        n_nbor=100,
        seed=None,
        dtype=None,
    ):
        """
        Base constructor of an object for classifying 2D images.

        :param src: ImageSource or subclass, provides images.
        :param n_nbor: Number of nearest neighbors to compute.
        :param seed: Optional RNG seed to be passed to random methods, (example Random NN).
        :param dtype: Numpy dtype, defaults to `src.dtype`.
        """
        self.src = src

        if dtype is not None:
            self.dtype = np.dtype(dtype)
            if self.dtype != self.src.dtype:
                logger.warning(
                    f"Class2D src.dtype {self.src.dtype} does not match self.dtype {self.dtype}."
                )
        else:
            self.dtype = self.src.dtype

        self.n_nbor = n_nbor
        self.seed = seed

    @abstractmethod
    def classify(self):
        """
        Classify the images from Source into classes with similar viewing angles.

        Returns classes and associated metadata (classes, reflections, distances)
        """
