import logging

import numpy as np

logger = logging.getLogger(__name__)


class Class2D:
    """
    Base class for 2D Image Classification methods.
    """

    def __init__(self, src, dtype=None):
        self.src = src

        if dtype is not None:
            self.dtype = np.dtype(dtype)
            if self.dtype != self.src.dtype:
                logger.warning(
                    f"Class2D src.dtype {self.src.dtype} does not match self.dtype {self.dtype}."
                )
        else:
            self.dtype = self.src.dtype
