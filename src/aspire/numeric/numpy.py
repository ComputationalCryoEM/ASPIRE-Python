import cupy as cp
import numpy as np


class Numpy:
    @staticmethod
    def asnumpy(x):
        if isinstance(x, cp.ndarray):
            x = x.get()
        return x

    def __getattr__(self, item):
        """
        Catch-all method to to allow a straight pass-through \
        of any attribute that is not supported above.
        """

        return getattr(np, item)
