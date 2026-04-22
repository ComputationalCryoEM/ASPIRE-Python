import cupy as cp
import numpy as np


class Cupy:
    def __getattr__(self, item):
        """
        Catch-all method to to allow a straight pass-through of any attribute that is not supported above.
        """
        return getattr(cp, item)

    @staticmethod
    def atleast_1d(x):
        """
        Provide an agnostic `atleast_1d`.

        Returns same type as input.
        """
        _fn = np.atleast_1d
        if cp and isinstance(x, cp.ndarray):
            _fn = cp.atleast_1d
        return _fn(x)
