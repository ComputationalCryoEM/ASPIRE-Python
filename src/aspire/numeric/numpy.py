import numpy as np

cp = None
try:
    import cupy as cp
except ModuleNotFoundError:
    pass


class Numpy:
    @staticmethod
    def asnumpy(x):
        if cp and isinstance(x, cp.ndarray):
            x = x.get()
        return x

    def __getattr__(self, item):
        """
        Catch-all method to to allow a straight pass-through \
        of any attribute that is not supported above.
        """

        return getattr(np, item)
