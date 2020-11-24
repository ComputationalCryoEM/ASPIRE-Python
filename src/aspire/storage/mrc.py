import numpy as np


class MrcStats:
    def __init__(self):
        """
        Instantiate an empty instance ready to receive slices of arrays.

        :return: MrcStats instance
        """

        self.amin = np.inf
        self.amax = -np.inf
        self.asum = 0.0
        self.asize = 0
        self.asum2 = 0.0

    def push(self, array_slice):
        """
        Incrementally add contribution of array_slice to stats.

        :param array_slice: An ndarray to contribute to stats.
        """

        self.amin = min(self.amin, np.min(array_slice))
        self.amax = max(self.amax, np.max(array_slice))
        # For mean we'll do div when called.
        self.asum += np.sum(array_slice)
        self.asum2 += np.sum(np.square(array_slice))
        self.asize += np.size(array_slice)

    @property
    def amean(self):
        """
        Compute the curren mean.
        """

        return self.asum / self.asize

    @property
    def arms(self):
        """
        Compute the current standard deviation.

        We use the simplest method.
        """

        var = self.asum2 / self.asize - (self.asum / self.asize) ** 2
        return np.sqrt(var)

    def update_header(self, mrcobj):
        """
        Assign the statistics to `mrcobj` header.

        :param mrcobj: The `mrcfile` instance we want stats written to.
        """

        mrcobj.header.dmin = self.amin.astype(np.float32)
        mrcobj.header.dmax = self.amax.astype(np.float32)
        mrcobj.header.dmean = self.amean.astype(np.float32)
        mrcobj.header.rms = self.arms.astype(np.float32)
