import numpy as np

from aspire.image import Image
from aspire.source import ImageSource
from aspire.storage import Micrograph


class MrcStack(ImageSource):
    def __init__(self, filepath, dtype=np.float32):
        self.im = Micrograph(filepath, square=True).im

        super().__init__(
            L=self.im.res,
            n=self.im.n_images,
            dtype=dtype,
        )

    def _images(self, start=0, num=np.inf, indices=None):
        if indices is None:
            indices = np.arange(start, min(start + num, self.n))
        return Image(self.im[indices, :, :])
