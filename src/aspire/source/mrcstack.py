import numpy as np

from aspire.image import Image
from aspire.io.micrograph import Micrograph
from aspire.source import ImageSource


class MrcStack(ImageSource):
    def __init__(self, filepath):
        self.im = Micrograph(filepath, square=True).im

        super().__init__(
            L=self.im.shape[0],
            n=self.im.shape[-1]
        )

    def _images(self, start=0, num=np.inf, indices=None):
        if indices is None:
            indices = np.arange(start, min(start + num, self.n))
        return Image(self.im[:, :, indices])
