from aspire.io.micrograph import Micrograph
from aspire.image import ImageStack
from aspire.source import ImageSource


class MrcStack(ImageSource):
    def __init__(self, filepath):
        self.im = Micrograph(filepath, square=True).im

        super().__init__(
            L=self.im.shape[0],
            n=self.im.shape[-1]
        )

    def _images(self, start=0, num=None):
        end = self.n
        if num is not None:
            end = min(start + num, self.n)
        x = self.im[:, :, start:end]
        return ImageStack(x)
