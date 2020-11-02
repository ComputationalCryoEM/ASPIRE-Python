import logging

from aspire.denoising import Denoiser

logger = logging.getLogger(__name__)


class ClassAvg(Denoiser):
    """
    Define a derived class for denoising 2D images using class average methods
    """

    def __init__(self, img_src, class_index):
        """
        constructor of an object for denoising 2D images using class averaging method.
        """
        pass

    def class_averaging(self):

        pass

    def output_images(self):
        """
        Output the clean images
        """
        pass
