import logging

logger = logging.getLogger(__name__)


class Denoise:
    """
    Define a base class for denoising 2D images
    """

    def __init__(self, src, as_type='single'):
        """
        constructor of an object for denoising 2D images
        """

        ensure(basis.d == 2, 'Only two-dimensional basis functions are needed.')

        self.src = src
        self.as_type = as_type

        self.L = src.L
        self.n = src.n

    def output_images(self):
        """
        Output the clean images
        """
        pass




