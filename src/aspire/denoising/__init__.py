import logging

from aspire.io.starfile import save_star

logger = logging.getLogger(__name__)


class Denoiser:
    """
    Define a base class for denoising 2D images
    """
    def __init__(self, src):
        """
        Initialize an object for denoising 2D images from the image source

        :param src: The source object of 2D images with metadata
        """
        self.src = src

        self.nimg = src.n

    def denoise(self):
        """
        Denoise 2D images
        """
        raise NotImplementedError('subclasses must implement this')

    def to_src(self):
        """
        Return an ImageSource object with denoised images
        """
        raise NotImplementedError('subclasses must implement this')

    def save(self, starfile_filepath, batch_size=512, overwrite=False):
        """
        Output the denoised images to specified file

        :param starfile_filepath: Path to STAR file for saving image_source
        :param batch_size: Batch size of images to query from the `ImageSource` object.
            Every `batch_size` rows, entries are written to STAR file,
            and the `.mrcs` files saved.
        :param overwrite: Whether to overwrite any .mrcs files found at the target location.
        """
        src = self.to_src()

        save_star(src, starfile_filepath, batch_size=batch_size, overwrite=overwrite)
