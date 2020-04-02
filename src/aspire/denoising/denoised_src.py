import os.path
import logging
import numpy as np
import pandas as pd

from aspire.image import Image
from aspire.source import ImageSource
from aspire.io.starfile import StarFileBlock, StarFile

logger = logging.getLogger(__name__)


class DenoisedImageSource(ImageSource):
    """
    Define a derived ImageSource class to perform operations for denoised 2D images
    """

    def __init__(self, src, denoiser):
        """
        Initialize a denoised ImageSource object from original ImageSource of noisy images

        :param src: Original ImageSource object storing noisy images
        :param denoiser: A Denoiser object for specifying a method for denoising
        """

        super().__init__(src.L, src. n, dtype=src.dtype,metadata=src._metadata.copy())
        self._im = None
        # self._metadata=src._metadata.copy()
        self.denoiser = denoiser

    def _images(self, start=0, num=np.inf, indices=None, batch_size=512):
        """
        Internal function to return a set of images after denoising

        :param start: The inclusive start index from which to return images.
        :param num: The exclusive end index up to which to return images.
        :param num: The indices of images to return.
        :return: an `Image` object after denoisng.
        """
        if indices is None:
            indices = np.arange(start, min(start + num, self.n))
        else:
            start = indices.min()
        end = indices.max()

        im = np.empty((self.L, self.L, len(indices)))

        logger.info(f'Loading {len(indices)} images complete')
        for istart in range(start, end, batch_size):
            imgs_denoised = self.denoiser.images(istart, batch_size)
            im = imgs_denoised.data

        return Image(im)

    def _create_star(self, starfile_filepath, batch_size=512):
        """
        Create a new STAR file and corresponding individual name for output .mrcs files
        Note that .mrcs files are saved at the same location as the STAR file.

        :param starfile_filepath: Path to STAR file where we want to save image_source
        :param batch_size: Batch size of images to query from the `ImageSource` object. Every `batch_size` rows,
            entries are written to STAR file, and the `.mrcs` files saved.
        :return: None
        """
        # TODO: Accessing protected member - provide a way to get a handle on the _metadata attribute.
        df = self._metadata.copy()
        # Drop any column that doesn't start with a *single* underscore
        df = df.drop([str(col) for col in df.columns if not col.startswith('_') or col.startswith('__')], axis=1)

        self.starfile_filepath = None
        # Create a new column for outputting new .mrcs files
        df['_rlnImageName'] = ''
        self.mrcs_fileout = [None for i in range(self.n)]
        with open(starfile_filepath, 'w') as f:
            for i_start in np.arange(0, self.n, batch_size):
                i_end = min(self.n, i_start + batch_size)
                num = i_end - i_start

                mrcs_filename = os.path.splitext(os.path.basename(starfile_filepath))[0] + f'_{i_start}_{i_end-1}.mrcs'
                mrcs_filepath = os.path.join(
                    os.path.dirname(starfile_filepath),
                    mrcs_filename
                )
                for ib in range(i_start, i_end):
                    self.mrcs_fileout[ib] = mrcs_filepath
                df['_rlnImageName'][i_start: i_end] = pd.Series(
                    ['{0:06}@{1}'.format(j + 1, mrcs_filepath) for j in range(num)])

            starfile = StarFile(blocks=[StarFileBlock(loops=[df])])
            starfile.save(f)
