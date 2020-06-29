"""
This script illustrates how to preprocess experimental CryoEM images
before performing reconstructing 3D map.
"""

import logging

from aspire.source.relion import RelionSource
from aspire.estimation.noise import WhiteNoiseEstimator

logger = logging.getLogger('aspire')

starfile_in = '/tigress/junchaox/CryoEMdata/empiar10028/shiny_2sets.star'
data_folder = '/tigress/junchaox/CryoEMdata/empiar10028/'
pixel_size = 1.34
max_rows = 1000

logger.info(f'Read in images from {starfile_in} and preprocess the images.')
source = RelionSource(
    starfile_in,
    data_folder,
    pixel_size=pixel_size,
    max_rows=max_rows
)

logger.info('Perform phase flip to input images.')
starfile_out = '/tigress/junchaox/CryoEMdata/empiar10028/preprocess/shiny_2sets_phase_flip.star'
imgs_out = source.phase_flip()
source.cache(imgs_out)
source.save(starfile_out, batch_size=512, save_mode='single', overwrite=False)

max_resolution = 60
logger.info('Downsample the resolution to {max_resolution} X {max_resolution}')
starfile_out = '/tigress/junchaox/CryoEMdata/empiar10028/preprocess/shiny_2sets_down_sample.star'
if max_resolution < source.L:
    source.downsample(max_resolution)
source.save(starfile_out, batch_size=512, save_mode='single', overwrite=False)

logger.info('Normalize the background.')
starfile_out = '/tigress/junchaox/CryoEMdata/empiar10028/preprocess/shiny_2sets_normalize_background.star'
source.normalize_background()
source.save(starfile_out, batch_size=512, save_mode='single', overwrite=False)

logger.info('Whiten the noise of images using the noise estimator')
starfile_out = '/tigress/junchaox/CryoEMdata/empiar10028/preprocess/shiny_2sets_whitening.star'
noise_estimator = WhiteNoiseEstimator(source)
source.whiten(noise_estimator.filter)
source.save(starfile_out, batch_size=512, save_mode='single', overwrite=False)

logger.info('Reverse the global density contrast')
starfile_out = '/tigress/junchaox/CryoEMdata/empiar10028/preprocess/shiny_2sets_reverse_contrast.star'
source.reverse_contrast()
source.save(starfile_out, batch_size=512, save_mode='single', overwrite=False)



