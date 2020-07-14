"""
This script illustrates how to preprocess experimental CryoEM images
before performing reconstructing 3D map.
"""

import logging
import matplotlib.pyplot as plt

from aspire.source.relion import RelionSource
from aspire.estimation.noise import WhiteNoiseEstimator

logger = logging.getLogger('aspire')

starfile_in = '/tigress/junchaox/CryoEMdata/empiar10028/shiny_2sets.star'
data_folder = '/tigress/junchaox/CryoEMdata/empiar10028/'
pixel_size = 1.34
num_imgs = 1024

logger.info('This script illustrates how to preprocess experimental CryoEM images')
logger.info(f'Read in images from {starfile_in} and preprocess the images')
source = RelionSource(
    starfile_in,
    data_folder,
    pixel_size=pixel_size,
    max_rows=num_imgs
)

logger.info('Obtain original images')
imgs_od = source.images(start=0, num=num_imgs).asnumpy()

logger.info('Perform phase flip to input images')
starfile_out = f'{data_folder}/preprocess/shiny_2sets_phase_flip.star'
source.phase_flip()
imgs_pf = source.images(start=0, num=num_imgs).asnumpy()
source.save(starfile_out, batch_size=512, save_mode='single', overwrite=False)

max_resolution = 60
logger.info(f'Downsample the resolution to {max_resolution} X {max_resolution}')
starfile_out = f'{data_folder}/preprocess/shiny_2sets_down_sample.star'
if max_resolution < source.L:
    source.downsample(max_resolution)
imgs_ds = source.images(start=0, num=num_imgs).asnumpy()
source.save(starfile_out, batch_size=512, save_mode='single', overwrite=False)

logger.info('Normalize the background')
starfile_out = f'{data_folder}/preprocess/shiny_2sets_normalize_background.star'
source.normalize_background()
imgs_nb = source.images(start=0, num=num_imgs).asnumpy()
source.save(starfile_out, batch_size=512, save_mode='single', overwrite=False)

logger.info('Whiten the noise of images using the noise estimator')
starfile_out = f'{data_folder}/preprocess/shiny_2sets_whitening.star'
noise_estimator = WhiteNoiseEstimator(source)
source.whiten(noise_estimator.filter)
imgs_wt = source.images(start=0, num=num_imgs).asnumpy()
source.save(starfile_out, batch_size=512, save_mode='single', overwrite=False)

logger.info('Reverse the global density contrast')
starfile_out = f'{data_folder}/preprocess/shiny_2sets_invert_contrast.star'
source.invert_contrast()
imgs_rc = source.images(start=0, num=num_imgs).asnumpy()
source.save(starfile_out, batch_size=512, save_mode='single', overwrite=False)

# plot the first images
idm = 0
plt.subplot(2, 3, 1)
plt.imshow(imgs_od[..., idm], cmap='gray')
plt.colorbar(orientation='horizontal')
plt.title('original image')
plt.subplot(2, 3, 2)
plt.imshow(imgs_pf[..., idm], cmap='gray')
plt.colorbar(orientation='horizontal')
plt.title('phase flip')
plt.subplot(2, 3, 3)
plt.imshow(imgs_ds[..., idm], cmap='gray')
plt.colorbar(orientation='horizontal')
plt.title('down sample')
plt.subplot(2, 3, 4)
plt.imshow(imgs_nb[..., idm], cmap='gray')
plt.colorbar(orientation='horizontal')
plt.title('normalize background')
plt.subplot(2, 3, 5)
plt.imshow(imgs_wt[..., idm], cmap='gray')
plt.colorbar(orientation='horizontal')
plt.title('noise whitening')
plt.subplot(2, 3, 6)
plt.imshow(imgs_rc[..., idm], cmap='gray')
plt.colorbar(orientation='horizontal')
plt.title('contrast reverse')
plt.show()
