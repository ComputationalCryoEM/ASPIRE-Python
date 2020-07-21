"""
This script illustrates the preprocess steps before starting the pipeline of
reconstructing 3D map using the simulated 2D images.
"""
import logging
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import os

from aspire.estimation.noise import WhiteNoiseEstimator
from aspire.source.simulation import Simulation
from aspire.utils.filters import (RadialCTFFilter, ScalarFilter)
from aspire.utils.preprocess import downsample


logger = logging.getLogger('aspire')

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/')

logger.info('This script illustrates orientation estimation using '
            'synchronization matrix and voting method')

# Set the downsample size of images
img_size = 33

# Set the total number of images generated from the 3D map
num_imgs = 512

# Set the number of 3D maps
num_maps = 1

# Set the noise variance and build the noise filter
noise_variance = 4e-1
noise_filter = ScalarFilter(dim=2, value=noise_variance)

# Specify the CTF parameters not used for this example
# but necessary for initializing the simulation object
pixel_size = 5*65/33             # Pixel size of the images (in angstroms).
voltage = 200                    # Voltage (in KV)
defocus_min = 1.5e4              # Minimum defocus value (in angstroms).
defocus_max = 2.5e4              # Maximum defocus value (in angstroms).
defocus_ct = 7                   # Number of defocus groups.
Cs = 2.0                         # Spherical aberration
alpha = 0.1                      # Amplitude contrast

logger.info('Initialize simulation object and CTF filters.')
# Create CTF filters
CTF_filters = [RadialCTFFilter(pixel_size, voltage, defocus=d, Cs=2.0, alpha=0.1)
           for d in np.linspace(defocus_min, defocus_max, defocus_ct)]

# Load the map file of a 70S Ribosome and downsample the 3D map to desired resolution.
infile = mrcfile.open(os.path.join(DATA_DIR, 'clean70SRibosome_vol_65p.mrc'))
logger.info(f'Load 3D map from mrc file, {infile}')
vols = infile.data
vols = vols[..., np.newaxis]

# Downsample the volume to a desired resolution and increase density
# by 1.0e5 time for a better graph view
logger.info(f'Downsample map to a resolution of {img_size} x {img_size} x {img_size}')
vols = downsample(vols, (img_size,) * 3) * 1.0e5

# Create a simulation object with specified filters and the downsampled 3D map
logger.info('Use downsampled map to creat simulation object.')
source = Simulation(
    L=img_size,
    n=num_imgs,
    vols=vols,
    C=num_maps,
    filters=CTF_filters,
    noise_filter=noise_filter
)

logger.info('Obtain original images.')
imgs_od = source.images(start=0, num=10).asnumpy()

logger.info('Perform phase flip to input images.')
source.phase_flip()
imgs_pf = source.images(start=0, num=10).asnumpy()

logger.info('Normalize images to background noise.')
source.normalize_background()
imgs_nb = source.images(start=0, num=10).asnumpy()

logger.info('Whiten noise of images')
noise_estimator = WhiteNoiseEstimator(source)
source.whiten(noise_estimator.filter)
imgs_wt = source.images(start=0, num=10).asnumpy()

logger.info('Invert the global density contrast if need')
source.invert_contrast()
imgs_rc = source.images(start=0, num=10).asnumpy()


# plot the first images
logger.info('Plot first image from each preprocess steps')
idm = 0
plt.subplot(2, 3, 1)
plt.imshow(imgs_od[..., idm], cmap='gray')
plt.colorbar(orientation='horizontal')
plt.title('original image')

plt.subplot(2, 3, 2)
plt.imshow(imgs_pf[..., idm], cmap='gray')
plt.colorbar(orientation='horizontal')
plt.title('phase flip')

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
plt.title('invert contrast')
plt.show()

