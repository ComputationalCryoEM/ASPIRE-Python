"""
This script illustrates the estimation of orientation angles using synchronization
matrix and voting method, based on simulated data projected from a 3D CryoEM map.
"""

import os
import logging
import numpy as np

import mrcfile

from aspire.source.simulation import Simulation

from aspire.utils.filters import RadialCTFFilter
from aspire.utils.preprocess import downsample
from aspire.utils.coor_trans import qrand_rots
from aspire.utils.coor_trans import register_rotations
from aspire.utils.preprocess import vol2img
from aspire.utils.matrix import anorm

from aspire.orientation.commonline_sync import CommLineSync
from aspire.utils.matlab_compat import randn

logger = logging.getLogger('aspire')

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/')

logger.info('This script illustrates orientation estimation using '
            'synchronization matrix and voting method')

# Set the sizes of images 64 x 64
img_size = 33
# Set the total number of images generated from the 3D map
num_imgs = 128

# Set the number of 3D maps
num_maps = 1

# Set the signal-noise ratio
sn_ratio = 32

# Specify the CTF parameters not used for this example
# but necessary for initializing the simulation object
pixel_size = 5                   # Pixel size of the images (in angstroms).
voltage = 200                    # Voltage (in KV)
defocus_min = 1.5e4              # Minimum defocus value (in angstroms).
defocus_max = 2.5e4              # Maximum defocus value (in angstroms).
defocus_ct = 7                   # Number of defocus groups.
Cs = 2.0                         # Spherical aberration
alpha = 0.1                      # Amplitude contrast

logger.info('Initialize simulation object and CTF filters.')
# Create CTF filters
filters = [RadialCTFFilter(pixel_size, voltage, defocus=d, Cs=2.0, alpha=0.1)
           for d in np.linspace(defocus_min, defocus_max, defocus_ct)]

# Load the map file of a 70S Ribosome and downsample the 3D map to desired resolution.
# The downsampling should be done by the internal function of sim object in future.
# Below we use alternative implementation to obtain the exact result with Matlab version.
logger.info(f'Load 3D map and downsample 3D map to desired grids '
            f'of {img_size} x {img_size} x {img_size}.')
infile = mrcfile.open(os.path.join(DATA_DIR, 'clean70SRibosome_vol_65p.mrc'))
vols = infile.data
vols = vols[..., np.newaxis]
vols = downsample(vols, (img_size*np.ones(3, dtype=int)))

# Create a simulation object with specified filters and the downsampled 3D map
logger.info('Use downsampled map to creat simulation object.')
sim = Simulation(
    L=img_size,
    n=num_imgs,
    vols=vols,
    C=num_maps,
    filters=filters
)

# Generate 2D clean images from input 3D map. The following statement can be used from the sim object:
# imgs_clean = sim.clean_images(start=0, num=num_imgs)
# To be more consistent with the Matlab version in the numbers, we need to use the statements as below
# to replace the images generated automatically in the simulation object:
logger.info('Generate random distributed rotation angles and obtain corresponding 2D clean images.')
rots_true = qrand_rots(num_imgs, seed=0)
rots_true_inv = np.swapaxes(rots_true, 0, 2)

imgs_clean = vol2img(sim.vols[..., 0], rots_true)
imgs_clean = np.swapaxes(imgs_clean, 0, 1)

# Apply the noise at the desired singal-noise ratio to the filtered clean images
logger.info('Apply noise filters to clean images.')
power_clean = anorm(imgs_clean)**2/np.size(imgs_clean)
noise_var = power_clean/sn_ratio
imgs_noise = imgs_clean + np.sqrt(noise_var)*randn(img_size, img_size, num_imgs, seed=0)
sim.cache(imgs_noise)

# Initialize an orientation estimation object and perform view angle estimation
orient_est = CommLineSync(sim, ntheta=36)
orient_est.estimate_rotations()
rots_est = orient_est.rotations

regrot, mse_reg, diff, O, flag = register_rotations(rots_est, rots_true_inv)

logger.info(f'MSE deviation of the estimated rotations using register_rotations : {mse_reg[0]}')

