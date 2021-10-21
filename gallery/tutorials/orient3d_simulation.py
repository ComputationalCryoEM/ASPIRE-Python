"""
3D Image Orientation
====================

This script illustrates the estimation of orientation angles using a synchronization
matrix and the voting method, based on simulated data projected from a 3D cryo-EM map.
"""

import logging
import os

import mrcfile
import numpy as np

from aspire.abinitio import CLSyncVoting
from aspire.operators import RadialCTFFilter
from aspire.source.simulation import Simulation
from aspire.utils.coor_trans import (
    get_aligned_rotations,
    get_rots_mse,
    register_rotations,
)
from aspire.volume import Volume

logger = logging.getLogger(__name__)

DATA_DIR = "data"

logger.info(
    "This script illustrates orientation estimation using "
    "synchronization matrix and voting method"
)

# %%
# Initialize Simulation Object and CTF Filters
# --------------------------------------------

# Define a precision for this experiment
dtype = np.float32

# Set the sizes of images
img_size = 33

# Set the total number of images generated from the 3D map
num_imgs = 128

# Specify the CTF parameters not used for this example
# but necessary for initializing the simulation object
pixel_size = 5  # Pixel size of the images (in angstroms)
voltage = 200  # Voltage (in KV)
defocus_min = 1.5e4  # Minimum defocus value (in angstroms)
defocus_max = 2.5e4  # Maximum defocus value (in angstroms)
defocus_ct = 7  # Number of defocus groups.
Cs = 2.0  # Spherical aberration
alpha = 0.1  # Amplitude contrast

logger.info("Initialize simulation object and CTF filters.")
# Create CTF filters
filters = [
    RadialCTFFilter(pixel_size, voltage, defocus=d, Cs=2.0, alpha=0.1)
    for d in np.linspace(defocus_min, defocus_max, defocus_ct)
]

# %%
# Downsampling
# ------------

# Load the map file of a 70S Ribosome and downsample the 3D map to desired resolution.
# The downsampling should be done by the internal function of Volume object in future.
logger.info(
    f"Load 3D map and downsample 3D map to desired grids "
    f"of {img_size} x {img_size} x {img_size}."
)
infile = mrcfile.open(os.path.join(DATA_DIR, "clean70SRibosome_vol_65p.mrc"))
vols = Volume(infile.data.astype(dtype))
vols = vols.downsample((img_size,) * 3)

# %%
# Create Simulation Object and Obtain True Rotation Angles
# --------------------------------------------------------

# Create a simulation object with specified filters and the downsampled 3D map
logger.info("Use downsampled map to creat simulation object.")
sim = Simulation(L=img_size, n=num_imgs, vols=vols, unique_filters=filters, dtype=dtype)

logger.info("Get true rotation angles generated randomly by the simulation object.")
rots_true = sim.rots

# %%
# Estimate Orientation and Rotation Angles
# ----------------------------------------

# Initialize an orientation estimation object and perform view angle estimation
logger.info("Estimate rotation angles using synchronization matrix and voting method.")
orient_est = CLSyncVoting(sim, n_theta=36)
orient_est.estimate_rotations()
rots_est = orient_est.rotations

# %%
# Mean Squared Error
# ------------------

# Get register rotations after performing global alignment
Q_mat, flag = register_rotations(rots_est, rots_true)
regrot = get_aligned_rotations(rots_est, Q_mat, flag)
mse_reg = get_rots_mse(regrot, rots_true)
logger.info(
    f"MSE deviation of the estimated rotations using register_rotations : {mse_reg}"
)

# Basic Check
assert mse_reg < 0.06
