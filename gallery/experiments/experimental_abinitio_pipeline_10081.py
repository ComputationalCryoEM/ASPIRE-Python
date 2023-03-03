"""
Abinitio Pipeline - Experimental Data Empiar 10081
==================================================

This notebook introduces a selection of
components corresponding to loading real Relion picked
particle Cryo-EM data and running key ASPIRE-Python
Abinitio model components as a pipeline.

Specifically this pipeline uses the
EMPIAR 10081 picked particles data, available here:

https://www.ebi.ac.uk/empiar/EMPIAR-10081

https://www.ebi.ac.uk/emdb/EMD-8511
"""

# %%
# Imports
# -------
# First import some of the usual suspects.
# In addition, import some classes from
# the ASPIRE package that will be used throughout this experiment.

import logging

from aspire.abinitio import CLSyncVoting
from aspire.basis import FFBBasis3D
from aspire.denoising import ClassAvgSourcev11
from aspire.noise import AnisotropicNoiseEstimator
from aspire.reconstruction import MeanEstimator
from aspire.source import ArrayImageSource, RelionSource

logger = logging.getLogger(__name__)


# %%
# Parameters
# ---------------
# Example simulation configuration.

n_imgs = None  # Set to None for all images in starfile, can set smaller for tests.
img_size = 32  # Downsample the images/reconstruction to a desired resolution
n_classes = 2000  # How many class averages to compute.
n_nbor = 100  # How many neighbors to stack
starfile_in = (
    "/scratch/ExperimentalData/staging/10081/data/Particles/micrographs/data.star"
)
data_folder = "../.."  # This depends on the specific starfile entries.
volume_filename_prefix_out = f"10081_abinitio_c{n_classes}_m{n_nbor}_{img_size}.mrc"
pixel_size = 1.3


# %%
# Source data and Preprocessing
# -----------------------------
#
# `RelionSource` is used to access the experimental data via a `starfile`.
# Begin by downsampling to our chosen resolution, then preprocess
# to correct for CTF and noise.

# Create a source object for the experimental images
src = RelionSource(
    starfile_in, pixel_size=pixel_size, max_rows=n_imgs, data_folder=data_folder
)

# Downsample the images
logger.info(f"Set the resolution to {img_size} X {img_size}")
src.downsample(img_size)

# Use phase_flip to attempt correcting for CTF.
logger.info("Perform phase flip to input images.")
src.phase_flip()

# Estimate the noise and `Whiten` based on the estimated noise
aiso_noise_estimator = AnisotropicNoiseEstimator(src)
src.whiten(aiso_noise_estimator.filter)

# %%
# Class Averaging
# ----------------------
#
# Now perform classification and averaging for each class.

logger.info("Begin Class Averaging")

# Now perform classification and averaging for each class.
# Automaticaly configure parallel processing
avgs = ClassAvgSourcev11(src, n_nbor=n_nbor, num_procs=None)

# We'll manually cache `n_classes` worth to speed things up.
avgs = ArrayImageSource(avgs.images[:n_classes])


# %%
# Common Line Estimation
# ----------------------
#
# Next create a CL instance for estimating orientation of projections
# using the Common Line with Synchronization Voting method.

logger.info("Begin Orientation Estimation")

orient_est = CLSyncVoting(avgs, n_theta=360)
# Get the estimated rotations
orient_est.estimate_rotations()
rots_est = orient_est.rotations

# %%
# Volume Reconstruction
# ----------------------
#
# Using the estimated rotations, attempt to reconstruct a volume.

logger.info("Begin Volume reconstruction")

# Assign the estimated rotations to the class averages
avgs.rotations = rots_est

# Create a reasonable Basis for the 3d Volume
basis = FFBBasis3D((img_size,) * 3, dtype=src.dtype)

# Setup an estimator to perform the back projection.
estimator = MeanEstimator(avgs, basis)

# Perform the estimation and save the volume.
estimated_volume = estimator.estimate()
estimated_volume.save(volume_filename_prefix_out, overwrite=True)
