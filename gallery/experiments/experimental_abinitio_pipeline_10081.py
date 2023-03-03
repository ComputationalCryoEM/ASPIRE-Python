"""
<<<<<<< HEAD:gallery/experiments/experimental_abinitio_pipeline_10081.py
Abinitio Pipeline - Experimental Data Empiar 10081
=======
Abinitio Pipeline - Experimental Data Empiar 10005
>>>>>>> f9f10b550f4983e54b9b31881c928f9b2c723385:gallery/experiments/experimental_abinitio_pipeline_10005.py
==================================================

This notebook introduces a selection of
components corresponding to loading real coordinate
base particle Cryo-EM data and running key ASPIRE-Python
Abinitio model components as a pipeline.

Specifically this pipeline uses the
<<<<<<< HEAD:gallery/experiments/experimental_abinitio_pipeline_10081.py
EMPIAR 10081 picked particles data, available here:

https://www.ebi.ac.uk/empiar/EMPIAR-10081

https://www.ebi.ac.uk/emdb/EMD-8511
=======
EMPIAR 10005 picked particles data, available here:

https://www.ebi.ac.uk/empiar/EMPIAR-10005

https://www.ebi.ac.uk/emdb/EMD-5778
>>>>>>> f9f10b550f4983e54b9b31881c928f9b2c723385:gallery/experiments/experimental_abinitio_pipeline_10005.py
"""

# %%
# Imports
# -------
# First import some of the usual suspects.
# In addition, import some classes from
# the ASPIRE package that will be used throughout this experiment.

import glob
import logging
import os

from aspire.abinitio import CLSyncVoting
from aspire.basis import FFBBasis3D
from aspire.denoising import ClassAvgSourcev11
from aspire.noise import AnisotropicNoiseEstimator
from aspire.reconstruction import MeanEstimator
from aspire.source import ArrayImageSource, CentersCoordinateSource

logger = logging.getLogger(__name__)


# %%
# Parameters
# ---------------
# Example simulation configuration.

<<<<<<< HEAD:gallery/experiments/experimental_abinitio_pipeline_10081.py
=======
working_dir = "/scratch/ExperimentalData/staging/10017/data"
>>>>>>> f9f10b550f4983e54b9b31881c928f9b2c723385:gallery/experiments/experimental_abinitio_pipeline_10005.py
n_imgs = None  # Set to None for all images in starfile, can set smaller for tests.
img_size = 32  # Downsample the images/reconstruction to a desired resolution
n_classes = 2000  # How many class averages to compute.
n_nbor = 100  # How many neighbors to stack
<<<<<<< HEAD:gallery/experiments/experimental_abinitio_pipeline_10081.py
starfile_in = "/scratch/ExperimentalData/staging/10081/data/Particles/micrographs/data.star"
data_folder = "."  # This depends on the specific starfile entries.
volume_filename_prefix_out = f"10081_abinitio_c{n_classes}_m{n_nbor}_{img_size}.mrc"
pixel_size = 1.3
=======
volume_filename_prefix_out = f"10005_abinitio_c{n_classes}_m{n_nbor}_{img_size}.mrc"
particle_size = 300
>>>>>>> f9f10b550f4983e54b9b31881c928f9b2c723385:gallery/experiments/experimental_abinitio_pipeline_10005.py


# %%
# Source data and Preprocessing
# -----------------------------
#
# `CentersCoordinateSource` is used to access the experimental data from
# mrcs using coordinate files.
#
# Begin by downsampling to our chosen resolution, then preprocess
# to correct for CTF and noise.

mrcs = sorted(glob.glob(os.path.join(working_dir, "*.mrc")))
coords = sorted(glob.glob(os.path.join(working_dir, "*.coord")))
files = list(zip(mrcs, coords))

# Create a source object for the experimental images
src = CentersCoordinateSource(files, particle_size=particle_size, max_rows=n_imgs)

# Downsample the images
logger.info(f"Set the resolution to {img_size} X {img_size}")
src.downsample(img_size)

# Use phase_flip to attempt correcting for CTF.
logger.info("Perform phase flip to input images.")
src.phase_flip()

# Estimate the noise and `Whiten` based on the estimated noise
aiso_noise_estimator = AnisotropicNoiseEstimator(src)
src.normalize_background()
src.whiten(aiso_noise_estimator.filter)

# %%
# Class Averaging
# ----------------------
#
# Now perform classification and averaging for each class.

logger.info("Begin Class Averaging")

# Now perform classification and averaging for each class.
<<<<<<< HEAD:gallery/experiments/experimental_abinitio_pipeline_10081.py
 # Automaticaly configure parallel processing
=======
# Automaticaly configure parallel processing
>>>>>>> f9f10b550f4983e54b9b31881c928f9b2c723385:gallery/experiments/experimental_abinitio_pipeline_10005.py
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
