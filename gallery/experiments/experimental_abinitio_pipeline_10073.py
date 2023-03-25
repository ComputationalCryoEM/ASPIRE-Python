"""
Abinitio Pipeline - Experimental Data Empiar 10073
==================================================

This notebook introduces a selection of
components corresponding to loading real Relion picked
particle Cryo-EM data and running key ASPIRE-Python
Abinitio model components as a pipeline.

Specifically this pipeline uses the
EMPIAR 10073 picked particles data, available here:

https://www.ebi.ac.uk/empiar/EMPIAR-10073

https://www.ebi.ac.uk/emdb/EMD-8012
"""

# %%
# Imports
# -------
# First import some of the usual suspects.
# In addition, import some classes from
# the ASPIRE package that will be used throughout this experiment.

import logging
from pathlib import Path

import numpy as np

from aspire.abinitio import CLSyncVoting
from aspire.basis import FFBBasis2D, FFBBasis3D
from aspire.classification import (
    BFRAverager2D,
    ContrastImageQualityFunction,
    GlobalWithRepulsionClassSelector,
)
from aspire.denoising import DefaultClassAvgSource
from aspire.reconstruction import MeanEstimator
from aspire.source import RelionSource

logger = logging.getLogger(__name__)


# %%
# Parameters
# ---------------
# Example simulation configuration.

n_imgs = None  # Set to None for all images in starfile, can set smaller for tests.
img_size = 32  # Downsample the images/reconstruction to a desired resolution
n_classes = 2000  # How many class averages to compute.
n_nbor = 50  # How many neighbors to stack
starfile_in = "10073/data/shiny_correctpaths_cleanedcorruptstacks.star"
data_folder = "."  # This depends on the specific starfile entries.
volume_output_filename = f"10073_abinitio_c{n_classes}_m{n_nbor}_{img_size}.mrc"
pixel_size = 1.43


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

src.cache()

# %%
# Class Averaging
# ----------------------
#
# Now perform classification and averaging for each class.

logger.info("Begin Class Averaging")

# Now perform classification and averaging for each class.
# This also demonstrates customizing a ClassAvgSource, by using global
# contrast selection.  This computes the entire set of class averages,
# and sorts them by (highest) contrast.

# Build up the customized components.
basis = FFBBasis2D(src.L, dtype=src.dtype)
averager = BFRAverager2D(basis, src, num_procs=32)
quality_function = ContrastImageQualityFunction()
class_selector = GlobalWithRepulsionClassSelector(averager, quality_function)

# Assemble the components into the Source.
avgs = DefaultClassAvgSource(
    src, n_nbor=n_nbor, averager=averager, class_selector=class_selector
)
np.save(
    "experimental_10073_class_averages__indices.npy",
    avgs.selection_indices,
)

# We'll continue our pipeline with the first `n_classes` from `avgs`.
avgs = avgs[:n_classes]

# Save off the set of class average images.
avgs.save("experimental_10073_class_averages_global.star", overwrite=True)


# %%
# Common Line Estimation
# ----------------------
#
# Next create a CL instance for estimating orientation of projections
# using the Common Line with Synchronization Voting method.

logger.info("Begin Orientation Estimation")

# Run orientation estimation on ``avgs``.
orient_est = CLSyncVoting(avgs, n_theta=180)
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
estimated_volume.save(volume_output_filename, overwrite=True)
logger.info(f"Saved Volume to {str(Path(volume_output_filename).resolve())}")
