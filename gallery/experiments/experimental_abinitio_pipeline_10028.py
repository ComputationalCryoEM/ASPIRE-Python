"""
Abinitio Pipeline - Experimental Data Empiar 10028
==================================================

This notebook introduces a selection of
components corresponding to loading real Relion picked
particle cryo-EM data and running key ASPIRE-Python
Abinitio model components as a pipeline.

Specifically this pipeline uses the
EMPIAR 10028 picked particles data, available here:

https://www.ebi.ac.uk/empiar/EMPIAR-10028

https://www.ebi.ac.uk/emdb/EMD-2660
"""

# %%
# Imports
# -------
# First import some of the usual suspects.
# In addition, import some classes from
# the ASPIRE package that will be used throughout this experiment.

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aspire.abinitio import CLSync3N
from aspire.denoising import DefaultClassAvgSource, DenoisedSource, DenoiserCov2D
from aspire.noise import AnisotropicNoiseEstimator
from aspire.reconstruction import MeanEstimator
from aspire.source import OrientedSource, RelionSource

logger = logging.getLogger(__name__)


# %%
# Parameters
# ---------------
# Example simulation configuration.

interactive = False  # Draw blocking interactive plots?
do_cov2d = True  # Use CWF coefficients
n_imgs = None  # Set to None for all images in starfile, can set smaller for tests.
img_size = 32  # Downsample the images/reconstruction to a desired resolution
n_classes = 2000  # How many class averages to compute.
n_nbor = 100  # How many neighbors to stack
starfile_in = "10028/data/shiny_2sets_fixed9.star"
data_folder = "."  # This depends on the specific starfile entries.
volume_output_filename = f"10028_abinitio_c{n_classes}_m{n_nbor}_{img_size}.mrc"
pixel_size = 1.34


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
src = src.downsample(img_size)

# Peek
if interactive:
    src.images[:10].show()

# Use phase_flip to attempt correcting for CTF.
logger.info("Perform phase flip to input images.")
src = src.phase_flip()

# Estimate the noise and `Whiten` based on the estimated noise
aiso_noise_estimator = AnisotropicNoiseEstimator(src)
src = src.whiten(aiso_noise_estimator)

# Plot the noise profile for inspection
if interactive:
    plt.imshow(aiso_noise_estimator.filter.evaluate_grid(img_size))
    plt.show()

# Peek, what do the whitened images look like...
if interactive:
    src.images[:10].show()

# # Optionally invert image contrast, depends on data convention.
# # This is not needed for 10028, but included anyway.
# logger.info("Invert the global density contrast")
# src = src.invert_contrast()

# Caching is used for speeding up large datasets on high memory machines.
src = src.cache()

# %%
# Optional: CWF Denoising
# -----------------------
#
# Optionally generate an alternative source that is denoised with `cov2d`,
# then configure a customized averager. This allows the use of CWF denoised
# images for classification, but stacks the original images for averages
# used in the remainder of the reconstruction pipeline.
#
# In this example, this behavior is controlled by the `do_cov2d` boolean variable.
# When disabled, the original src and default averager is used.
# If you will not be using cov2d,
# you may remove this code block and associated variables.

classification_src = src
custom_averager = None
if do_cov2d:
    # Use CWF denoising
    cwf_denoiser = DenoiserCov2D(src)
    # Use denoised src for classification
    classification_src = DenoisedSource(src, cwf_denoiser)
    # Cache for speedup.  Avoids recomputing.
    classification_src = classification_src.cache()
    # Peek, what do the denoised images look like...
    if interactive:
        classification_src.images[:10].show()

# %%
# Class Averaging
# ----------------------
#
# Now perform classification and averaging for each class.

logger.info("Begin Class Averaging")

# Now perform classification and averaging for each class.
# This also demonstrates the potential to use a different source for classification and averaging.

avgs = DefaultClassAvgSource(
    classification_src,
    n_nbor=n_nbor,
    averager_src=src,
)
# We'll continue our pipeline with the first `n_classes` from `avgs`.
avgs = avgs[:n_classes].cache()

# Save off the set of class average images.
avgs.save("experimental_10028_class_averages.star", overwrite=True)

if interactive:
    avgs.images[:10].show()


# %%
# Common Line Estimation
# ----------------------
#
# Next create a CL instance for estimating orientation of projections
# using the Common Line with Synchronization Voting method.

logger.info("Begin Orientation Estimation")

# Create a custom orientation estimation object for ``avgs``.
orient_est = CLSync3N(avgs, n_theta=72)

# Create an ``OrientedSource`` class instance that performs orientation
# estimation in a lazy fashion upon request of images or rotations.
oriented_src = OrientedSource(avgs, orient_est)

# %%
# Volume Reconstruction
# ----------------------
#
# Using the oriented source, attempt to reconstruct a volume.

logger.info("Begin Volume reconstruction")

# Setup an estimator to perform the back projection.
estimator = MeanEstimator(oriented_src)

# Perform the estimation and save the volume.
estimated_volume = estimator.estimate()
estimated_volume.save(volume_output_filename, overwrite=True)
logger.info(f"Saved Volume to {str(Path(volume_output_filename).resolve())}")

# Peek at result
if interactive:
    plt.imshow(np.sum(estimated_volume.asnumpy()[0], axis=-1))
    plt.show()
