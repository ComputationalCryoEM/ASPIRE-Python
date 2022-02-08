"""
ASPIRE-Python Abinitio Pipeline
================================

In this notebook we will introduce a selection of
components corresponding to a pipeline using
the EMD-10028 picked particles dataset.
"""

# %%
# Imports
# -------
# First we import some of the usual suspects.
# In addition, we import some classes from
# the ASPIRE package that we will use throughout this experiment.

import logging

import matplotlib.pyplot as plt
import numpy as np

from aspire.abinitio import CLSyncVoting
from aspire.basis import FFBBasis2D, FFBBasis3D
from aspire.classification import BFSReddyChatterjiAlign2D, RIRClass2D
from aspire.denoising import DenoiserCov2D
from aspire.noise import AnisotropicNoiseEstimator
from aspire.operators import FunctionFilter, RadialCTFFilter
from aspire.reconstruction import MeanEstimator
from aspire.source import RelionSource
from aspire.volume import Volume

logger = logging.getLogger(__name__)


# %%
# Parameters
# ---------------
# Example simulation configuration.

interactive = False  # Do we want to draw blocking interactive plots?
do_cov2d = True  # Use CWF coefficients
n_imgs = None  # Set to None for all images in starfile
img_size = 77  # Downsample the images/reconstruction to a desired resolution
n_classes = 2000  # How many class averages to compute.
n_nbor = 100  # How many neighbors to stack
starfile_in = "10028/data/shiny_2sets.star"
volume_filename_prefix_out = f"10028_recon_{num_imgs}_c{n_classes}_m{n_nbor}_{img_size}.mrc"
pixel_size = 1.34

# Create a source object for the experimental images
src = RelionSource(
    starfile_in, pixel_size=pixel_size, max_rows=n_imgs
)

# Downsample the images
logger.info(f"Set the resolution to {img_size} X {img_size}")
src.downsample(img_size)

# Peek
if interactive:
    src.images(0, 10).show()

# Currently we use phase_flip to attempt correcting for CTF.
logger.info("Perform phase flip to input images.")
src.phase_flip()

# We should estimate the noise and `Whiten` based on the estimated noise
aiso_noise_estimator = AnisotropicNoiseEstimator(src)
src.whiten(aiso_noise_estimator.filter)

# Plot the noise profile for inspection
if interactive:
    plt.imshow(aiso_noise_estimator.filter.evaluate_grid(img_size))
    plt.show()

# Peek, what do the whitened images look like...
if interactive:
    src.images(0, 10).show()

# # Optionally invert image contrast, depends on data.
# logger.info("Invert the global density contrast")
# src.invert_contrast()

# On Simulation data, better results so far were achieved without cov2d
# However, we can demonstrate using CWF denoised images for classification.
classification_src = src
custom_aligner = None
if do_cov2d:
    # Use CWF denoising
    cwf_denoiser = DenoiserCov2D(src)
    # Use denoised src for classification
    classification_src = cwf_denoiser.denoise()
    # Peek, what do the denoised images look like...
    if interactive:
        classification_src.images(0, 10).show()

    # Use regular `src` for the alignment and composition (averaging).
    composite_basis = FFBBasis2D((src.L,) * 2, dtype=src.dtype)
    custom_aligner = BFSReddyChatterjiAlign2D(
        None, src, composite_basis, dtype=src.dtype
    )


# %%
# Class Averaging
# ----------------------
#
# Now we perform classification and averaging for each class.

logger.info("Begin Class Averaging")

rir = RIRClass2D(
    classification_src,  # Source used for classification
    fspca_components=400,
    bispectrum_components=300,  # Compressed Features after last PCA stage.
    n_nbor=n_nbor,
    n_classes=n_classes,
    large_pca_implementation="legacy",
    nn_implementation="sklearn",
    bispectrum_implementation="legacy",
    aligner=custom_aligner,
)

classes, reflections, distances = rir.classify()
# Only care about the averages returned right now.
avgs = rir.averages(classes, reflections, distances)[0]
if interactive:
    avgs.images(0, 10).show()

# %%
# Common Line Estimation
# ----------------------
#
# Now we can create a CL instance for estimating orientation of projections
# using the Common Line with Synchronization Voting method.

logger.info("Begin Orientation Estimation")

orient_est = CLSyncVoting(avgs, n_theta=36)
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
avgs.rots = rots_est

# Create a reasonable Basis for the 3d Volume
basis = FFBBasis3D((img_size,) * 3, dtype=src.dtype)

# Setup an estimator to perform the back projection.
estimator = MeanEstimator(avgs, basis)

# Perform the estimation and save the volume.
estimated_volume = estimator.estimate()
estimated_volume.save(volume_filename_prefix_out, overwrite=True)

# Peek at result
if interactive:
    plt.imshow(np.sum(estimated_volume[0], axis=-1))
    plt.show()
