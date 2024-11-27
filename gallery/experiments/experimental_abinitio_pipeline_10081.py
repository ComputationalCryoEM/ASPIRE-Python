"""
Abinitio Pipeline - Experimental Data Empiar 10081
==================================================

This notebook introduces a selection of
components corresponding to loading real Relion picked
particle cryo-EM data and running key ASPIRE-Python
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
from pathlib import Path

from aspire.abinitio import CLSymmetryC3C4
from aspire.denoising import ClassAvgSourceLegacy
from aspire.noise import AnisotropicNoiseEstimator
from aspire.reconstruction import MeanEstimator
from aspire.source import OrientedSource, RelionSource

logger = logging.getLogger(__name__)


# %%
# Parameters
# ---------------
# Example simulation configuration.

n_imgs = None  # Set to None for all images in starfile, can set smaller for tests.
img_size = 32  # Downsample the images/reconstruction to a desired resolution
n_classes = 500  # How many class averages to compute.
n_nbor = 100  # How many neighbors to stack
starfile_in = "10081/data/particle_stacks/data.star"
data_folder = "."  # This depends on the specific starfile entries.
volume_output_filename = f"10081_abinitio_c{n_classes}_m{n_nbor}_{img_size}.mrc"
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
    starfile_in,
    pixel_size=pixel_size,
    max_rows=n_imgs,
    data_folder=data_folder,
    symmetry_group="C4",
)

# Downsample the images
logger.info(f"Set the resolution to {img_size} X {img_size}")
src = src.downsample(img_size)

# Use phase_flip to attempt correcting for CTF.
logger.info("Perform phase flip to input images.")
src = src.phase_flip()

# Estimate the noise and `Whiten` based on the estimated noise
aiso_noise_estimator = AnisotropicNoiseEstimator(src)
src = src.whiten(aiso_noise_estimator.filter)

# Caching is used for speeding up large datasets on high memory machines.
src = src.cache()

# %%
# Class Averaging
# ----------------------
#
# Now perform classification and averaging for each class.

logger.info("Begin Class Averaging")

# Now perform classification and averaging for each class.
# Automatically configure parallel processing
avgs = ClassAvgSourceLegacy(src, n_nbor=n_nbor)

# We'll continue our pipeline with the first ``n_classes`` from ``avgs``.
avgs = avgs[:n_classes]

# Save off the set of class average images.
avgs.save("experimental_10081_class_averages.star", overwrite=True)

# %%
# Common Line Estimation
# ----------------------
#
# Next create a CL instance for estimating orientation of projections
# using the Common Line with Synchronization Voting method.

logger.info("Begin Orientation Estimation")

# Create a custom orientation estimation object for ``avgs``.
orient_est = CLSymmetryC3C4(avgs, symmetry="C4", n_theta=72, max_shift=0)

# Create an ``OrientedSource`` class instance that performs orientation
# estimation in a lazy fashion upon request of images or rotations.
oriented_src = OrientedSource(avgs, orient_est)

# %%
# Volume Reconstruction
# ----------------------
#
# Using the oriented source, attempt to reconstruct a volume.  Since
# this is a Cn symmetric molecule, as specified by ``RelionSource(...,
# symmetry_group="C4", ...)``, the ``symmetry_group`` source attribute
# will flow through the pipeline to ``avgs``. Then each image will be
# repeated for each of the 3 additional rotations during
# back-projection.  This boosts the effective number of images used in
# the reconstruction from ``n_classes`` to ``4*n_classes``.

logger.info("Begin Volume reconstruction")


# Setup an estimator to perform the back projection.
estimator = MeanEstimator(oriented_src)

# Perform the estimation and save the volume.
estimated_volume = estimator.estimate()
estimated_volume.save(volume_output_filename, overwrite=True)
logger.info(f"Saved Volume to {str(Path(volume_output_filename).resolve())}")
