"""
Abinitio Pipeline - Experimental Data EMPIAR 10073
==================================================

This notebook introduces a selection of
components corresponding to loading real Relion picked
particle cryo-EM data and running key ASPIRE-Python
ab initio model components as a pipeline.

This demonstrates reproducing results similar to those found in:

.. admonition:: Publication

   | Common lines modeling for reference free Ab-initio reconstruction in cryo-EM
   | Journal of Structural Biology 2017
   | https://doi.org/10.1016/j.jsb.2017.09.007

Specifically this pipeline uses the
EMPIAR 10073 picked particles data, available here:

https://www.ebi.ac.uk/empiar/EMPIAR-10073
"""

# %%
# Imports
# -------
# Import packages that will be used throughout this experiment.

import logging
from pathlib import Path

import numpy as np

from aspire.abinitio import CLSync3N
from aspire.basis import DiracBasis3D
from aspire.denoising import LegacyClassAvgSource
from aspire.reconstruction import MeanEstimator
from aspire.source import ArrayImageSource, OrientedSource, RelionSource
from aspire.utils import fuzzy_mask

logger = logging.getLogger(__name__)


# %%
# Parameters
# ---------------
#
# Use of GPU is expected for a large configuration.
# If running on a less capable machine, or simply experimenting, it is
# strongly recommended to reduce ``img_size``, ``n_imgs``, and
# ``n_nbor``.

# Inputs
starfile_in = "10073/data/shiny_correctpaths_cleanedcorruptstacks.star"
data_folder = "."  # This depends on the specific starfile entries.
pixel_size = 1.43  # Defined with the dataset from EMPIAR

# Config
n_imgs = None  # Set to None for all images in starfile, can set smaller for tests.
img_size = 129  # Downsample the images/reconstruction to a desired resolution
n_classes = 3000  # How many class averages to compute.
n_nbor = 50  # How many neighbors to stack

# Outputs
preprocessed_fn = f"10073_preprocessed_{img_size}px.star"
class_avg_fn = f"10073_var_sorted_cls_avgs_m{n_nbor}_{img_size}px.star"
oriented_fn = f"10073_oriented_class_averages_{img_size}px.star"
volume_output_filename = f"10073_abinitio_c{n_classes}_m{n_nbor}_{img_size}px.mrc"


# %%
# Source data and Preprocessing
# -----------------------------
#
# ``RelionSource`` is used to access the experimental data via a `STAR` file.

# Create a source object for the experimental images.
src = RelionSource(
    starfile_in, pixel_size=pixel_size, max_rows=n_imgs, data_folder=data_folder
)

# Use ``phase_flip`` to attempt correcting for CTF.
logger.info("Perform phase flip to input images.")
src = src.phase_flip().cache()

# Legacy MATLAB right cropped the images to an odd resolution.
src = src.crop(src.L - 1).cache()

# Downsample the images.
logger.info(f"Set the resolution to {img_size} X {img_size}")
src = src.legacy_downsample(img_size).cache()

# Normalize the background of the images.
src = src.legacy_normalize_background().cache()

# Estimate the noise and whiten based on the estimated noise.
src = src.legacy_whiten().cache()

# Optionally invert image contrast.
logger.info("Invert the global density contrast")
src = src.invert_contrast().cache()

# Save the preprocessed image stack.
src.save(preprocessed_fn, save_mode="single", overwrite=True)


# %%
# Class Averaging
# ----------------------
#
# Now perform classification and averaging for each class.

logger.info("Begin Class Averaging")

avgs = LegacyClassAvgSource(src, n_nbor=n_nbor).cache()

# Save the entire set of class averages to disk so they can be reused.
avgs.save(class_avg_fn, save_mode="single", overwrite=True)

# We'll continue our pipeline by selecting ``n_classes`` from ``avgs``.
# To capture a broader range of viewing angles, uniformly select every ``k`` image.
k = (avgs.n - 1) // n_classes
avgs = avgs[::k].cache()


# %%
# Common Line Estimation
# ----------------------
#
# Estimate orientation of projections and assign to source by
# applying ``OrientedSource`` to the class averages from the prior
# step. By default this applies the Common Line with Synchronization
# Voting ``CLSync3N`` method.  Here additional weighting techniques
# are applied for common lines detection by customizing the
# orientation estimator component.

logger.info("Apply custom mask")
# 10073 benefits from a masking procedure that is more aggressive than the default.
# Note, since we've manually masked, the default masking is disabled below in `CLSync3N`.
# This also upcasts to double precision, which is helpful for this reconstruction.
mask = fuzzy_mask((img_size, img_size), np.float64, r0=0.4 * img_size, risetime=2)
avgs = ArrayImageSource(avgs.images[:] * mask)

logger.info("Begin Orientation Estimation")
# Configure the `CLSync3N` algorithm,
#  customized by enabling weighting and disabling default mask.
ori_est = CLSync3N(avgs, mask=False, S_weighting=True, J_weighting=True)
# Handles calling code to find and assign orientations and shifts.
oriented_src = OrientedSource(avgs, ori_est)

# Save off the set of class average images, along with the estimated orientations and shifts.
oriented_src.save(oriented_fn, save_mode="single", overwrite=True)


# %%
# Volume Reconstruction
# ----------------------
#
# Using the oriented source, attempt to reconstruct a volume.

logger.info("Begin Volume reconstruction")

# Set up an estimator to perform the backprojection.
# Legacy MATLAB FIRM used Dirac basis.
basis3d = DiracBasis3D(oriented_src.L)
estimator = MeanEstimator(oriented_src, basis=basis3d)

# Perform the estimation and save the volume.
estimated_volume = estimator.estimate()
estimated_volume.save(volume_output_filename, overwrite=True)
logger.info(f"Saved Volume to {str(Path(volume_output_filename).resolve())}")
