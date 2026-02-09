"""
Abinitio Pipeline - Experimental Data EMPIAR 11618
==================================================

This notebook introduces a selection of
components corresponding to loading real Relion picked
particle cryo-EM data and running key ASPIRE-Python
ab initio model components as a pipeline.

Specifically this pipeline uses the
EMPIAR 11618 picked particles data, available here:

https://www.ebi.ac.uk/empiar/EMPIAR-11618
"""

# %%
# Imports
# -------
# Import packages that will be used throughout this experiment.

import logging
from pathlib import Path

from aspire.classification import RandomClassSelector
from aspire.denoising import LegacyClassAvgSource
from aspire.reconstruction import MeanEstimator
from aspire.source import OrientedSource, RelionSource

logger = logging.getLogger(__name__)


# %%
# Parameters
# ---------------
#
# Use of GPU is expected for a large configuration.

# Inputs
# User is responsible for downloading `11618` raw data from EMPIAR,
# and assigning correct path.
input_dir = "11618/data/particles/"
starfile_in = f"{input_dir}/J43_particles.star"
data_folder = f"{input_dir}"

# Config
n_imgs = None  # Set to None for all images in starfile; set smaller for tests
img_size = 129  # Downsample the images/reconstruction to a desired resolution
n_classes = 1000  # Number of class averages to use for reconstruction
n_nbor = 32  # How many neighbors to stack for each class average

# Outputs
preprocessed_fn = f"11618_preprocessed_pf_ds{img_size}px_nbg_wt_inv.star"
class_avg_fn = f"11618_rand{n_classes}_cls_avgs_m{n_nbor}_{img_size}px.star"
volume_output_filename = f"11618_abinitio_c{n_classes}_m{n_nbor}_{img_size}px.mrc"

# %%
# Source data and Preprocessing
# -----------------------------
#
# ``RelionSource`` is used to access the experimental data via a `STAR` file.
# Begin by preprocessing to correct for CTF, then downsample to ``img_size``
# and apply noise correction.


# Create a source object for the experimental images
src = RelionSource(starfile_in, max_rows=n_imgs, data_folder=data_folder)

# Use phase_flip to attempt correcting for CTF.
# Caching is used throughout for speeding up large datasets on high memory machines.
logger.info("Perform phase flip to input images.")
src = src.phase_flip().cache()

# Downsample the images.
logger.info(f"Set the resolution to {img_size} X {img_size}")
src = src.downsample(img_size).cache()

logger.info("Apply noise correction to images")
# Normalize the background of the images.
src = src.normalize_background().cache()

# Estimate the noise and whiten based on the estimated noise.
src = src.whiten().cache()

# Optionally invert image contrast.
logger.info("Invert the global density contrast if needed")
src = src.invert_contrast().cache()

# Save the preprocessed images.
# These can be reused to experiment with later stages of the pipeline
# without repeating the preprocessing computations.
src.save(preprocessed_fn, save_mode="single", overwrite=True)

# %%
# Class Averaging
# ----------------------
#
# Now perform classification and averaging for each class.
# `RandomClassSelector` will randomize the class averages chosen for
# reconstruction.  This avoids having to compute all class averages
# and sort them by some notion of quality, which is computationally
# intensive.  Instead this example computes a small random sample of
# classes from the entire data set.

logger.info("Begin Class Averaging")
avgs = LegacyClassAvgSource(src, class_selector=RandomClassSelector(), n_nbor=n_nbor)

# Compute and cache the random set of `n_classes` from `src`.
avgs = avgs[:n_classes].cache()

# Save the random set of class averages to disk so they can be re-used.
avgs.save(class_avg_fn, save_mode="single", overwrite=True)


# %%
# Common Line Estimation
# ----------------------
#
# Estimate orientation of projections and assign to source by
# applying ``OrientedSource`` to the class averages from the prior
# step. By default this applies the Common Line with Synchronization
# Voting ``CLSync3N`` method.

logger.info("Begin Orientation Estimation")
oriented_src = OrientedSource(avgs)

# %%
# Volume Reconstruction
# ----------------------
#
# Using the oriented source, attempt to reconstruct a volume.

logger.info("Begin Volume reconstruction")
# Set up an estimator to perform the backprojection.
estimator = MeanEstimator(oriented_src)

# Perform the estimation and save the volume.
estimated_volume = estimator.estimate()
estimated_volume.save(volume_output_filename, overwrite=True)
logger.info(f"Saved Volume to {str(Path(volume_output_filename).resolve())}")
