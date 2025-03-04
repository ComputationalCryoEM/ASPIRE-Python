"""
Abinitio Pipeline - Experimental Data EMPIAR 10028
==================================================

This notebook introduces a selection of
components corresponding to loading real Relion picked
particle cryo-EM data and running key ASPIRE-Python
Abinitio model components as a pipeline.

This demonstrates reproducing results similar to those found in::

  Common lines modeling for reference free Ab-initio reconstruction in cryo-EM
  Journal of Structural Biology 2017
  https://doi.org/10.1016/j.jsb.2017.09.007

Specifically this pipeline uses the
EMPIAR 10028 picked particles data, available here:

https://www.ebi.ac.uk/empiar/EMPIAR-10028
"""

# %%
# Imports
# -------
# Import packages that will be used throughout this experiment.

import logging
from pathlib import Path

from aspire.denoising import LegacyClassAvgSource
from aspire.reconstruction import MeanEstimator
from aspire.source import OrientedSource, RelionSource

logger = logging.getLogger(__name__)


# %%
# Parameters
# ---------------
#
# Use of GPU is expected for a large configuration.
# If running on a less capable machine, or simply experimenting, it is
# strongly recommened to reduce ``img_size``, ``_n_imgs``, and
# ``n_nbor``.

# Inputs
starfile_in = "10028/data/shiny_2sets_fixed9.star"
data_folder = "."  # This depends on the specific starfile entries.
pixel_size = 1.34  # Defined with the dataset from EMPIAR

# Config
n_imgs = None  # Set to None for all images in starfile, can set smaller for tests.
img_size = 179  # Downsample the images/reconstruction to a desired resolution
n_classes = 3000  # How many class averages to compute.
n_nbor = 50  # How many neighbors to stack

# Outputs
preprocessed_fn = f"10028_preprocessed_{img_size}px.star"
oriented_fn = f"10028_oriented_class_averages_{img_size}px.star"
volume_output_filename = f"10028_abinitio_c{n_classes}_m{n_nbor}_{img_size}px.mrc"


# %%
# Source data and Preprocessing
# -----------------------------
#
# ``RelionSource`` is used to access the experimental data via a `STAR` file.
# Begin by preprocessing to correct for CTF, then downsample to ``img_size``
# and apply noise correction.
#
# ASPIRE-Python has the ability to automatically adjust CTF filters
# for downsampling, and this can be employed simply by changing the
# order of preprocessing steps, saving time by phase flipping lower
# resolution images.  Howver,tThis script intentionally follows the order
# described in the original publication.

# Create a source object for the experimental images
src = RelionSource(
    starfile_in, pixel_size=pixel_size, max_rows=n_imgs, data_folder=data_folder
)

# Use phase_flip to attempt correcting for CTF.
# Caching is used throughout for speeding up large datasets on high memory machines.
logger.info("Perform phase flip to input images.")
src = src.phase_flip().cache()

# Downsample the images.
logger.info(f"Set the resolution to {img_size} X {img_size}")
src = src.downsample(img_size).cache()

# Normalize the background of the images.
src = src.normalize_background().cache()

# Estimate the noise and whiten based on the estimated noise.
src = src.legacy_whiten().cache()

# Optionally invert image contrast.
logger.info("Invert the global density contrast")
src = src.invert_contrast().cache()

# Save the preprocessed images.
# These can be resused to experiment with later stages of the pipeline
# without repeating the preprocessing computations.
src.save(preprocessed_fn, save_mode="single", overwrite=True)

# %%
# Class Averaging
# ----------------------
#
# Now perform classification and averaging for each class.

logger.info("Begin Class Averaging")
avgs = LegacyClassAvgSource(src, n_nbor=n_nbor)

# We'll continue our pipeline with the first ``n_classes`` from
# ``avgs``.  The classes will be selected by the ``class_selector`` of a
# ``ClassAvgSource``, which in this case will be the class averages
# having the largest variance.  Note global sorting requires computing
# all class averages, which is computationally intensive.
avgs = avgs[:n_classes].cache()


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

# Save off the selected set of class average images, along with the
# estimated orientations and shifts.  These can be reused to
# experiment with alternative volume reconstructions.
oriented_src.save(oriented_fn, save_mode="single", overwrite=True)

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
