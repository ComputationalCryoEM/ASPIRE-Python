"""
Abinitio Pipeline - Experimental Data Empiar 10081
==================================================

This notebook introduces a selection of
components corresponding to loading real Relion picked
particle cryo-EM data and running key ASPIRE-Python
Abinitio model components as a pipeline. Some of the
components are tailored specifically to handle the C4
cyclic symmetry exhibited by this dataset.

This demonstrates reproducing results similar to those found in:

.. admonition:: Publication

   | Gabi Pragier and Yoel Shkolnisky
   | A common lines approach for ab-initio modeling of cyclically-symmetric molecules
   | Inverse Problems, 35(12), p.124005, 2019.
   | DOI 10.1088/1361-6420/ab2fb2

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
from aspire.denoising import LegacyClassAvgSource
from aspire.noise import AnisotropicNoiseEstimator
from aspire.reconstruction import MeanEstimator
from aspire.source import OrientedSource, RelionSource

logger = logging.getLogger(__name__)


# %%
# Parameters
# ---------------

# Use of GPU is expected for a large configuration.
# If running on a less capable machine, or simply experimenting, it is
# strongly recommended to reduce ``img_size``, ``n_imgs``, and
# ``n_nbor``.

# Inputs
starfile_in = "10081/data/particle_stacks/data.star"
data_folder = "."  # This depends on the specific starfile entries.
pixel_size = 1.3

# Config
n_imgs = None  # Set to None for all images in starfile, can set smaller for tests.
img_size = 129  # Downsample the images/reconstruction to a desired resolution
n_classes = 5000  # How many class averages to compute.
n_nbor = 50  # How many neighbors to stack

# Outputs
preprocessed_fn = f"10081_preprocessed_{img_size}px.star"
class_avg_fn = f"10081_var_sorted_cls_avgs_m{n_nbor}_{img_size}px.star"
oriented_fn = f"10081_oriented_class_averages_{img_size}px.star"
volume_output_filename = f"10081_abinitio_c{n_classes}_m{n_nbor}_{img_size}.mrc"


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

# Use phase_flip to attempt correcting for CTF.
logger.info("Perform phase flip to input images.")
src = src.phase_flip().cache()

# Legacy MATLAB cropped the images to an odd resolution.
src = src.crop_pad(src.L - 1).cache()

# Downsample the images.
logger.info(f"Set the resolution to {img_size} X {img_size}")
src = src.legacy_downsample(img_size).cache()

# Estimate the noise and whiten based on the estimated noise.
src = src.legacy_whiten().cache()

# Optionally invert image contrast.
logger.info("Invert the global density contrast")
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

logger.info("Begin Class Averaging")

# Now perform classification and averaging for each class.
# Automatically configure parallel processing
avgs = LegacyClassAvgSource(src, n_nbor=n_nbor)

# Save the entire set of class averages to disk so they can be re-used.
avgs.save(class_avg_fn, save_mode="single", overwrite=True)

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
# Create a custom orientation estimation object for ``avgs``.
# Here we use the ``CLSymmetryC3C4`` algorithm, which is
# designed for molecules with C3 or C4 symmetry.

logger.info("Begin Orientation Estimation")
orient_est = CLSymmetryC3C4(avgs, symmetry="C4")

# Create an ``OrientedSource`` class instance that performs orientation
# estimation in a lazy fashion upon request of images or rotations.
oriented_src = OrientedSource(avgs, orient_est)

# Save off the selected set of class average images, along with the
# estimated orientations and shifts.  These can be reused to
# experiment with alternative volume reconstructions.
oriented_src.save(oriented_fn, save_mode="single", overwrite=True)

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
