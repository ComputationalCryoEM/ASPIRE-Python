"""
Symmetric NUG Pipeline
======================

This notebook demonstrates reproducing simliar results to the
experiment found in:

.. admonition:: Publication

   | Shi, Y., Singer, A., Wang, J., & Yang, R.
   | Orientation estimation of cryo-EM images for molecules with
   | arbitrary symmetry by semidefinite programming

This pipeline begins with 17 class averages that were generated (as described
in the publication) from the EMPIAR 10081 picked particle data, available here:

https://www.ebi.ac.uk/empiar/EMPIAR-10081

First, orientation estimation is performed using the Non-Unique Games
algorithm adapted for handling symmetric particles. A volume is then
reconstructed from the estimated orientations and saved to an output
file for inspection. The reconstructed volume can be compared with the
EMPIAR-10081 associated reconstruction found here:

https://www.ebi.ac.uk/emdb/EMD-8511
"""

# %%
# Imports
# -------
# First import the necessary utilities and ASPIRE
# classes used throughout the notebook.
import logging

import numpy as np

from aspire.abinitio import CommonlineNUG
from aspire.downloader import nug_10081
from aspire.image import Image
from aspire.reconstruction import MeanEstimator
from aspire.source import ArrayImageSource, OrientedSource

logger = logging.getLogger(__name__)


# %%
# Setup
# -----

# Note: Use of GPU is recommended for the NUG algorithm.
# Below we setup some output files to save off orientation
# and reconstruction results.
oriented_fn = "nug_10081_oriented.star"
volume_output_fn = "nug_10081_abinitio.mrc"


# %%
# Load the Dataset
# ----------------
#
# The 17 class averages can be accessed as an ASPIRE ``Image`` object
# via the built-in downloader utility.

logger.info("Load Precomputed Class Averages")
ims = nug_10081()
avgs = ArrayImageSource(ims, pixel_size=1.3, symmetry_group="C4")


# %%
# Orientation Estimation
# ----------------------
#
# Create an orientation estimation object for the ``avgs``.
# The ``CommonlineNUG`` algorithm will detect the symmetry from
# the ``avgs`` metadata and use the approriate NUG ADMM-solver to estimate
# orientations. To replicate the original experiment as closely as possible,
# we use the 15 iterations of proximal refinement and disable searching over
# the commonline shift space.

logger.info("Begin Orientation Estimation")
orient_est = CommonlineNUG(avgs, max_shift=0, pr_iters=15)

# Create an ``OrientedSource`` class instance that performs orientation
# estimation in a lazy fashion upon request of images or rotations.
oriented_src = OrientedSource(avgs, orient_est)

# Estimate orientations and override the estimated shifts.
# In the case of insufficient number of images, estimating shifts fails.
oriented_src.rotations
oriented_src = oriented_src.update(offsets=0)

# Confirm offsets have been set to zero.
logger.info(f"Offsets: {oriented_src.offsets}")

# Save oriented source.
oriented_src.save(oriented_fn)

# %%
# Volume Reconstruction
# ----------------------
#
# Using the oriented source, attempt to reconstruct a volume.  Since
# this is a C4 symmetric molecule the ``symmetry_group`` source attribute
# will flow through the pipeline to ``oriented_src``. Then each image will be
# repeated for each of the 3 additional rotations during
# back-projection. In general, this boosts the effective number of images used in
# the reconstruction from ``n_imgs`` to ``4 * n_imgs``, or from 17 to 68 in this case.

# Setup an estimator to perform the back projection.
logger.info("Begin Volume reconstruction")
estimator = MeanEstimator(oriented_src)

# Perform the estimation and save the reconstructed volume.
est_vol = estimator.estimate()
est_vol.save(volume_output_fn)
