"""
Abinitio Halfset Pipeline - Experimental Data
=============================================

This demonstrates creating two half sets of experimental data,
performing independent reconstructions, and computing resulting FSC.

Specifically this pipeline uses the EMPIAR 10028 picked particles:

https://www.ebi.ac.uk/empiar/EMPIAR-10028
"""

# %%
# Imports
# -------

import logging
from pathlib import Path

from aspire.denoising import LegacyClassAvgSource
from aspire.reconstruction import MeanEstimator
from aspire.source import OrientedSource, RelionSource

logger = logging.getLogger(__name__)


# %%
# Parameters
# ---------------
# Example configuration.
#
# Use of GPU is expected for a large configuration.  If running on a
# less capable machine, or simply experimenting, it is strongly
# recommended to reduce the problem size by altering ``img_size``,
# ``n_classes``, ``n_nbor``, ``max_rows`` etc.

img_size = 179  # Downsample the images/reconstruction to a desired resolution
n_classes = 3000  # How many class averages to compute.
n_nbor = 50  # How many neighbors to stack
starfile_in = "10028/data/shiny_2sets_fixed9.star"
data_folder = "."  # This depends on the specific starfile entries.
pixel_size = 1.34
fsc_cutoff = 0.143

# %%
# Load and split Source data
# --------------------------
#
# ``RelionSource`` is used to access the experimental data via a `STAR` file.

# Create a source object loading all the experimental images
src = RelionSource(
    starfile_in, pixel_size=pixel_size, max_rows=None, data_folder=data_folder
)

# Split the data into two sets.
# This example uses evens and odds for simplicity, but random indices
# can also be used with similar results.

srcA = src[::2]
srcB = src[1::2]

# A dictionary can systematically organize our inputs and outputs for both sets.
pipelines = {
    "A": {
        "input": srcA,
    },
    "B": {
        "input": srcB,
    },
}

# %%
# Dual Pipelines
# --------------
# Preprocess by downsampling, correcting for CTF, and applying noise
# correction.  After preprocessing, class averages are generated then
# automatically selected for use in reconstruction.
#
# Each of the above steps are performed totally independently
# for each dataset, first for A, then for B.
#
# Caching is used throughout for speeding up large datasets on high memory machines.

for src_id, pipeline in pipelines.items():

    src = pipeline["input"]

    # Downsample the images
    logger.info(f"Set the resolution to {img_size} X {img_size}")
    src = src.downsample(img_size).cache()

    # Use phase_flip to attempt correcting for CTF.
    logger.info(f"Perform phase flip to {len(src)} input images for set {src_id}.")
    src = src.phase_flip().cache()

    # Normalize the background of the images.
    src = src.normalize_background().cache()

    # Estimate the noise and whiten based on the estimated noise.
    src = src.legacy_whiten().cache()

    # Optionally invert image contrast.
    logger.info("Invert the global density contrast")
    src = src.invert_contrast().cache()

    # Now perform classification and averaging for each class.
    logger.info("Begin Class Averaging")

    avgs = LegacyClassAvgSource(src, n_nbor=n_nbor)
    avgs = avgs[:n_classes].cache()

    # Common Line Estimation
    logger.info("Begin Orientation Estimation")
    oriented_src = OrientedSource(avgs)

    # Volume Reconstruction
    logger.info("Begin Volume reconstruction")

    # Setup an estimator to perform the back projection.
    estimator = MeanEstimator(oriented_src)

    # Perform the estimation and save the volume.
    pipeline["volume_output_filename"] = fn = (
        f"10028_abinitio_c{n_classes}_m{n_nbor}_{img_size}px_{src_id}.mrc"
    )
    estimated_volume = estimator.estimate()
    estimated_volume.save(fn, overwrite=True)
    logger.info(f"Saved Volume to {str(Path(fn).resolve())}")

    # Store volume result in pipeline dict.
    pipeline["estimated_volume"] = estimated_volume

# %%
# Compute FSC Score
# -----------------
# At this point both pipelines have completed reconstructions and may be compared using FSC.

# Recall our resulting volumes from the dictionary.
vol_a = pipelines["A"]["estimated_volume"]
vol_b = pipelines["B"]["estimated_volume"]

# Compute the FSC
# Save plot, in case display is not available.
vol_a.fsc(vol_b, cutoff=fsc_cutoff, plot="fsc_plot.png")
# Display plot and report
fsc, _ = vol_a.fsc(vol_b, cutoff=fsc_cutoff, plot=True)
logger.info(
    f"Found FSC of {fsc} Angstrom at cutoff={fsc_cutoff} and pixel size {vol_a.pixel_size} Angstrom/pixel."
)
