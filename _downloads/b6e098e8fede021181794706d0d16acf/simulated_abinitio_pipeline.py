"""
Abinitio Pipeline - Simulated Data
==================================

This notebook introduces a selection of
components corresponding to generating realistic
simulated cryo-EM data and running key ASPIRE-Python
Abinitio model components as a pipeline.
"""

# %%
# Imports
# -------
# First import some of the usual suspects.
# In addition, import some classes from
# the ASPIRE package that will be used throughout this experiment.

import logging

import matplotlib.pyplot as plt
import numpy as np

from aspire.abinitio import CLSyncVoting
from aspire.denoising import DefaultClassAvgSource, DenoisedSource, DenoiserCov2D
from aspire.downloader import emdb_2660
from aspire.noise import AnisotropicNoiseEstimator, CustomNoiseAdder
from aspire.operators import FunctionFilter, RadialCTFFilter
from aspire.reconstruction import MeanEstimator
from aspire.source import OrientedSource, Simulation
from aspire.utils import mean_aligned_angular_distance

logger = logging.getLogger(__name__)


# %%
# Parameters
# ---------------
# Some example simulation configurations.
# Small sim: img_size 32, num_imgs 10000, n_classes 1000, n_nbor 10
# Medium sim: img_size 64, num_imgs 20000, n_classes 2000, n_nbor 10
# Large sim: img_size 129, num_imgs 30000, n_classes 2000, n_nbor 20

interactive = True  # Draw blocking interactive plots?
do_cov2d = False  # Use CWF coefficients
img_size = 32  # Downsample the volume to a desired resolution
num_imgs = 10000  # How many images in our source.
n_classes = 1000  # How many class averages to compute.
n_nbor = 10  # How many neighbors to stack
noise_variance = 5e-7  # Set a target noise variance


# %%
# Simulation Data
# ---------------
# Start with the hi-res volume map EMDB-2660 sourced from EMDB,
# https://www.ebi.ac.uk/emdb/EMD-2660, and dowloaded via ASPIRE's downloader utility.
og_v = emdb_2660().astype(np.float64)
logger.info("Original volume map data" f" shape: {og_v.shape} dtype:{og_v.dtype}")


# Then create a filter based on that variance
# This is an example of a custom noise profile
def noise_function(x, y):
    alpha = 1
    beta = 1
    # White
    f1 = noise_variance
    # Violet-ish
    f2 = noise_variance * (x * x + y * y) / img_size * img_size
    return (alpha * f1 + beta * f2) / 2.0


custom_noise = CustomNoiseAdder(noise_filter=FunctionFilter(noise_function))

logger.info("Initialize CTF filters.")
# Create some CTF effects
pixel_size = og_v.pixel_size  # Pixel size (in angstroms)
voltage = 200  # Voltage (in KV)
defocus_min = 1.5e4  # Minimum defocus value (in angstroms)
defocus_max = 2.5e4  # Maximum defocus value (in angstroms)
defocus_ct = 7  # Number of defocus groups.
Cs = 2.0  # Spherical aberration
alpha = 0.1  # Amplitude contrast

# Create filters
ctf_filters = [
    RadialCTFFilter(pixel_size, voltage, defocus=d, Cs=2.0, alpha=0.1)
    for d in np.linspace(defocus_min, defocus_max, defocus_ct)
]

# Finally create the Simulation
src = Simulation(
    n=num_imgs,
    vols=og_v,
    noise_adder=custom_noise,
    unique_filters=ctf_filters,
    dtype=np.float64,
)

# Downsample
src = src.downsample(img_size).cache()

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

# Cache to memory for some speedup
src = src.cache()

# %%
# Optional: CWF Denoising
# -----------------------
#
# Optionally generate an alternative source that is denoised with `cov2d`,
# then configure a customized aligner. This allows the use of CWF denoised
# images for classification, but stacks the original images for averages
# used in the remainder of the reconstruction pipeline.
#
# In this example, this behavior is controlled by the `do_cov2d` boolean variable.
# When disabled, the original src and default aligner is used.
# If you will not be using cov2d,
# you may remove this code block and associated variables.

classification_src = src
if do_cov2d:
    # Use CWF denoising
    cwf_denoiser = DenoiserCov2D(src)
    # Use denoised src for classification
    classification_src = DenoisedSource(src, cwf_denoiser)
    # Peek, what do the denoised images look like...
    if interactive:
        classification_src.images[:10].show()

# %%
# Class Averaging
# ----------------------
#
# Now perform classification and averaging for each class.
# This also demonstrates the potential to use a different source for classification and averaging.

avgs = DefaultClassAvgSource(
    classification_src,
    n_nbor=n_nbor,
    averager_src=src,
    num_procs=None,  # Automatically configure parallel processing
)
# We'll continue our pipeline with the first ``n_classes`` from ``avgs``.
avgs = avgs[:n_classes]

if interactive:
    avgs.images[:10].show()

# Save off the set of class average images.
avgs.save("simulated_abinitio_pipeline_class_averages.star", overwrite=True)

# %%
# Common Line Estimation
# ----------------------
#
# Next create a CL instance for estimating orientation of projections
# using the Common Line with Synchronization Voting method.

logger.info("Begin Orientation Estimation")

# Stash true rotations for later comparison.
# Note class selection re-ordered our images, so we remap the indices back to the original source.
indices = avgs.index_map  # Also available from avgs.src.selection_indices[:n_classes]
true_rotations = src.rotations[indices]

# Create a custom orientation estimation object for ``avgs``.
orient_est = CLSyncVoting(avgs, n_theta=180)

# Initialize an ``OrientedSource`` class instance that performs orientation
# estimation in a lazy fashion upon request of images or rotations.
oriented_src = OrientedSource(avgs, orient_est)

logger.info("Compare with known rotations")
# Compare with known true rotations. ``mean_aligned_angular_distance`` globally aligns the estimated
# rotations to the ground truth and finds the mean angular distance between them.
mean_ang_dist = mean_aligned_angular_distance(oriented_src.rotations, true_rotations)
logger.info(
    f"Mean angular distance between globally aligned estimates and ground truth rotations: {mean_ang_dist}\n"
)

# %%
# Volume Reconstruction
# ----------------------
#
# Using the estimated rotations, attempt to reconstruct a volume.

logger.info("Begin Volume reconstruction")

# Setup an estimator to perform the back projection.
estimator = MeanEstimator(oriented_src)

# Perform the estimation and save the volume.
estimated_volume = estimator.estimate()
fn = f"estimated_volume_n{num_imgs}_c{n_classes}_m{n_nbor}_{img_size}.mrc"
estimated_volume.save(fn, overwrite=True)

# Peek at result
if interactive:
    plt.imshow(np.sum(estimated_volume.asnumpy()[0], axis=-1))
    plt.show()
