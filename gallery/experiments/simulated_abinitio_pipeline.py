"""
ASPIRE-Python Abinitio Pipeline
================================

In this notebook we will introduce a selection of
components corresponding to a pipeline.
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
from aspire.basis import FFBBasis3D
from aspire.classification import RIRClass2D
from aspire.denoising import DenoiserCov2D
from aspire.noise import AnisotropicNoiseEstimator
from aspire.operators import FunctionFilter, RadialCTFFilter
from aspire.reconstruction import MeanEstimator
from aspire.source import ArrayImageSource, Simulation
from aspire.utils.coor_trans import (
    get_aligned_rotations,
    get_rots_mse,
    register_rotations,
)
from aspire.volume import Volume

logger = logging.getLogger(__name__)

# Do we want to draw blocking interactive plots?
interactive = False

# %%
# Simulation Data
# ---------------
# We'll start with a fairly hi-res volume available from EMPIAR/EMDB.
# https://www.ebi.ac.uk/emdb/EMD-2660
# https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-2660/map/emd_2660.map.gz
og_v = Volume.load("emd_2660.map", dtype=np.float64)
logger.info("Original volume map data" f" shape: {og_v.shape} dtype:{og_v.dtype}")

# Downsample the volume to a desired resolution
img_size = 64

logger.info(f"Downsampling to {(img_size,)*3}")
v = og_v.downsample(img_size)
L = v.resolution

num_imgs = 20000  # How many images in our source.
n_classes = 2000  # How many class averages to compute.
n_nbor = 10  # How many neighbors to stack

# Set a target noise variance
noise_variance = 1e-4


# Then create a filter based on that variance
# This is an example of a custom noise profile
def noise_function(x, y):
    alpha = 1
    beta = 1
    # White
    f1 = noise_variance
    # Violet-ish
    f2 = noise_variance * (x * x + y * y) / L * L
    return (alpha * f1 + beta * f2) / 2.0


custom_noise_filter = FunctionFilter(noise_function)

logger.info("Initialize CTF filters.")
# Create some CTF effects
pixel_size = 5 * 65 / img_size  # Pixel size of the images (in angstroms)
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
    L=v.resolution,
    n=num_imgs,
    vols=v,
    noise_filter=custom_noise_filter,
    unique_filters=ctf_filters,
)
# Peek
if interactive:
    src.images(0, 10).show()

# # TODO: Seemed to cause a crash, maybe dtype/blkdiag related
# logger.info("Normalize images to background noise.")
# src.normalize_background()
# # Peek
# if interactive: src.images(0, 10).show()

# Currently we use phase_flip to attempt correcting for CTF.
logger.info("Perform phase flip to input images.")
src.phase_flip()

# We should estimate the noise and `Whiten` based on the estimated noise
aiso_noise_estimator = AnisotropicNoiseEstimator(src)
src.whiten(aiso_noise_estimator.filter)

# Plot the noise profile for inspection
if interactive:
    plt.imshow(aiso_noise_estimator.filter.evaluate_grid(L))
    plt.show()

# Peek, what do the whitened images look like...
if interactive:
    src.images(0, 10).show()

# logger.info("Invert the global density contrast")
# src.invert_contrast()

# Use CWF denoising
cwf_denoiser = DenoiserCov2D(src)
src = cwf_denoiser.denoise()

# Peek, what do the denoised images look like...
if interactive:
    src.images(0, 10).show()


# Cache to memory for some speedup
src = ArrayImageSource(src.images(0, num_imgs).asnumpy(), angles=src.angles)

# %%
# Class Averaging
# ----------------------
#
# Now we perform classification and averaging for each class.

logger.info("Begin Class Averaging")

rir = RIRClass2D(
    src,
    fspca_components=400,
    bispectrum_components=300,  # Compressed Features after last PCA stage.
    n_nbor=n_nbor,
    n_classes=n_classes,
    large_pca_implementation="legacy",
    nn_implementation="sklearn",
    bispectrum_implementation="legacy",
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

# Stash true rotations for later comparison,
#   note this line only works with naive class selection...
true_rotations = src.rots[:n_classes]

orient_est = CLSyncVoting(avgs, n_theta=36)
# Get the estimated rotations
orient_est.estimate_rotations()
rots_est = orient_est.rotations

logger.info("Compare with known rotations")
# Compare with known true rotations
Q_mat, flag = register_rotations(rots_est, true_rotations)
regrot = get_aligned_rotations(rots_est, Q_mat, flag)
mse_reg = get_rots_mse(regrot, true_rotations)
logger.info(
    f"MSE deviation of the estimated rotations using register_rotations : {mse_reg}\n"
)

# %%
# Volume Reconstruction
# ----------------------
#
# Using the estimated rotations, attempt to reconstruct a volume.

logger.info("Begin Volume reconstruction")

# Assign the estimated rotations to the class averages
avgs.rots = rots_est

# Create a reasonable Basis for the 3d Volume
basis = FFBBasis3D((v.resolution,) * 3, dtype=v.dtype)

# Setup an estimator to perform the back projection.
estimator = MeanEstimator(avgs, basis)

# Perform the estimation and save the volume.
estimated_volume = estimator.estimate()
fn = f"estimated_volume_n{num_imgs}_c{n_classes}_m{n_nbor}_{img_size}.mrc"
estimated_volume.save(fn, overwrite=True)

# Peek at result
if interactive:
    plt.imshow(np.sum(estimated_volume[0], axis=-1))
    plt.show()
