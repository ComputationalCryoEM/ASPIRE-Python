"""
Commonlines Method using LUD
============================

This tutorial demonstrates using the Least Unsquared Deviations (LUD)
commonlines method for estimating particle orientations. This tutorial
reproduces the "Experiments on simulated images" found in the publication:

Orientation Determination of Cryo-EM Images Using Least Unsquared Deviations,
L. Wang, A. Singer, and  Z. Wen, SIAM J. Imaging Sciences, 6, 2450-2483 (2013).
"""

# %%
# Imports
# -------

import logging

import numpy as np

from aspire.abinitio import CommonlineLUD
from aspire.noise import WhiteNoiseAdder
from aspire.source import OrientedSource, Simulation
from aspire.utils import mean_aligned_angular_distance
from aspire.volume import Volume

logger = logging.getLogger(__name__)


# %%
# Generate Simulated Data
# -----------------------
# We generate simulated noisy images from a low res volume map of
# the 50S ribosomal subunit of E. coli.

n_imgs = 500  # Number of images in our source
snr = 1 / 8  # Signal-to-noise ratio
dtype = np.float64
res = 129

# We download the volume map and zero-pad to the indicated resolution.
vol = (
    Volume.load("../tutorials/data/clean70SRibosome_vol_65p.mrc")
    .astype(dtype)
    .downsample(res)
)
logger.info("Volume map data" f" shape: {vol.shape} dtype:{vol.dtype}")

# We generate a white noise adder with specifid SNR.
noise_adder = WhiteNoiseAdder.from_snr(snr=snr)

# Now we initialize a Simulation source to generate noisy, centered images.
src = Simulation(
    n=n_imgs,
    vols=vol,
    offsets=0,
    noise_adder=noise_adder,
    dtype=dtype,
)

# We can view the noisy images.
src.images[:5].show()

# %%
# Estimate Orientations
# ---------------------
# We use the LUD commonline algorithm to estimate the orientation of the noisy images.

logger.info("Begin Orientation Estimation")

# Create a custom orientation estimation object which uses the LUD algorithm.
# By default, we use the algortihm without spectral norm constraint.
orient_est = CommonlineLUD(src, n_theta=360)

# Initialize an ``OrientedSource`` class instance that performs orientation
# estimation in a lazy fashion upon request of images or rotations.
oriented_src = OrientedSource(src, orient_est)

# %%
# Results
# -------
# We measure our results by finding the mean angular distance between the
# ground truth rotations and the estimated rotations adjusted by the best
# global alignment.
mean_ang_dist = mean_aligned_angular_distance(oriented_src.rotations, src.rotations)
logger.info(
    f"Mean angular distance between globally aligned estimates and ground truth rotations: {mean_ang_dist}\n"
)
