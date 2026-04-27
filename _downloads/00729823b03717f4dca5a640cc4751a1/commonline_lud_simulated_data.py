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
from fractions import Fraction
from itertools import product

import numpy as np

from aspire.abinitio import CommonlineIRLS, CommonlineLUD
from aspire.noise import WhiteNoiseAdder
from aspire.source import Simulation
from aspire.utils import mean_aligned_angular_distance
from aspire.volume import Volume

logger = logging.getLogger(__name__)


# %%
# Parameters
# ----------
# Set up some initializing parameters. We will run the LUD algorithm using ADMM
# and IRLS methods under various spectral norm constraints and levels of noise.

SNR = ["1/8", "1/16", "1/32"]  # Signal-to-noise ratio
METHOD = ["ADMM", "IRLS"]
ALPHA = [0.90, 0.75, 0.67]  # Spectral norm constraint
n_imgs = 500  # Number of images in our source
dtype = np.float64
pad_size = 129

# %%
# Load Volume Map
# ---------------
# We will generate simulated noisy images from a low res volume
# map available in our data folder. This volume map is a 65 x 65 x 65
# voxel volume which we intend to upsample to 129 x 129 x 129.
# To do this we use our ``downsample`` method which, when provided a voxel
# size larger than the input volume, internally zero-pads in Fourier
# space to increase the overall shape of the volume.
vol = (
    Volume.load("../tutorials/data/clean70SRibosome_vol_65p.mrc")
    .astype(dtype)
    .downsample(pad_size)
)
logger.info("Volume map data" f" shape: {vol.shape} dtype:{vol.dtype}")

# %%
# Generate Noisy Images and Estimate Rotations
# --------------------------------------------
# A ``Simulation`` object is used to generate simulated data at various
# noise levels. Then rotations are estimated using the ``CommonlineLUD`` and
# ``CommonlineIRLS`` algorithms. Results are measured by computing the mean
# aligned angular distance between the ground truth rotations and the globally
# aligned estimated rotations.

# Build table to dislay results.
col_width = 21
table = []
table.append(
    f"{'METHOD':<{col_width}} {'SNR':<{col_width}} {'ALPHA':<{col_width}} {'Mean Angular Distance':<{col_width}}"
)
table.append("-" * (col_width * 4))

for method, snr, alpha in product(METHOD, SNR, ALPHA):
    # Generate a white noise adder with specified SNR.
    noise_adder = WhiteNoiseAdder.from_snr(snr=Fraction(snr))

    # Initialize a Simulation source to generate noisy, centered images.
    src = Simulation(
        n=n_imgs,
        vols=vol,
        offsets=0,
        amplitudes=1,
        noise_adder=noise_adder,
        dtype=dtype,
    ).cache()

    # Estimate rotations using the LUD algorithm.
    if method == "ADMM":
        orient_est = CommonlineLUD(src, alpha=alpha)
    else:
        orient_est = CommonlineIRLS(src, alpha=alpha)
    est_rotations = orient_est.estimate_rotations()

    # Find the mean aligned angular distance between estimates and ground truth rotations.
    mean_ang_dist = mean_aligned_angular_distance(est_rotations, src.rotations)

    # Append results to table.
    table.append(
        f"{method:<{col_width}} {snr:<{col_width}} {str(alpha):<{col_width}} {mean_ang_dist:<{col_width}}"
    )

# %%
# Display Results
# ---------------
# Display table of results for both methods using various spectral norm
# constraints and noise levels.

logger.info("\n" + "\n".join(table))
