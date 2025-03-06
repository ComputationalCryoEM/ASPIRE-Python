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
from aspire.source import Simulation
from aspire.utils import mean_aligned_angular_distance
from aspire.volume import Volume

logger = logging.getLogger(__name__)


# %%
# Parameters
# ----------
# Set up some initializing parameters. We will run the LUD algorithm
# for various levels of noise and output a table of results.

SNR = [1 / 8, 1 / 16, 1 / 32]  # Signal-to-noise ratio
n_imgs = 50  # Number of images in our source
dtype = np.float64
pad_size = 129
results = {
    "SNR": ["1/8", "1/16", "1/32"],
    "Mean Angular Distance": [],
}  # Dictionary to store results

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
# noise levels. Then rotations are estimated using ``CommonlineLUD` algorithm.
# Results are measured by computing the mean aligned angular distance between
# the ground truth rotations and the globally aligned estimated rotations.
for snr in SNR:
    # Generate a white noise adder with specifid SNR.
    noise_adder = WhiteNoiseAdder.from_snr(snr=snr)

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
    orient_est = CommonlineLUD(src)
    est_rotations = orient_est.estimate_rotations()

    # Find the mean aligned angular distance between estimates and ground truth rotations.
    mean_ang_dist = mean_aligned_angular_distance(est_rotations, src.rotations)

    # Store results.
    results["Mean Angular Distance"].append(mean_ang_dist)

# %%
# Display Results
# ---------------
# Display table of results for various noise levels.

# Column widths
col1_width = 10
col2_width = 22

# Create table as a string
table = []
table.append(f"{'SNR':<{col1_width}} {'Mean Angular Distance':<{col2_width}}")
table.append("-" * (col1_width + col2_width))

for snr, angle in zip(results["SNR"], results["Mean Angular Distance"]):
    table.append(f"{snr:<{col1_width}} {angle:<{col2_width}}")

# Log the table
logger.info("\n" + "\n".join(table))
