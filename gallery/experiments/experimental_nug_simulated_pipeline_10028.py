"""
C1 NUG Rotation Accuracy on Simulated EMDB-2660 Images
======================================================

Estimate orientations of noisy, simulated projections of EMDB-2660 using
the asymmetric (C1) NUG method. Compare the estimated rotations with the
known simulation rotations at several signal-to-noise ratios (SNRs). A
volume is reconstructed and saved off for comparison against the generating
volume.

This experiment is based on:

.. admonition:: Reference

   A. S. Bandeira, Y. Chen, R. R. Lederman, and A. Singer,
   "Non-unique games over compact groups and orientation estimation
   in cryo-EM," Inverse Problems 36(6), 064002 (2020).
   https://doi.org/10.1088/1361-6420/ab7d2c

The ASPIRE-Python implementation of NUG differs from the original MATLAB
implementation, so this example is intended to reproduce the experimental
workflow rather than its exact numerical results.
"""

# %%
# Imports
# -------
import logging

import numpy as np

from aspire.abinitio import CommonlineNUG
from aspire.downloader import emdb_2660
from aspire.noise import WhiteNoiseAdder
from aspire.reconstruction import MeanEstimator
from aspire.source import OrientedSource, Simulation
from aspire.utils import Rotation

logger = logging.getLogger(__name__)


# %%
# Experiment settings
# -------------------
# Start with 100 images. The paper also reports results for 500 images and
# SNR ranging from 1 to 1/128, which requires substantially more computation.
SEED = 1980
RESOLUTION = 129
N_IMAGES = (100,)
SNR_VALUES = (1, 1 / 2, 1 / 4, 1 / 8)

# Download and downsample once; every SNR uses the same underlying volume.
volume = emdb_2660().astype(np.float64).downsample(RESOLUTION)


# %%
# Estimate rotations at each SNR
# ------------------------------
results = []

for n_images in N_IMAGES:
    for snr in SNR_VALUES:
        logger.info(f"Estimating C1 rotations: n={n_images}, SNR={snr}")

        # Keeping the simulation seed fixed gives corresponding ground-truth
        # orientations across SNR conditions. The noise seed is also fixed,
        # so the noise realization is scaled as the requested SNR changes.
        source = Simulation(
            n=n_images,
            vols=volume,
            offsets=0,
            amplitudes=1,
            seed=SEED,
            noise_adder=WhiteNoiseAdder.from_snr(snr=snr, seed=SEED),
        ).cache()

        # The ADMM update order uses NumPy randomness. Reset it for a
        # reproducible run at each SNR.
        np.random.seed(SEED)

        estimator = CommonlineNUG(
            source,
            symmetry="C1",
            max_shift=0,
            max_iter=501,
            verbose=False,
        )

        oriented_source = OrientedSource(source, estimator)
        estimated_rotations = oriented_source.rotations

        # Rotation.mse registers the estimates to ground truth before
        # computing the mean squared error.
        mse = Rotation(estimated_rotations).mse(Rotation(source.rotations))
        results.append((n_images, snr, mse))

        print(f"n={n_images}  SNR={snr}  rotation MSE={mse}", flush=True)

        # The simulation has zero offsets. Use the estimated orientations
        # with zero shifts to reconstruct from the cached noisy images.
        oriented_source = oriented_source.update(offsets=0)

        snr_tag = f"{snr:g}".replace(".", "p")
        volume_filename = f"nug_2660_c1_n{n_images}_snr{snr_tag}_recon.mrc"

        logger.info(f"Reconstructing and saving {volume_filename}")
        estimated_volume = MeanEstimator(oriented_source).estimate()
        estimated_volume.save(volume_filename, overwrite=True)
# %%
# Summary
# -------
print("\nC1 NUG rotation accuracy")
print(f"{'Images':>6} {'SNR':>10} {'MSE':>14}")
for n_images, snr, mse in results:
    print(f"{n_images:6d} {snr:10g} {mse:14.6g}")
