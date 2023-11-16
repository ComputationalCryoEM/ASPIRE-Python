"""
Weighted Volume Reconstruction
==============================

This tutorial demonstrates using weighted volume reconstruction,
using a reference dataset.
"""

# %%
# Download an Example Dataset
# ---------------------------
from aspire import downloader
from aspire.source import ArrayImageSource

sim_data = downloader.simulated_channelspin()

# This data contains a Volume stack, an Image stack, weights and
# corresponding parameters that were used to derive the image stack
# from the volumes.  For example. the rotations below are the known
# true simulation projections. In practice these would be derived from
# an orientation estimation component.

imgs = sim_data["images"]  # Simulated image stack.
rots = sim_data["rots"]  # True projection rotations
weights = sim_data["weights"]  # Volume weights
vols = sim_data["vols"]  # True reference volumes

# %%
# Create a ImageSource
# ----------------
# lorem ipsum
src = ArrayImageSource(imgs, angles=rots.angles)


# %%
# Volume Reconstruction
# ---------------------
# Now that we have our class averages and rotation estimates, we can
# estimate the mean volume by supplying the class averages and basis
# for back projection.

from aspire.basis import FFBBasis3D
from aspire.reconstruction import WeightedVolumesEstimator

# Create a reasonable Basis
basis = FFBBasis3D(src.L, dtype=src.dtype)

# Setup an estimator to perform the back projection.
breakpoint()
estimator = WeightedVolumesEstimator(weights, src, basis, preconditioner="none")

# XXX
import os

import numpy as np

from aspire.volume import Volume

fn = "est_vol.npy"
if not os.path.exists(fn):
    # Perform the estimation.
    estimated_volume = estimator.estimate()
    np.save(fn, estimated_volume.asnumpy())

estimated_volume = Volume(np.load(fn))


# .. note:
#     The ``estimate`` requires a fair amount of compute time,
#     but there should be regularly logged progress towards convergence.

# %%
# Comparison of Estimated Volume with Source Volume
# -------------------------------------------------
# Generate and compare several random projections between the estimated volumes and the known volumes.

from aspire.utils import Rotation, uniform_random_angles

v = 0  # Volume under comparison
m = 3  # Number of projections

random_rotations = Rotation.from_euler(uniform_random_angles(m, dtype=src.dtype))

# Estimated volume projections
estimated_volume[v].project(random_rotations).show()

# Source volume projections
vols[v].project(random_rotations).show()
