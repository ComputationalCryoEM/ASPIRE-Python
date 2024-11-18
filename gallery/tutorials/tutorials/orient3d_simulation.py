"""
3D Image Orientation
====================

This script illustrates the estimation of orientation angles using a synchronization
matrix and the voting method, based on simulated data projected from a 3D cryo-EM map.
"""

import os

import numpy as np

from aspire.abinitio import CLSyncVoting
from aspire.operators import RadialCTFFilter
from aspire.source import OrientedSource, Simulation
from aspire.utils import mean_aligned_angular_distance
from aspire.volume import Volume

file_path = os.path.join(
    os.path.dirname(os.getcwd()), "data", "clean70SRibosome_vol_65p.mrc"
)

print(
    "This script illustrates orientation estimation using "
    "synchronization matrix and voting method"
)

# %%
# Initialize Simulation Object and CTF Filters
# --------------------------------------------

# Define a precision for this experiment
dtype = np.float32

# Set the sizes of images
img_size = 33

# Set the total number of images generated from the 3D map
num_imgs = 128

# Specify the CTF parameters not used for this example
# but necessary for initializing the simulation object
pixel_size = 5  # Pixel size of the images (in angstroms)
voltage = 200  # Voltage (in KV)
defocus_min = 1.5e4  # Minimum defocus value (in angstroms)
defocus_max = 2.5e4  # Maximum defocus value (in angstroms)
defocus_ct = 7  # Number of defocus groups.
Cs = 2.0  # Spherical aberration
alpha = 0.1  # Amplitude contrast

print("Initialize simulation object and CTF filters.")
# Create CTF filters
filters = [
    RadialCTFFilter(pixel_size, voltage, defocus=d, Cs=2.0, alpha=0.1)
    for d in np.linspace(defocus_min, defocus_max, defocus_ct)
]

# %%
# Downsampling
# ------------

# Load the map file of a 70S Ribosome and downsample the 3D map to desired resolution.
# The downsampling can be done by the internal function of Volume object.
print(
    f"Load 3D map and downsample 3D map to desired grids "
    f"of {img_size} x {img_size} x {img_size}."
)
vols = Volume.load(file_path, dtype=dtype)
vols = vols.downsample(img_size)

# %%
# Create Simulation Object and Obtain True Rotation Angles
# --------------------------------------------------------

# Create a simulation object with specified filters and the downsampled 3D map
print("Use downsampled map to creat simulation object.")
sim = Simulation(L=img_size, n=num_imgs, vols=vols, unique_filters=filters, dtype=dtype)

print("Get true rotation angles generated randomly by the simulation object.")
rots_true = sim.rotations

# %%
# Estimate Orientation
# --------------------

# Initialize an orientation estimation object and create an
# ``OrientedSource`` object to perform viewing angle and image offset
# estimation. Here, because of the small image size of the
# ``Simulation``, we customize the ``CLSyncVoting`` method to use
# fewer theta values when searching for common-lines between pairs of
# images. Additionally, since we are processing images with no noise,
# we opt not to use a ``fuzzy_mask``, an option that improves
# common-line detection in higher noise regimes.
print(
    "Estimate rotation angles and offsets using synchronization matrix and voting method."
)
orient_est = CLSyncVoting(sim, n_theta=36, mask=False)
oriented_src = OrientedSource(sim, orient_est)
rots_est = oriented_src.rotations


# %%
# Mean Angular Distance
# ---------------------

# ``mean_aligned_angular_distance`` will perform global alignment of the estimated rotations
# to the ground truth and find the mean angular distance between them (in degrees).
mean_ang_dist = mean_aligned_angular_distance(rots_est, rots_true)
print(
    f"Mean angular distance between estimates and ground truth: {mean_ang_dist} degrees"
)

# Basic Check
assert mean_ang_dist < 10

# %%
# Offsets Estimation
# ------------------

# The ground truth offsets from the simulation can be compared to
# those estimated by the commonlines method above.

# Calculate Estimation error in pixels for each image.
offs_diff = np.sqrt(np.sum((oriented_src.offsets - sim.offsets) ** 2, axis=1))

# Calculate the mean error in pixels across all images.
offs_err = offs_diff.mean()
print(f"Mean offset error in pixels {offs_err}, approx {offs_err/img_size*100:.1f}%")
