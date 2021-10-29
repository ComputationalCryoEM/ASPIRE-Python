"""
Generating 3D Volume Projections
================================

This script illustrates using ASPIRE's Simulation source to
generate projections of a Volume using prescribed rotations.

"""

import logging
import os

import mrcfile
import numpy as np

from aspire.operators import ScalarFilter
from aspire.source.simulation import Simulation
from aspire.utils import Rotation
from aspire.volume import Volume

logger = logging.getLogger(__name__)

# %%
# Configure how many images we'd like to project
# ----------------------------------------------
n_img = 10

# %%
# Load our Volume data
# --------------------
# This example starts with an mrc, loading it as an numpy array

DATA_DIR = "data"  # Tutorial example data folder
v_npy = mrcfile.open(
    os.path.join(DATA_DIR, "clean70SRibosome_vol_65p.mrc")
).data.astype(np.float64)

# Then using that to instantiate a Volume, which is downsampled to 60x60x60
v = Volume(v_npy).downsample(60)

# %%
# Defining rotations
# ------------------
# This will force a collection of in plane rotations about z.

# First get a list of angles to test
thetas = np.linspace(0, 2 * np.pi, num=n_img, endpoint=False)

# Define helper function for common 3D rotation matrix, about z.


def r_z(theta):
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


# Construct a sequence of rotation matrices using r_z(thetas)
_rots = np.empty((n_img, 3, 3))
for n, theta in enumerate(thetas):
    # Note we negate theta to match Rotation convention.
    _rots[n] = r_z(-theta)

# Instantiate ASPIRE's Rotation class with the rotation matrices.
# This will allow us to use or access the rotations in a variety of ways.
rots = Rotation.from_matrix(_rots)

# %%
# Configure Noise
# ---------------
# We can define controlled noise and have the Simulation apply it to our projection images.

noise_variance = 1e-10  # Normally this would be derived from a desired SNR.

# Then create a constant filter based on that variance, which is passed to Simulation
white_noise_filter = ScalarFilter(dim=2, value=noise_variance)


# %%
# Setup Simulation Source
# -----------------------

# Simulation will randomly shift and amplify images by default.
# Instead we define the following parameters.
shifts = np.zeros((n_img, 2))
amplitudes = np.ones(n_img)

# Create a Simulation Source object
src = Simulation(
    vols=v,  # our Volume
    L=v.resolution,  # resolution, should match Volume
    n=n_img,  # number of projection images
    C=len(v),  # Number of volumes in vols. 1 in this case
    angles=rots.angles,  # pass our rotations as Euler angles
    offsets=shifts,  # translations (wrt to origin)
    amplitudes=amplitudes,  # amplification ( 1 is identity)
    seed=12345,  # RNG seed for reproducibility
    dtype=v.dtype,  # match our datatype to the Volume.
    noise_filter=white_noise_filter,  # optionally prescribe noise
)

# %%
# Yield projection images from the Simulation Source
# --------------------------------------------------

# Consume images from the source by providing
# a starting index and number of images.
# Here we generate the first 3 and peek at them.
src.images(0, 3).show()
src.projections(0, 3).show()

# Here we return the first n_img images as a numpy array.
dirty_ary = src.images(0, n_img).asnumpy()

# And we have access to the clean images
clean_ary = src.projections(0, n_img).asnumpy()

# Similary, the angles/rotations/shifts/amplitudes etc.
