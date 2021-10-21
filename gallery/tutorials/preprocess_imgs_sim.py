"""
Image Preprocessing
===================

This script illustrates the preprocess steps implemented prior to starting the pipeline of
reconstructing a 3D map using simulated 2D images.
"""
import logging
import os

import matplotlib.pyplot as plt
import mrcfile
import numpy as np

from aspire.noise import WhiteNoiseEstimator
from aspire.operators import RadialCTFFilter, ScalarFilter
from aspire.source.simulation import Simulation
from aspire.volume import Volume

logger = logging.getLogger(__name__)

DATA_DIR = "data"


logger.info(
    "This script illustrates orientation estimation using "
    "synchronization matrix and voting method"
)

# %%
# Specify Parameters
# ------------------

# Set the downsample size of images
img_size = 33

# Set the total number of images generated from the 3D map
num_imgs = 512

# Set the noise variance and build the noise filter
noise_variance = 4e-1
noise_filter = ScalarFilter(dim=2, value=noise_variance)

# Specify the CTF parameters not used for this example
# but necessary for initializing the simulation object
pixel_size = 5 * 65 / img_size  # Pixel size of the images (in angstroms)
voltage = 200  # Voltage (in KV)
defocus_min = 1.5e4  # Minimum defocus value (in angstroms)
defocus_max = 2.5e4  # Maximum defocus value (in angstroms)
defocus_ct = 7  # Number of defocus groups
Cs = 2.0  # Spherical aberration
alpha = 0.1  # Amplitude contrast

# %%
# Build Simulation Object and Apply Noise
# ---------------------------------------

logger.info("Initialize simulation object and CTF filters.")
# Create CTF filters
ctf_filters = [
    RadialCTFFilter(pixel_size, voltage, defocus=d, Cs=2.0, alpha=0.1)
    for d in np.linspace(defocus_min, defocus_max, defocus_ct)
]

# Load the map file of a 70S ribosome and downsample the 3D map to desired resolution.
infile = mrcfile.open(os.path.join(DATA_DIR, "clean70SRibosome_vol_65p.mrc"))
logger.info(f"Load 3D map from mrc file, {infile}")
vols = Volume(infile.data)

# Downsample the volume to a desired resolution and increase density
# by 1.0e5 time for a better graph view
logger.info(f"Downsample map to a resolution of {img_size} x {img_size} x {img_size}")
vols = vols.downsample((img_size,) * 3) * 1.0e5

# Create a simulation object with specified filters and the downsampled 3D map
logger.info("Use downsampled map to create simulation object.")
source = Simulation(
    L=img_size,
    n=num_imgs,
    vols=vols,
    unique_filters=ctf_filters,
    noise_filter=noise_filter,
)

# %%
# Apply Preprocessing Techniques
# ------------------------------

logger.info("Obtain original images.")
imgs_od = source.images(start=0, num=1).asnumpy()

logger.info("Perform phase flip to input images.")
source.phase_flip()
imgs_pf = source.images(start=0, num=1).asnumpy()

max_resolution = 15
logger.info(f"Downsample resolution to {max_resolution} X {max_resolution}")
if max_resolution < source.L:
    source.downsample(max_resolution)
imgs_ds = source.images(start=0, num=1).asnumpy()

logger.info("Normalize images to background noise.")
source.normalize_background()
imgs_nb = source.images(start=0, num=1).asnumpy()

logger.info("Whiten noise of images")
noise_estimator = WhiteNoiseEstimator(source)
source.whiten(noise_estimator.filter)
imgs_wt = source.images(start=0, num=1).asnumpy()

logger.info("Invert the global density contrast if need")
source.invert_contrast()
imgs_rc = source.images(start=0, num=1).asnumpy()


# %%
# Plot First Image from Each Preprocess Step
# ------------------------------------------

# plot the first images
logger.info("Plot first image from each preprocess steps")
idm = 0
plt.subplot(2, 3, 1)
plt.imshow(imgs_od[idm], cmap="gray")
plt.colorbar(orientation="horizontal")
plt.title("original image")

plt.subplot(2, 3, 2)
plt.imshow(imgs_pf[idm], cmap="gray")
plt.colorbar(orientation="horizontal")
plt.title("phase flip")

plt.subplot(2, 3, 3)
plt.imshow(imgs_ds[idm], cmap="gray")
plt.colorbar(orientation="horizontal")
plt.title("downsample")

plt.subplot(2, 3, 4)
plt.imshow(imgs_nb[idm], cmap="gray")
plt.colorbar(orientation="horizontal")
plt.title("normalize background")

plt.subplot(2, 3, 5)
plt.imshow(imgs_wt[idm], cmap="gray")
plt.colorbar(orientation="horizontal")
plt.title("noise whitening")

plt.subplot(2, 3, 6)
plt.imshow(imgs_rc[idm], cmap="gray")
plt.colorbar(orientation="horizontal")
plt.title("invert contrast")
plt.tight_layout()
