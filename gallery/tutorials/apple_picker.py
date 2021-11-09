"""
Apple Picker
============

We demonstrate ASPIRE's particle picking methods using the ``Apple`` class.
"""

import logging

import matplotlib.pyplot as plt
import mrcfile

from aspire.apple.apple import Apple

logger = logging.getLogger(__name__)

# %%
# Load Micrograph Data
# --------------------
#
# Initiate ASPIRE's ``Apple`` class and load the micrograph data.

apple_picker = Apple()
filename = "data/falcon_2012_06_12-14_33_35_0.mrc"

# %%
# Pick Particles and Find Centers
# -------------------------------
#
# Here we use the ``process_micrograph`` method from the ``Apple`` class to find particles in the micrograph.
# It will also return an image suitable for display, and optionally save a jpg.

centers, particles_img = apple_picker.process_micrograph(
    filename, show_progress=False, create_jpg=True
)

# Note that if you only desire ``centers`` you may call ``process_micrograph_centers(filename,...)``.

# %%
# Read Micrograph
# ---------------
#
# Here we read in and plot the raw micrograph.

with mrcfile.open(filename, mode="r") as mrc:
    micro_img = mrc.data

plt.title("Sample Micrograph")
plt.imshow(micro_img)
plt.show()

# %%
# Plot the Picked Particles
# -----------------
#
# Observe the number of particles picked and plot the result from ``Apple``.

img_dim = micro_img.shape
particles = centers.shape[0]
logger.info(f"Dimensions of the micrograph are {img_dim}")
logger.info(f"{particles} particles were picked")

# sphinx_gallery_thumbnail_number = 2
plt.imshow(particles_img)
plt.show()
