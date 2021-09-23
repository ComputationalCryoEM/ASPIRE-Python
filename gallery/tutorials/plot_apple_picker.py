"""
Apple Picker
============

We demonstrate ASPIRE's particle picking methods using the ``Apple`` class.
"""

import matplotlib.pyplot as plt
import mrcfile

from aspire.apple.apple import Apple

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

centers = apple_picker.process_micrograph(filename, show_progress=False)

# %%
# Read Micrograph
# ---------------

with mrcfile.open(filename, mode="r") as mrc:
    micro_img = mrc.data

# %%
# Verify Dimensions
# -----------------
#
# Here we peek at the image dimensions and observe the number of particles picked.

micro_img.shape
apple_picker.tau2
centers.shape

# %%
# Plot with Picked Particles
# --------------------------
#
# We use the ``process_micrograph`` method to build a plot showing the picked particles.

# The next comment sets the gallery thumbnail to be the 2nd image in the script (ie. this image)
# sphinx_gallery_thumbnail_number = 2
plt.title("A simple chirp")
plt.imshow(micro_img)

# %%
# Plot with Particle Picker
# -------------------------

img = apple_picker.process_micrograph(filename, return_centers=False, return_img=True)
plt.imshow(img)
