"""
Apple Picker
============

We demonstrate ASPIRE's particle picking methods using the ``Apple`` class.
"""

import os

import matplotlib.pyplot as plt
import mrcfile

from aspire.apple.apple import Apple

# %%
# Read and Plot Micrograph
# ------------------------
#
# Here we demonstrate reading in and plotting a raw micrograph.

file_path = os.path.join(
    os.path.dirname(os.getcwd()), "data", "falcon_2012_06_12-14_33_35_0.mrc"
)

with mrcfile.open(file_path, mode="r") as mrc:
    micro_img = mrc.data

plt.title("Sample Micrograph")
plt.imshow(micro_img, cmap="gray")
plt.show()

# %%
# Initialize Apple
# ----------------
#
# Initiate ASPIRE's ``Apple`` class.
# ``Apple`` admits many options relating to particle sizing and mrc processing.

apple_picker = Apple(
    particle_size=78, min_particle_size=19, max_particle_size=156, tau1=710, tau2=7100
)


# %%
# Pick Particles and Find Centers
# -------------------------------
#
# Here we use the ``process_micrograph`` method from the ``Apple`` class to find particles in the micrograph.
# It will also return an image suitable for display, and optionally save a jpg.

centers, particles_img = apple_picker.process_micrograph(file_path, create_jpg=True)

# Note that if you only desire ``centers`` you may call ``process_micrograph_centers(file_path,...)``.

# %%
# Plot the Picked Particles
# -------------------------
#
# Observe the number of particles picked and plot the result from ``Apple``.

img_dim = micro_img.shape
particles = centers.shape[0]
print(f"Dimensions of the micrograph are {img_dim}")
print(f"{particles} particles were picked")

# sphinx_gallery_thumbnail_number = 2
plt.imshow(particles_img, cmap="gray")
plt.show()
