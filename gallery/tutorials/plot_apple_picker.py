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

apple_picker = Apple()
filename = "../../tutorials/data/falcon_2012_06_12-14_33_35_0.mrc"

# %%
# Find Image Centers
# ------------------

centers = apple_picker.process_micrograph(filename, show_progress=False)

# %%
# Read Micrograph
# ---------------

with mrcfile.open(filename, mode="r") as mrc:
    micro_img = mrc.data

# %%
# Output Dimensions
# -----------------

micro_img.shape
apple_picker.tau2
centers.shape

# %%
# Plot Micrograph Image
# ---------------------

plt.title("A simple chirp")
plt.imshow(micro_img)

# %%
# Plot with Particle Picker
# -------------------------

img = apple_picker.process_micrograph(filename, return_centers=False, return_img=True)
plt.imshow(img)
