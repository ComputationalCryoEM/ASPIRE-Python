"""
ASPIRE Image Class
==================

This tutorial demonstrates some methods of the ASPIRE ``Image`` class
"""

import os

import numpy as np

from aspire.image import Image
from aspire.operators import CTFFilter

DATA_DIR = "data"
img_data = np.load(os.path.join(DATA_DIR, "monuments.npy"))
img_data.shape, img_data.dtype

# %%
# Create an Image Instance
# ------------------------

# Create an ASPIRE Image instance from the data
#   We'll tell it to convert to floating point data as well.
im = Image(img_data, dtype=np.float64)

# %%
# Plot the Image Stack
# --------------------

# Plot the Image stack
im.show()

# %%
# Apply a Uniform Shift
# ---------------------

# Apply a single shift to each image.
shifts = np.array([100, 30])
im.shift(shifts).show()

# %%
# Apply Image-wise Shifts
# -----------------------

# Or apply shifts corresponding to to each image.
shifts = np.array([[300 * i, 100 * i] for i in range(1, im.n_images + 1)])
im.shift(shifts).show()

# %%
# Downsampling
# ------------

im.downsample(80).show()

# %%
# CTF Filter
# ----------

# pixel_size/defous_u/defocus_v in Angstrom, voltage in kV
filter = CTFFilter(pixel_size=1, voltage=100, defocus_u=1500, defocus_v=2000)
im.filter(filter).show()
