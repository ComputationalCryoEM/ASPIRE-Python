"""
Class Averaging
===============

We demonstrate class averaging
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage

from aspire.classification import RIRClass2D
from aspire.image.xform import NoiseAdder
from aspire.operators import ScalarFilter
from aspire.source import ArrayImageSource  # Helpful hint if you want to BYO array.
from aspire.utils import gaussian_2d

logger = logging.getLogger(__name__)

# %%
# Build Simulated Data
# --------------------

# %%
# Circular 2D Gaussian Image
# ^^^^^^^^^^^^^^^^^^^^^^^^^^

L = 100
round_disc = gaussian_2d(L, sigma_x=L / 4, sigma_y=L / 4)
plt.imshow(round_disc)

# %%
# Oval 2D Gaussian Image
# ^^^^^^^^^^^^^^^^^^^^^^

oval_disc = gaussian_2d(L, sigma_x=L / 20, sigma_y=L / 5)
plt.imshow(oval_disc)

# %%
# Example Data Set
# ^^^^^^^^^^^^^^^^
#
# We concatenate and shuffle 512 rotations of the Gaussian images above to create our data set.

# How many entries (angles) in our stack
N = 512
thetas = np.linspace(start=0, stop=360, num=N, endpoint=False)

classRound = np.zeros((N, L, L))
classOval = np.zeros((N, L, L))

for i, theta in enumerate(thetas):
    classRound[i] = np.asarray(PILImage.fromarray(round_disc).rotate(theta))
    classOval[i] = np.asarray(PILImage.fromarray(oval_disc).rotate(theta))

# We'll make an example data set by concatentating then shuffling these.
example_array = np.concatenate((classRound, classOval))
np.random.shuffle(example_array)

# So now that we have cooked up an example dataset, lets create an ASPIRE source
src = ArrayImageSource(example_array)

# Let's peek at the images to make sure they're shuffled up nicely
src.images(0, 10).show()
src.n

# %%
# Class Average
# -------------
#
# We use the ASPIRE ``RIRClass2D`` class to classify the images via the Rotationally Invariant Representation (RIR)
# algorithm. We then yield class averages by performing ``classify``.

rir = RIRClass2D(
    src,
    fspca_components=400,
    bispectrum_components=300,  # Compressed Features after last PCA stage.
    n_nbor=5,
    n_classes=10,
    large_pca_implementation="legacy",
    nn_implementation="sklearn",  # I have sk version output hist of "distance" distribution for consideration.
    bispectrum_implementation="legacy",
)  # replaced PCA and NN codes with third party (slightly faster and more robust)

classes, reflections, rotations, corr = rir.classify()

# %%
# Display Classes
# ---------------

avgs = rir.output(classes, reflections, rotations)
avgs.images(0, 10).show()

# %%
# Class Averaging with Noise
# --------------------------

# %%
# Add Noise to Data Set
# ---------------------

# Using the sample variance, we'll compute a target noise variance
var = np.var(src.images(0, src.n).asnumpy())
noise_var = 0.5 * var

# We create a uniform noise to apply to the 2D images
noise_filter = ScalarFilter(dim=2, value=noise_var)

# Then create a NoiseAdder.
noise = NoiseAdder(seed=123, noise_filter=noise_filter)

# Add the noise to the images
src_with_noise = noise.forward(src.images(0, src.n))

# Recast as an ASPIRE source
noisey_src = ArrayImageSource(src_with_noise)

# Let's peek at the noisey images
noisey_src.images(0, 10).show()
noisey_src.n

# %%
# RIR with Noise
# --------------

rir_n = RIRClass2D(
    noisey_src,
    fspca_components=400,
    bispectrum_components=300,  # Compressed Features after last PCA stage.
    n_nbor=5,
    n_classes=10,
    large_pca_implementation="legacy",
    nn_implementation="sklearn",  # I have sk version output hist of "distance" distribution for consideration.
    bispectrum_implementation="legacy",
)  # replaced PCA and NN codes with third party (slightly faster and more robust)

classes, reflections, rotations, corr = rir_n.classify()

# %%
# Display Classes
# ---------------

avgs = rir_n.output(classes, reflections, rotations)
avgs.images(0, 10).show()
