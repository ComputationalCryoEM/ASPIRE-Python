"""
Class Averaging
===============

We demonstrate class averaging using the rotationally invariant representation algorithm.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage

from aspire.classification import RIRClass2D, TopClassSelector
from aspire.image import Image
from aspire.noise import WhiteNoiseAdder
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
round_disc = gaussian_2d(L, sigma=L / 4)
plt.imshow(round_disc, cmap="gray")
plt.show()

# %%
# Oval 2D Gaussian Image
# ^^^^^^^^^^^^^^^^^^^^^^

oval_disc = gaussian_2d(L, sigma=(L / 20, L / 5))
plt.imshow(oval_disc, cmap="gray")
plt.show()

# %%
# Handed Image
# ^^^^^^^^^^^^
#
# Create richer test set by including an asymmetric image.

# Create a second oval.
oval_disc2 = gaussian_2d(L, mu=(L / 5, L / 6), sigma=(L / 15, L / 20))

# Strategically add it to `oval_disc`.
yoval_discL = oval_disc.copy()
yoval_discL += oval_disc2
plt.imshow(yoval_discL, cmap="gray")
plt.show()

# %%
# Reflected Image
# ^^^^^^^^^^^^^^^
#
# Also include the reflection of  the asymmetric image.

yoval_discR = np.flipud(yoval_discL)
plt.imshow(yoval_discR, cmap="gray")
plt.show()

# %%
# Example Data Set Source
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# We concatenate and shuffle 512 rotations of the Gaussian images above to create our data set.

# How many entries (angles) in our stack
N = 512
thetas = np.linspace(start=0, stop=360, num=N, endpoint=False)

classRound = np.zeros((N, L, L))
classOval = np.zeros((N, L, L))
classYOvalL = np.zeros((N, L, L))
classYOvalR = np.zeros((N, L, L))

for i, theta in enumerate(thetas):
    classRound[i] = np.asarray(PILImage.fromarray(round_disc).rotate(theta))
    classOval[i] = np.asarray(PILImage.fromarray(oval_disc).rotate(theta))
    classYOvalL[i] = np.asarray(PILImage.fromarray(yoval_discL).rotate(theta))
    classYOvalR[i] = np.asarray(PILImage.fromarray(yoval_discR).rotate(theta))

# We'll make an example data set by concatentating then shuffling these.
example_array = np.concatenate((classRound, classOval, classYOvalL, classYOvalR))
np.random.seed(1234567)
np.random.shuffle(example_array)

# So now that we have cooked up an example dataset, lets create an ASPIRE source
src = ArrayImageSource(example_array)

# Let's peek at the images to make sure they're shuffled up nicely
src.images[:10].show()

# %%
# Class Average
# -------------
#
# We use the ASPIRE ``RIRClass2D`` class to classify the images via the rotationally invariant representation (RIR)
# algorithm. We then yield class averages by performing ``classify``.

rir = RIRClass2D(
    src,
    fspca_components=400,
    bispectrum_components=300,  # Compressed Features after last PCA stage.
    n_nbor=10,
    n_classes=10,
    large_pca_implementation="legacy",
    nn_implementation="legacy",
    bispectrum_implementation="legacy",
    num_procs=1,  # Change to "auto" if your machine has many processors
)

classes, reflections, dists = rir.classify()
avgs = rir.averages(classes, reflections, dists)

# %%
# Display Classes
# ^^^^^^^^^^^^^^^

avgs.images[:10].show()

# %%
# Class Averaging with Noise
# --------------------------

# %%
# Add Noise to Data Set
# ^^^^^^^^^^^^^^^^^^^^^

# Using the sample variance, we'll compute a target noise variance
# Noise
var = np.var(src.images[:].asnumpy())
noise_var = var * 2**4

# Then create noise with the ``WhiteNoiseAdder`` class.
noise = WhiteNoiseAdder(var=noise_var, seed=123)

# Add noise to the images by performing ``forward``
noisy_im = noise.forward(src.images[:])

# Recast as an ASPIRE source
noisy_src = ArrayImageSource(noisy_im)

# Let's peek at the noisey images
noisy_src.images[:10].show()

# %%
# RIR with Noise
# ^^^^^^^^^^^^^^
# This also demonstrates changing the Nearest Neighbor search to using
# scikit-learn, and using ``TopClassSelector``.``TopClassSelector``
# will deterministically select the first ``n_classes``.  This is useful
# for development and debugging.  By default ``RIRClass2D`` uses a
# ``RandomClassSelector``.

noisy_rir = RIRClass2D(
    noisy_src,
    fspca_components=400,
    bispectrum_components=300,
    n_nbor=10,
    n_classes=10,
    selector=TopClassSelector(),
    large_pca_implementation="legacy",
    nn_implementation="sklearn",
    bispectrum_implementation="legacy",
    num_procs=1,  # Change to "auto" if your machine has many processors
)

classes, reflections, dists = noisy_rir.classify()
avgs = noisy_rir.averages(classes, reflections, dists)

# %%
# Display Classes
# ^^^^^^^^^^^^^^^

avgs.images[:10].show()


# %%
# Review a class
# --------------
#
# Select a class to review.

review_class = 5

# Display the original image.
noisy_src.images[review_class].show()

# Report the identified neighbor indices
logger.info(f"Class {review_class}'s neighors: {classes[review_class]}")

# Report the identified neighbors
Image(noisy_src.images[:][classes[review_class]]).show()

# Display the averaged result
avgs.images[review_class].show()

# %%
# Alignment Details
# -----------------
#
# Alignment details are exposed when avaialable from an underlying ``averager``.
# In this case, we'll get the estimated alignments for the ``review_class``.
# These alignment arrays are indexed the same as ``classes``,
# having shape (n_classes, n_nbor).

est_rotations = noisy_rir.averager.rotations[review_class]
est_shifts = noisy_rir.averager.shifts[review_class]
est_correlations = noisy_rir.averager.correlations[review_class]

logger.info(f"Estimated Rotations: {est_rotations}")
logger.info(f"Estimated Shifts: {est_shifts}")
logger.info(f"Estimated Correlations: {est_correlations}")

# Compare the original unaligned images with the estimated alignment.
# Get the indices from the classification results.
nbr = 3
original_img_0_idx = classes[review_class][0]
original_img_nbr_idx = classes[review_class][nbr]

# Lookup the images.
original_img_0 = noisy_src.images[original_img_0_idx].asnumpy()[0]
original_img_nbr = noisy_src.images[original_img_nbr_idx].asnumpy()[0]

# Rotate using estimated rotations.
angle = est_rotations[nbr] * 180 / np.pi
if reflections[review_class][nbr]:
    original_img_nbr = np.flipud(original_img_nbr)
rotated_img_nbr = np.asarray(PILImage.fromarray(original_img_nbr).rotate(angle))

plt.subplot(2, 2, 1)
plt.title("Original Images")
plt.imshow(original_img_0)
plt.xlabel("Img 0")
plt.subplot(2, 2, 2)
plt.imshow(original_img_nbr)
plt.xlabel(f"Img {nbr}")

plt.subplot(2, 2, 3)
plt.title("Est Rotation Applied")
plt.imshow(original_img_0)
plt.xlabel("Img 0")
plt.subplot(2, 2, 4)
plt.imshow(rotated_img_nbr)
plt.xlabel(f"Img {nbr} rotated {angle:.4}*")
plt.tight_layout()
plt.show()
