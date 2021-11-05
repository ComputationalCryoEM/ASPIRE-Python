"""
Class Averaging
===============

We demonstrate class averaging using the rotationally invariant representation algorithm.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage

from aspire.classification import RIRClass2D
from aspire.image import Image
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
plt.show()

# %%
# Oval 2D Gaussian Image
# ^^^^^^^^^^^^^^^^^^^^^^

oval_disc = gaussian_2d(L, sigma_x=L / 20, sigma_y=L / 5)
plt.imshow(oval_disc)
plt.show()

# %%
# Handed Image
# ^^^^^^^^^^^^
#
# Create richer test set by including an asymmetric image.

# Create a second oval.
oval_disc2 = gaussian_2d(L, L / 5, L / 6, sigma_x=L / 15, sigma_y=L / 20)

# Strategically add it to `oval_disc`.
yoval_discL = oval_disc.copy()
yoval_discL += oval_disc2
plt.imshow(yoval_discL)
plt.show()

# %%
# Reflected Image
# ^^^^^^^^^^^^^^^
#
# Also include the reflection of  the asymmetric image.

yoval_discR = np.flipud(yoval_discL)
plt.imshow(yoval_discR)
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
src.images(0, 10).show()

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
)

classes, reflections, rotations, shifts, corr = rir.classify()

# %%
# Display Classes
# ^^^^^^^^^^^^^^^

avgs = rir.output(classes, reflections, rotations)
avgs.images(0, 10).show()

# %%
# Class Averaging with Noise
# --------------------------

# %%
# Add Noise to Data Set
# ^^^^^^^^^^^^^^^^^^^^^

# Using the sample variance, we'll compute a target noise variance
# Noise
var = np.var(src.images(0, src.n).asnumpy())
noise_var = var * 2 ** 4

# We create a uniform noise to apply to the 2D images
noise_filter = ScalarFilter(dim=2, value=noise_var)

# Then create noise with the ``NoiseAdder`` class.
noise = NoiseAdder(seed=123, noise_filter=noise_filter)

# Add noise to the images by performing ``forward``
noisy_im = noise.forward(src.images(0, src.n))

# Recast as an ASPIRE source
noisy_src = ArrayImageSource(noisy_im)

# Let's peek at the noisey images
noisy_src.images(0, 10).show()

# %%
# RIR with Noise
# ^^^^^^^^^^^^^^

# This also demonstrates changing the Nearest Neighbor search to using scikit-learn.
noisy_rir = RIRClass2D(
    noisy_src,
    fspca_components=400,
    bispectrum_components=300,
    n_nbor=10,
    n_classes=10,
    large_pca_implementation="legacy",
    nn_implementation="sklearn",
    bispectrum_implementation="legacy",
)

classes, reflections, rotations, shifts, corr = noisy_rir.classify()

# %%
# Display Classes
# ^^^^^^^^^^^^^^^

avgs = noisy_rir.output(classes, reflections, rotations)
avgs.images(0, 10).show()


# %%
# Review a class
# --------------
#
# Select a class to review.

review_class = 5

# Display the original image.
noisy_src.images(review_class, 1).show()

# Report the identified neighbor indices
logger.info(f"Class {review_class}'s neighors: {classes[review_class]}")

# Report the identified neighbors
Image(noisy_src.images(0, np.inf)[classes[review_class]]).show()

# Report their associated rots_refls
rots_refls = ["index, Rotation, Reflection"]
for i in range(classes.shape[1]):
    rots_refls.append(
        f"{i}, {rotations[review_class, i] * 180 / np.pi}, {reflections[review_class, i]}"
    )
rots_refls = "\n".join(rots_refls)

logger.info(
    f"Class {review_class}'s  estimated Rotations and Reflections:\n{rots_refls}"
)

# Display the averaged result
avgs.images(review_class, 1).show()
