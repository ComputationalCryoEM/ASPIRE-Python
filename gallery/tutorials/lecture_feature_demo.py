"""
ASPIRE-Python Introduction
==========================

In this notebook we will introduce some code from ASPIRE-Python
that corresponds to topics from MATH586.
"""

# %%
# Imports
# -------
# First we import some of the usual suspects. In addition, we import some classes from
# the ASPIRE package that we will use throughout this tutorial.

# %%
# Installation
# ^^^^^^^^^^^^
#
# Attempt to install ASPIRE on your machine. ASPIRE can generally install on
# Linux, Mac, and Windows
# under Anaconda Python, by following the instructions in the README.
# `The instructions for developers is the most comprehensive
# <https://github.com/ComputationalCryoEM/ASPIRE-Python/blob/master/README.md#for-developers>`_.
# Linux is the most tested platform.
#
# ASPIRE requires some resources to run, so if you wouldn't run typical data
# science codes on your machine (maybe a netbook for example),
# you may use Tiger/Adroit/Della etc if you have access.
# After logging into Tiger, ``module load anaconda3/2020.7``
# and continue to follow the anaconda instructions for
# developers in the link above.
# Those instructions should create a working environment for tinkering with
# ASPIRE code found in this notebook.

import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from aspire.noise import (
    AnisotropicNoiseEstimator,
    CustomNoiseAdder,
    WhiteNoiseAdder,
    WhiteNoiseEstimator,
)
from aspire.operators import FunctionFilter
from aspire.source import RelionSource, Simulation
from aspire.utils import Rotation
from aspire.volume import Volume

# %%
# ``Image`` Class
# ---------------
#
# The `Image <https://computationalcryoem.github.io/ASPIRE-Python/aspire.image.html#aspire.image.image.Image>`_ class
# is a thin wrapper over numpy arrays for a stack containing 1 or more images.
# In this notebook we won't be working directly with the ``Image`` class a lot, but it will be one of the fundemental structures behind the scenes.
# A lot of ASPIRE code passes around ``Image`` and ``Volume`` classes.
#
# Examples of using the Image class can be found in:
#
# - ``gallery/tutorials/basic_image_array.py``
#
# - ``gallery/tutorials/image_class.py``

# %%
# ``Volume`` Class
# ----------------
#
# Like ``Image``, the `Volume <https://computationalcryoem.github.io/ASPIRE-Python/aspire.volume.html#aspire.volume.Volume>`_ class
# is a thin wrapper over numpy arrays that provides specialized methods for a stack containing 1 or more volumes.
#
# Here we will instantiate a Volume using a numpy array and use it to downsample to a desired resolution (64 should be good).
# For the data source I chose to download a real volume density map from `EMDB <https://www.ebi.ac.uk/pdbe/entry/emdb/EMD-2660>`_.
# The download was uncompressed in my local directory.  The notebook defaults to a small low resolution sample file you may use to sanity check.
# Unfortunately real data can be quite large so we do not ship it with the repo.

# %%
# Initialize Volume
# -----------------

# A low res example file is included in the repo as a sanity check.
# We can instantiate this as an ASPIRE Volume instance using ``Volume.load()``.
DATA_DIR = "data"
v = Volume.load(os.path.join(DATA_DIR, "clean70SRibosome_vol_65p.mrc"))

# More interesting data requires downloading locally.
# v = Volume.load("path/to/EMD-2660/map/emd_2660.map")

# Downsample the volume to a desired resolution
img_size = 64
# Volume.downsample() returns a new Volume instance.
#   We will use this lower resolution volume later, calling it `v2`.
v2 = v.downsample(img_size)
L = v2.resolution

# %%
# Plot Data
# ---------

# Alternatively, for quick sanity checking purposes we can view as a contour plot.
#   We'll use three orthographic projections, one per axis
fig, axs = plt.subplots(1, 3)
for i in range(3):
    axs[i].imshow(np.sum(v2.asnumpy()[0], axis=i), cmap="gray")
plt.show()

# %%
# ``Rotation`` Class - Generating Random Rotations
# ------------------------------------------------
#
# To get general projections this brings us to generating random rotations which we will apply to our volume.
#
# While you may bring your own 3x3 matrices or generate manually (say from your own Euler angles),
# ASPIRE has a `Rotation class <https://computationalcryoem.github.io/ASPIRE-Python/aspire.utils.html#module-aspire.utils.rotation>`_
# which can do this random rotation generation for us.  It also has some other utility methods if you would want to compare with something manual.
#
# The following code will generate some random rotations, and use the ``Volume.project()`` method to return an ``Image`` instance representing the stack of projections.
# We can display projection images using the ``Image.show()`` method.

num_rotations = 2
rots = Rotation.generate_random_rotations(n=num_rotations, seed=12345)

# %%
# We can access the numpy array holding the actual stack of 3x3 matrices:
logging.info(rots)
logging.info(rots.matrices)

# Using the first (and in this case, only) volume, compute projections using the stack of rotations:
projections = v.project(0, rots)
logging.info(projections)

# %%
# ``project()`` returns an Image instance, so we can call ``show``.

projections.show()
# Neat, we've generated random projections of some real data.

# %%
# The ``source`` Package
# ----------------------
#
# `aspire.source <https://computationalcryoem.github.io/ASPIRE-Python/aspire.source.html#module-aspire.source.simulation>`_
# package contains a collection of data source interfaces.
# The idea is that we can design an experiment using a synthetic ``Simulation`` source or our own provided array via ``ArrayImageSource``;
# then later swap out the source for a large experimental data set using something like ``RelionSource``.
#
# We do this because the experimental datasets are too large to fit in memory.
# They cannot be provided as a massive large array, and instead require methods to orchestrate batching.
# Depending on the application, they may also require corresponding batched algorithms.
# The ``Source`` classes try to make most of this opaque to an end user.  Ideally we can swap one source for another.
#
# For now we will build up to the creation and application of synthetic data set based on the real volume data used previously.

# %%
# ``Simulation`` Class
# --------------------
#
# Generating realistic synthetic data sources is a common task.
# The process of generating then projecting random rotations is integrated into the
# `Simulation <https://computationalcryoem.github.io/ASPIRE-Python/aspire.source.html#module-aspire.source.simulation>`_ class.
# Using ``Simulation``, we can generate arbitrary numbers of projections for use in experiments.
# Later we will demonstrate additional features which allow us to create more realistic data sources.

num_imgs = 100  # Total images in our source.
# Generate a Simulation instance based on the original volume data.
sim = Simulation(L=v.resolution, n=num_imgs, vols=v)
# Display the first 10 images
sim.images[:10].show()  # Hi Res

# Repeat for the lower resolution (downsampled) volume v2.
sim2 = Simulation(L=v2.resolution, n=num_imgs, vols=v2)
sim2.images[:10].show()  # Lo Res

# Note both of those simulations have the same rotations
#   because they had the same seed by default,
# We can set our own seed to get a different random samples (of rotations).
sim_seed = Simulation(L=v.resolution, n=num_imgs, vols=v, seed=42)
sim_seed.images[:10].show()

# We can also view the rotations used to create these projections
# logging.info(sim2.rotations)  # Commented due to long output

# %%
# Simulation with Filters and Noise
# ---------------------------------
#
# Filters
# ^^^^^^^
#
# `Filters <https://computationalcryoem.github.io/ASPIRE-Python/aspire.operators.html#module-aspire.operators.filters>`_
# are a collection of classes which once configured can be applied to ``Source`` pipelines.
# Common filters we might use are ``ScalarFilter``, ``PowerFilter``, ``FunctionFilter``, and ``CTFFilter``.
# ``CTFFilter`` is detailed in the ``ctf.py`` demo.
#
# Adding to Simulation
# ^^^^^^^^^^^^^^^^^^^^
#
# We can customize Sources by adding stages to their generation pipeline.
# In this case of a Simulation source, we want to corrupt the projection images with significant noise.
#
# First we create a constant two dimension filter (constant value set to our desired noise variance).
# Then when used in the ``noise_filter``, this scalar will be multiplied by a random sample.
# Similar to before, if you require a different sample, this would be controlled via a ``seed``.

# Get the sample variance
var = np.var(sim2.images[:].asnumpy())
logging.info(f"Sample Variance: {var}")
target_noise_variance = 100.0 * var
logging.info(f"Target Noise Variance: {target_noise_variance}")
# Then create a NoiseAdder based on that variance.
white_noise_adder = WhiteNoiseAdder(target_noise_variance)

# We can create a similar simulation with this additional noise_filter argument:
sim3 = Simulation(L=v2.resolution, n=num_imgs, vols=v2, noise_adder=white_noise_adder)
sim3.images[:10].show()
# These should be rather noisy now ...

# %%
# More Advanced Noise - Whitening
# -------------------------------
#
# We can estimate the noise across the stack of images
#
# The ``noise`` Package
# ^^^^^^^^^^^^^^^^^^^^^
#
# The `aspire.noise <https://computationalcryoem.github.io/ASPIRE-Python/aspire.noise.html>`_
# package contains several useful classes for generating and estimating different types of noise.
#
# In this case, we know the noise to be white, so we can proceed directly to
# `WhiteNoiseEstimator <https://computationalcryoem.github.io/ASPIRE-Python/aspire.noise.html#aspire.noise.noise.WhiteNoiseEstimator>`_.  The noise estimators consume from a ``Source``.
#
# The white noise estimator should log a diagnostic variance value. How does this compare with the known noise variance above?

# %%

# Create another Simulation source to tinker with.
sim_wht = Simulation(
    L=v2.resolution, n=num_imgs, vols=v2, noise_adder=white_noise_adder
)

# Estimate the white noise.
noise_estimator = WhiteNoiseEstimator(sim_wht)
logging.info(noise_estimator.estimate())

# %%
# A Custom ``FunctionFilter``
# ---------------------------
#
# We will now apply some more interesting noise, using a custom function, and then apply a ``whitening`` process to our data.
#
# Using ``FunctionFilter`` we can create our own custom functions to apply in a pipeline.
# Here we want to apply a custom filter as a noise adder.  We can use a function of two variables for example.


def noise_function(x, y):
    return 1e-7 * np.exp(-(x * x + y * y) / (2 * 0.3**2))


# In python, functions are first class objects.
# We take advantage of that to pass this function around as a variable.
# It will be evaluated later...
custom_noise = CustomNoiseAdder(noise_filter=FunctionFilter(noise_function))

# Create yet another Simulation source to tinker with.
sim4 = Simulation(L=v2.resolution, n=num_imgs, vols=v2, noise_adder=custom_noise)
sim4.images[:10].show()

# %%
# Noise Whitening
# ---------------
#
# Applying the ``Simulation.whiten()`` method just requires passing a `NoiseEstimator` instance.
# Then we can inspect some of the whitened images.  While noise is still present, we can see a dramatic change.

# Estimate noise.
aiso_noise_estimator = AnisotropicNoiseEstimator(sim4)

# Whiten based on the estimated noise
sim4 = sim4.whiten(aiso_noise_estimator)

# What do the whitened images look like...
sim4.images[:10].show()

# %%
# Real Experimental Data - ``RelionSource``
# -----------------------------------------
#
# Now that we have some basics,
# we can try to replace the simulation with a real experimental data source.
#
# Lets attempt the same CL experiment, but with a ``RelionSource``.

src = RelionSource(
    "data/sample_relion_data.star",
    data_folder="",
    pixel_size=5.0,
    max_rows=1024,
)

src = src.downsample(img_size)

src.images[:10].show()

# %%
# We have hit the point where we need denoising algorithms to perform orientation estimation as demonstrated in our ``pipeline_demo.py``.
