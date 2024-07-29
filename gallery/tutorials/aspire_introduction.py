"""
ASPIRE-Python Introduction
==========================

In this notebook we will introduce the core API components, then
demonstrate basic usage corresponding to topics from Princeton's
MAT586.
"""

# %%
# Installation
# ------------
#
# ASPIRE can generally install on Linux, Mac, and Windows under
# Anaconda Python, by following the instructions in the README.  `The
# instructions for developers is the most comprehensive
# <https://github.com/ComputationalCryoEM/ASPIRE-Python/blob/main/README.md#for-developers>`_.
# Windows is provided, but generally Linux and MacOS are recommended,
# with Linux being the most diversely tested platform.
#

# %%
# Princeton Research Computing
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ASPIRE requires some resources to run, so if you wouldn't run
# typical data science codes on your machine (a netbook for example),
# you may use Tiger/Adroit/Della at Princeton or another cluster.
# After logging into Tiger, ``module load anaconda3/2020.7`` and
# continue to follow the Anaconda instructions for developers in the
# link above.  Those instructions should create a working environment
# for tinkering with ASPIRE code found in this notebook.

# %%
# Imports
# ^^^^^^^
# First we import some typical scientific computing packages.
# Along the way we will import relevant components from ``aspire``.
# Users may also import ``aspire`` once as a top level package.

import os

import matplotlib.pyplot as plt
import numpy as np

import aspire
from aspire.image import Image

# %%
# API Primitives
# --------------
# The ASPIRE framework is a collection of modules containing
# interoperable extensible components.  Underlying the more
# sophisticated components and algorithms are some core data
# structures.  Sophisticated components are designed to interoperate
# by exchanging, consuming, or producing these basic structures.  The
# most common structures encountered when starting out are:

# %%
# .. list-table:: Core API Components
#    :header-rows: 1
#
#    * - Component
#      - Description
#    * - ``Coef``
#      - Utility class for stacks of 1D arrays.
#    * - ``Image``
#      - Utility class for stacks of 2D arrays.
#    * - ``Volume``
#      - Utility class for stacks of 3D arrays.
#    * - ``Rotations``
#      - Utility class for stacks of 3D rotations.
#    * - ``Filter``
#      - Constructs and applies Image filters.
#    * - ``Basis``
#      - Basis conversions and operations.
#    * - ``Source``
#      - Produces primitive components. ``ImageSource`` produces ``Image`` instances.


# %%
# ``Image`` Class
# ---------------
#
# The `Image
# <https://computationalcryoem.github.io/ASPIRE-Python/aspire.image.html#aspire.image.image.Image>`_
# class is a thin wrapper over Numpy arrays for a stack containing 1
# or more images (2D data).  In this notebook we won't be working
# directly with the ``Image`` class a lot, but it will be one of the
# fundamental structures behind the scenes.  A lot of ASPIRE code
# passes around ``Image`` and ``Volume`` instances.

# %%
# Create an ``Image`` instance from random data.
img_data = np.random.random((100, 100))
img = Image(img_data)
print(f"img shape: {img.shape}")  # Note this produces a stack of one.
print(f"str(img): {img}")

# %%
# Create an Image for a stack of 3 100x100 images.
img_data = np.random.random((3, 100, 100))
img = Image(img_data)

# %%
# Most often, Images will behave like Numpy arrays, but you
# explicitly access the underlying Numpy array via ``asnumpy()``.
img.asnumpy()

# %%
# Images have a built in ``show()`` method, which works well for
# peeking at data.
img.show()

# %%
# .. note::
#     The user is responsible for using ``show`` responsibly.  Avoid
#     asking for large numbers of images that you would not normally
#     plot.  Ten or less is reasonable.

# %%
# More examples using the Image class can be found in:
#
# - :ref:`sphx_glr_auto_tutorials_tutorials_image_class.py`
# - :ref:`sphx_glr_auto_tutorials_tutorials_basic_image_array.py`


# %%
# ``Volume`` Class
# ----------------
#
# Like ``Image``, the `Volume
# <https://computationalcryoem.github.io/ASPIRE-Python/aspire.volume.html#aspire.volume.Volume>`_
# class is a thin wrapper over Numpy arrays that provides specialized
# methods for a stack containing one or more volumes (3D data).

# %%
# Initialize Volume - ``load``
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# A ``Volume`` may be instantiated with Numpy data similarly to
# ``Image``.  Both ``Image`` and ``Volume`` provide ``save`` and
# ``load`` methods which can be used to work with files.  For
# ``Volumes`` ``.map`` and ``.mrc`` are currently supported.  For
# ``.npy``, Numpy can be used.
#
# For example, in the following note we demonstrate instantiating an ASPIRE Volume
# instance using ``Volume.load()``:
#
# .. note::
#    Instantiate an ASPIRE Volume from file::
#
#        from aspire.volume import Volume
#
#        aspire_volume = Volume.load("/path/to/volume.mrc")

# %%
# In addition to the ``Volume.load()`` method, a few common
# starting datasets can be downloaded from EMDB using ASPIRE's downloading
# utility. Below we download the high resolution volume map EMDB-2660, sourced from
# `<https://www.ebi.ac.uk/pdbe/entry/emdb/EMD-2660>`_.

from aspire.downloader import emdb_2660

vol = emdb_2660()

# %%
# Downsample Volume
# ^^^^^^^^^^^^^^^^^
# Here we downsample the above volume to a desired image size (64
# should be good).

img_size = 64

# Volume.downsample() returns a new Volume instance.
#   We will use this lower resolution volume later, calling it `v2`.
vol_ds = vol.downsample(img_size)
# L is often used as short hand for image and volume sizes (in pixels/voxels).
L = vol_ds.resolution

# %%
# Plot Data
# """""""""
# For quick sanity checking purposes we can view some plots.
#   We'll use three orthographic projections, one per axis.
orthographic_projections = np.empty((3, L, L), dtype=vol_ds.dtype)
for i in range(3):
    orthographic_projections[i] = np.sum(vol_ds, axis=(0, i + 1))
Image(orthographic_projections).show()

# %%
# ``Rotation`` Class
# ------------------
# While you may bring your own 3x3 matrices or generate manually (say
# from your own Euler angles), ASPIRE has a `Rotation class
# <https://computationalcryoem.github.io/ASPIRE-Python/aspire.utils.html#module-aspire.utils.rotation>`_
# which can do this random rotation generation for us.  It also has
# some other utility methods, including support for Rodrigues
# rotations (ie, axis-angle).  Other ASPIRE components dealing with 3D
# rotations will generally expect instances of ``Rotation``.
#
# A common task in computational cryo-EM is generating random
# projections, by applying random 3D rotations to a volume and projecting along the z-axis.
# The following code will generate some random rotations,
# and use the ``Volume.project()`` method to return an ``Image``
# instance representing the stack of projections.  We can display
# projection images using the ``Image.show()`` method.

from aspire.utils import Rotation

num_rotations = 2
rots = Rotation.generate_random_rotations(n=num_rotations, seed=12345)

# %%
# We can access the Numpy array holding the actual stack of 3x3 matrices:
print(rots)
print(rots.matrices)

# %%
# Using the ``Volume.project()`` method we compute
# projections using the stack of rotations:
projections = vol.project(rots)
print(projections)

# %%
# ``project()`` returns an Image instance, so we can call ``show``.
projections.show()

# %%
# Neat, we've generated random projections of some real data.  This
# tutorial will go on to show how this can be performed systematically with
# other cryo-EM data simulation tasks.

# %%
# The ``filter`` Package
# ----------------------
# `Filters
# <https://computationalcryoem.github.io/ASPIRE-Python/aspire.operators.html#module-aspire.operators.filters>`_
# are a collection of classes which once configured can be applied to
# ``Images``, typically in an ``ImageSource`` pipeline which will be
# discussed in a later section.  Specifically, applying a ``Filter``
# convolves the filter with the images contained in the ``Image``
# instance.

# %%
#
#  .. mermaid::
#
#    classDiagram
#        class Filter{
#            +evaluate()
#            +basis_mat()
#            +scale()
#            +evaluate_grid()
#            +dual()
#            +sign()
#         }
#
#         Filter o-- FunctionFilter
#         Filter o-- ArrayFilter
#         Filter o-- ScalarFilter
#         Filter o-- ZeroFilter
#         Filter o-- IdentityFilter
#         Filter o-- CTFFilter
#         CTFFilter o-- RadialCTFFilter

# %%
# ``CTFFilter`` and ``RadialCTFFilter`` are the most common filters
# encountered when starting out and are detailed in
# :ref:`sphx_glr_auto_tutorials_tutorials_ctf.py`.  The other filters
# are used behind the scenes in components like ``NoiseAdders`` or
# more advanced customized pipelines.  Several filters for internal or
# advanced use cases are omitted from the diagram, but can be found in
# the `aspire.operators.filter` module.

# %%
# ``Basis``
# ---------
# ASPIRE provides a selection of ``Basis`` classes designed for
# working with cryo-EM data in two and three dimensions.  Most of
# these basis implementations are optimized for efficient rotations,
# often called the *"steerable"* property.  As of this writing most
# algorithms in ASPIRE are written to work well with the fast
# Fourier-Bessel (FFB) basis classes ``FFBBasis2D`` and
# ``FFBBasis3D``.  These correspond to direct slower reference
# ``FBBasis2D`` and ``FBBasis3D`` classes.
#
# Recently, a related Fourier-Bessel method using fast Laplacian
# eigenfunction (FLE) transforms was integrated as ``FLEBasis2D``.
# Additional prolate spheroidal wave function (PSWF) methods are
# available via ``FPSWFBasis2D`` and ``FPSWFBasis3D``, but their
# integration into other components like 2D covariance analysis is
# incomplete, and slated for a future release.

# %%
# The ``source`` Package
# ----------------------
#
# The `aspire.source
# <https://computationalcryoem.github.io/ASPIRE-Python/aspire.source.html#module-aspire.source.simulation>`_
# package contains a collection of data source interfaces.
# Ostensibly, a ``Source`` is a producer of some primitive type, most
# notably ``Image``.  ASPIRE components that consume (process) images
# are designed to accept an ``ImageSource``.
#
# The first reason for this is to normalize the way a wide variety of
# higher-level components interface.  ``ImageSource`` instances have a
# consistent property ``images`` which must be implemented to serve up
# images dynamically using a square-bracket ``[]`` syntax familiar to
# Numpy users. This supports batch computation among other things.
# ``Source`` instances also store and serve up metadata like
# `rotations`, `dtype`, and support pipelining transformations.
#
# The second reason is so we can design an experiment using a
# synthetic ``Simulation`` source or our own provided Numpy arrays via
# ``ArrayImageSource`` and then later swap out the source for a large
# experimental data set using something like ``RelionSource``.
# Experimental datasets can be too large to practically fit or process
# entirely in memory, and force the use of iteratively-batched
# approaches.
#
# Generally, the ``source`` package attempts to make most of this
# opaque to an end user.  Ideally we can simply swap one source for
# another.  For now we will build up to the creation and application
# of synthetic data set based on the various manual interactions
# above.

# %%
#
#  .. mermaid::
#
#    classDiagram
#        class ImageSource{
#            +L
#            +n
#            +dtype
#            ...
#            +images[]
#            +cache()
#            +downsample()
#            +whiten()
#            +phase_flip()
#            +invert_conrast()
#            +normalize_background()
#            +save()
#            +save_images()
#            ...
#            }
#        ImageSource o-- ArrayImageSource
#        ImageSource o-- Simulation
#        ImageSource o-- RelionSource
#        ImageSource o-- CoordinateSource
#        CoordinateSource o-- BoxesCoordinateSource
#        CoordinateSource o-- CentersCoordinateSource


# %%
# ``Simulation`` Class
# ^^^^^^^^^^^^^^^^^^^^
# Generating realistic synthetic data sources is a common task.  The
# process of generating then projecting random rotations is integrated
# into the `Simulation
# <https://computationalcryoem.github.io/ASPIRE-Python/aspire.source.html#module-aspire.source.simulation>`_
# class.  Using ``Simulation``, we can generate arbitrary numbers of
# projections for use in experiments.  Then additional features are
# introduced which allow us to create more realistic data sources.

from aspire.source import Simulation

# Total images in our source.
num_imgs = 100

# %%
# Generate a Simulation instance based on the original volume data.
sim = Simulation(n=num_imgs, vols=vol)
# Display the first 10 images
sim.images[:10].show()  # Hi Res

# %%
# Repeat for the lower resolution (downsampled) volume vol_ds.
sim = Simulation(n=num_imgs, vols=vol_ds)
sim.images[:10].show()  # Lo Res

# %%
# Note both of those simulations have the same rotations because they
# had the same seed by default, We recreate ``sim`` with a distinct
# seed to get different random samples (of rotations).
sim = Simulation(n=num_imgs, vols=vol_ds, seed=42)
sim.images[:10].show()

# %%
# We can also view the rotations used to create these projections.
print(sim.rotations)

# %%
# Given any ``Source``, we can also take slices using typical slicing
# syntax, or provide our own iterable of indices.

sim_evens = sim[0::2]
sim_odds = sim[1::2]

# We can also generate random selections.
# Shuffle indices then take the first 5.
shuffled_inds = np.random.choice(sim.n, 5, replace=False)
sim_shuffled_subset = sim[shuffled_inds]

# %%
# Underneath those slices, ASPIRE relies on ``IndexedSource``, which
# we can also call direcly to remap indices.

from aspire.source import IndexedSource

sim_shuffled_subset = IndexedSource(sim, shuffled_inds)


# %%
# The ``noise`` Package
# ---------------------
# The `aspire.noise
# <https://computationalcryoem.github.io/ASPIRE-Python/aspire.noise.html>`_
# package contains several useful classes for generating and
# estimating different types of noise.

# %%
# ``NoiseAdder``
# ^^^^^^^^^^^^^^
# ``NoiseAdder`` subclasses are used to add common or customized noise
# to ``Simulation`` image generation pipelines.

# %%
# ``WhiteNoiseAdder``
# """""""""""""""""""
# ``WhiteNoiseAdder`` is the most common type of synthetic noise.

from aspire.noise import WhiteNoiseAdder

# %%
# Get the sample variance, then create a NoiseAdder based on that variance.
var = np.var(sim.images[:].asnumpy())
print(f"Sample Variance: {var}")
target_noise_variance = 10.0 * var
print(f"Target Noise Variance: {target_noise_variance}")
white_noise_adder = WhiteNoiseAdder(target_noise_variance)

# %%
# We can customize Sources by adding stages to their generation
# pipeline.  In this case of a Simulation source, we want to corrupt
# the projection images with noise.  Internally the
# ``WhiteNoiseAdder`` creates a ``ScalarFilter`` which is multiplied
# (convolution) by a Gaussian random sample.  Similar to before, if
# you require a different sample, this can be controlled via a
# ``seed``.

# Creating the new simulation with this additional noise is easy:
sim = Simulation(n=num_imgs, vols=vol_ds, noise_adder=white_noise_adder)
# These should be rather noisy now ...
sim.images[:10].show()

# %%
# ``WhiteNoiseEstimator``
# """""""""""""""""""""""
# We can estimate the noise across an ``ImageSource``, and
# we've generated a simulation with known noise variance.
# Lets see how the estimate compares.
#
# In this case, we know the noise to be white, so we can proceed directly to
# `WhiteNoiseEstimator <https://computationalcryoem.github.io/ASPIRE-Python/aspire.noise.html#aspire.noise.noise.WhiteNoiseEstimator>`_.
# The noise estimators consume from an ``ImageSource``.
#
# The white noise estimator should log a diagnostic variance value.
# Internally, it also uses the estimation results to build a
# ``Filter`` which can be used in more advanced denoising methods.

from aspire.noise import WhiteNoiseEstimator

noise_estimator = WhiteNoiseEstimator(sim)
noise_estimator.estimate()


# %%
# A Custom ``FunctionFilter``
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We will now apply some more interesting noise, using a custom
# function, and then apply a ``whitening`` process to our data.
#
# Using ``FunctionFilter`` we can create our own custom functions to
# apply in a pipeline.  Here we want to apply a custom noise function.
# We will use a function of two variables for this example.

from aspire.noise import CustomNoiseAdder
from aspire.operators import FunctionFilter


def noise_function(x, y):
    return 1e-7 * np.exp(-(x * x + y * y) / (2 * 0.3**2))


# In Python, functions are first class objects.  We take advantage of
# that to pass this function around as a variable.  The function is
# evaluated later, internally, during pipeline execution.
custom_noise = CustomNoiseAdder(noise_filter=FunctionFilter(noise_function))

# Create yet another Simulation source to tinker with.
sim = Simulation(n=num_imgs, vols=vol_ds, noise_adder=custom_noise)
sim.images[:10].show()

# %%
# Noise Whitening
# ^^^^^^^^^^^^^^^
# We will now combine a more advanced noise estimation technique with
# an ``ImageSource`` preprocessing method ``whiten``.
#
# First an anisotropic noise estimate is performed.

from aspire.noise import AnisotropicNoiseEstimator

# Estimate noise.
aiso_noise_estimator = AnisotropicNoiseEstimator(sim)

# %%
# Applying the ``Simulation.whiten()`` method requires passing a
# corresponding ``NoiseEstimator`` instance.  Then we can inspect some
# of the whitened images.  While noise is still present, we can see a
# dramatic change.

# Whiten based on the estimated noise.
sim = sim.whiten(aiso_noise_estimator)

# %%
# What do the whitened images look like?
sim.images[:10].show()

# %%
# Common Image Corruptions
# ------------------------
# ``Simulation`` provides several configurable types of common cryo-EM
# image corruptions.  Users should be aware that amplitude and offset
# corruption is enabled by default.

# %%
# Amplitudes
# ^^^^^^^^^^
# Simulation automatically generates random amplitude variability.
# To disable, set to ``amplitudes=1``.

# %%
# Offsets
# ^^^^^^^
# Simulation automatically generates random offsets.
# To disable, set to ``offsets=0``.

# %%
# Noise
# ^^^^^
# By default, no noise corruption is configured.
# To enable, see ``NoiseAdder`` components.

# %%
# CTF
# ^^^
# By default, no CTF corruption is configured.
# To enable, we must configure one or more ``CTFFilter`` instances.
# Usually we will create a range of filters for a variety of
# defocus levels.

from aspire.operators import RadialCTFFilter

# Radial CTF Filter params.
defocus_min = 15000  # unit is angstroms
defocus_max = 25000
defocus_ct = 7

# Generate several CTFs.
ctf_filters = [
    RadialCTFFilter(pixel_size=vol_ds.pixel_size, defocus=d)
    for d in np.linspace(defocus_min, defocus_max, defocus_ct)
]

# %%
# Combining into a Simulation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Here we'll combine the parameters above into a new simulation.

sim = Simulation(
    n=num_imgs,
    vols=vol_ds,
    amplitudes=1,
    offsets=0,
    noise_adder=white_noise_adder,
    unique_filters=ctf_filters,
    seed=42,
)

# Simulation has two unique accessors ``clean_images`` which disables
# noise, and ``projections`` which are clean uncorrupted projections.
# Both act like calls to `image` and return show-able ``Image``
# instances.

# %%
# Clean projections.
sim.projections[:3].show()

# %%
# Images with only CTF applied.
sim.clean_images[:3].show()

# %%
# And now the first four corrupted images.
sim.images[:4].show(columns=2, colorbar=False)

# %%
# .. note::
#     Above the ``show`` call has been customized as a 2 column grid
#     with out colorbar legend.

# %%
# Real Experimental Data - ``RelionSource``
# -----------------------------------------
#
# Now that we have some basics, we can try to replace the simulation
# with a real experimental data source.

from aspire.source import RelionSource

src = RelionSource(
    "data/sample_relion_data.star",
    data_folder="",
    pixel_size=5.0,
    max_rows=1024,
)

# %%
# Add downsampling to the ``src`` pipeline.
src = src.downsample(img_size)

# %%
# ``RelionSource`` will auto-populate ``CTFFilter`` instances from the
# STAR file metadata when available. Having these filters allows us to
# perform a phase flipping correction.
src = src.phase_flip()

# %%
# Display the experimental data images.
src.images[:10].show()

# %%
# Pipeline Roadmap
# ----------------
# Now that the primitives have been introduced we can explore
# higher-level components.  The higher-level components are designed
# to be modular and cacheable (to memory or disk) to support
# experimentation with entire pipelines or focused algorithmic
# development on specific components.  Most pipelines will follow a
# flow of data and components moving mostly left to right in the table
# below.  This table is not exhaustive, but represents some of the
# most common components.

# %%
# +----------------+--------------------+-----------------+----------------+---------------------+
# |  Image Processing                                     | Ab initio                            |
# +----------------+--------------------+-----------------+----------------+---------------------+
# | Data           | Preprocessing      | Denoising       | Orientation    |  3D Reconstruction  |
# +================+====================+=================+================+=====================+
# |Simulation      | NoiseEstimator     | Class Averaging | CLSyncVoting   | MeanVolumeEstimator |
# +----------------+--------------------+-----------------+----------------+---------------------+
# |RelionSource    | downsample         | cov2d (CWF)     | CLSymmetryC2   |                     |
# +----------------+--------------------+-----------------+----------------+---------------------+
# |CoordinateSource| whiten             |                 | CLSymmetryC3C4 |                     |
# +----------------+--------------------+-----------------+----------------+---------------------+
# |                | phase_flip         |                 | CLSymmetryCn   |                     |
# +----------------+--------------------+-----------------+----------------+---------------------+
# |                |normalize_background|                 | CommonlineSDP  |                     |
# +----------------+--------------------+-----------------+----------------+---------------------+
# |                | CTFEstimator       |                 |                |                     |
# +----------------+--------------------+-----------------+----------------+---------------------+

# %%
# We're now ready to explore a small example end-to-end ab initio
# pipeline using simulated data.
# :ref:`sphx_glr_auto_tutorials_pipeline_demo.py`

# %%
# Larger simulations and experiments based on EMPIAR data can be found
# in :ref:`Experiments <exp>`.
