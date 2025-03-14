"""
Ab-initio Pipeline Demonstration
================================

This tutorial demonstrates some key components of an ab-initio
reconstruction pipeline using synthetic data generated with ASPIRE's
``Simulation`` class of objects.
"""

# %%
# Download an Example Volume
# --------------------------
# We begin by downloading a high resolution volume map of the 80S
# Ribosome, sourced from EMDB: https://www.ebi.ac.uk/emdb/EMD-2660.
# This is one of several volume maps that can be downloaded with
# ASPIRE's data downloading utility by using the following import.
# sphinx_gallery_start_ignore
# flake8: noqa
# sphinx_gallery_end_ignore
from aspire.downloader import emdb_2660, emdb_8012

# Load 80s Ribosome as a ``Volume`` object.
original_vol = emdb_2660()
test_vol = emdb_8012()

# Downsample the volume
res = 41
vol = original_vol.downsample(res)

# %%
# .. note::
#     A ``Volume`` can be saved using the ``Volume.save()`` method as follows::
#
#         fn = f"downsampled_80s_ribosome_size{res}.mrc"
#         vol.save(fn, overwrite=True)


# %%
# Create a Simulation Source
# --------------------------
# ASPIRE's ``Simulation`` class can be used to generate a synthetic
# dataset of projection images.  A ``Simulation`` object produces
# random projections of a supplied Volume and applies noise and CTF
# filters. The resulting stack of 2D images is stored in an ``Image``
# object.


# %%
# CTF Filters
# ^^^^^^^^^^^^^^^^^^^^^
# Let's start by creating CTF filters. The ``operators`` package
# contains a collection of filter classes that can be supplied to a
# ``Simulation``.  We use ``RadialCTFFilter`` to generate a set of CTF
# filters with various defocus values.

# Create CTF filters
import numpy as np

from aspire.operators import RadialCTFFilter

# Radial CTF Filter
defocus_min = 15000  # unit is angstroms
defocus_max = 25000
defocus_ct = 7

ctf_filters = [
    RadialCTFFilter(pixel_size=vol.pixel_size, defocus=d)
    for d in np.linspace(defocus_min, defocus_max, defocus_ct)
]

# %%
# Initialize Simulation Object
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We feed our ``Volume`` and filters into ``Simulation`` to generate
# the dataset of images.  When controlled white Gaussian noise is
# desired, ``WhiteNoiseAdder.from_snr()`` can be used to generate a
# simulation data set around a specific SNR.
#
# Alternatively, users can bring their own images using an
# ``ArrayImageSource``, or define their own custom noise functions via
# ``Simulation(..., noise_adder=CustomNoiseAdder(...))``.  Examples
# can be found in ``tutorials/class_averaging.py`` and
# ``experiments/simulated_abinitio_pipeline.py``.

from aspire.noise import WhiteNoiseAdder
from aspire.source import Simulation

# set parameters
n_imgs = 2500

# SNR target for white gaussian noise.
snr = 0.5

# %%
# .. note::
#   Note, the SNR value was chosen based on other parameters for this
#   quick tutorial, and can be changed to adjust the power of the
#   additive noise.

# For this ``Simulation`` we set all 2D offset vectors to zero,
# but by default offset vectors will be randomly distributed.
src = Simulation(
    n=n_imgs,  # number of projections
    vols=vol,  # volume source
    offsets=0,  # Default: images are randomly shifted
    unique_filters=ctf_filters,
    noise_adder=WhiteNoiseAdder.from_snr(snr=snr),  # desired SNR
)


# %%
# Several Views of the Projection Images
# --------------------------------------
# We can access several views of the projection images.

# with no corruption applied
src.projections[0:10].show()

# %%

# with no noise corruption
src.clean_images[0:10].show()

# %%

# with noise and CTF corruption
src.images[0:10].show()


# %%
# CTF Correction
# --------------
# We apply ``phase_flip()`` to correct for CTF effects.

src = src.phase_flip()

# %%
# Cache
# -----
# We apply ``cache`` to store the results of the ``ImageSource``
# pipeline up to this point.  This is optional, but can provide
# benefit when used intently on machines with adequate memory.

src = src.cache()
src.images[0:10].show()

# %%
# Class Averaging
# ---------------
# We use ``RIRClass2D`` object to classify the images via the
# rotationally invariant representation (RIR) algorithm. Class
# selection is customizable. The classification module also includes a
# set of protocols for selecting a set of images to be used after
# classification.  Here we're using the simplest
# ``DebugClassAvgSource`` which internally uses the
# ``TopClassSelector`` to select the first ``n_classes`` images from
# the source.  In practice, the selection is done by sorting class
# averages based on some configurable notion of quality (contrast,
# neighbor distance etc).

from aspire.classification import RIRClass2D

# set parameters
n_classes = 200
n_nbor = 6

# We will customize our class averaging source. Note that the
# ``fspca_components`` and ``bispectrum_components`` were selected for
# this small tutorial.
rir = RIRClass2D(
    src,
    fspca_components=40,
    bispectrum_components=30,
    n_nbor=n_nbor,
)

from aspire.denoising import DebugClassAvgSource

avgs = DebugClassAvgSource(
    src=src,
    classifier=rir,
)

# We'll continue our pipeline using only the first ``n_classes`` from
# ``avgs``.  The ``cache()`` call is used here to precompute results
# for the ``:n_classes`` slice.  This avoids recomputing the same
# images twice when peeking in the next cell then requesting them in
# the following ``CLSyncVoting`` algorithm.  Outside of demonstration
# purposes, where we are repeatedly peeking at various stage results,
# such caching can be dropped allowing for more lazy evaluation.
avgs = avgs[:n_classes].cache()


# %%
# View the Class Averages
# -----------------------

# Show class averages
avgs.images[0:10].show()

# %%

# Show original images corresponding to those classes. This 1:1
# comparison is only expected to work because we used
# ``TopClassSelector`` to classify our images.
src.images[0:10].show()


# %%
# Orientation Estimation
# ----------------------
# We create an ``OrientedSource``, which consumes an ``ImageSource`` object, an
# orientation estimator, and returns a new source which lazily estimates orientations.
# In this case we supply ``avgs`` for our source and a ``CLSyncVoting``
# class instance for our orientation estimator. The ``CLSyncVoting`` algorithm employs
# a common-lines method with synchronization and voting.

from aspire.abinitio import CLSyncVoting
from aspire.source import OrientedSource

# Stash true rotations for later comparison
true_rotations = src.rotations[:n_classes]

# For this low resolution example we will customize the ``CLSyncVoting``
# instance to use fewer theta points ``n_theta`` then the default value of 360.
orient_est = CLSyncVoting(avgs, n_theta=72)

# Instantiate an ``OrientedSource``.
oriented_src = OrientedSource(avgs, orient_est)


# %%
# Mean Error of Estimated Rotations
# ---------------------------------
# ASPIRE has the built-in utility function, ``mean_aligned_angular_distance``, which globally
# aligns the estimated rotations to the true rotations and computes the mean
# angular distance (in degrees).

from aspire.utils import mean_aligned_angular_distance

# Compare with known true rotations
mean_ang_dist = mean_aligned_angular_distance(oriented_src.rotations, true_rotations)
print(f"Mean aligned angular distance: {mean_ang_dist} degrees")


# %%
# Volume Reconstruction
# ---------------------
# Now that we have our class averages and rotation estimates, we can
# estimate the mean volume by supplying the class averages and basis
# for back projection.

from aspire.reconstruction import MeanEstimator

# Setup an estimator to perform the back projection.
estimator = MeanEstimator(oriented_src)

# Perform the estimation and save the volume.
estimated_volume = estimator.estimate()


# %%
# Comparison of Estimated Volume with Source Volume
# -------------------------------------------------
# To get a visual confirmation that our results are sane, we rotate
# the estimated volume by the estimated rotations and project along
# the z-axis.  These estimated projections should align with the
# original projection images.

# Get the first 10 projections from the estimated volume using the
# estimated orientations.  Recall that ``project`` returns an
# ``Image`` instance, which we can peek at with ``show``.
projections_est = estimated_volume.project(oriented_src.rotations[0:10])

# We view the first 10 projections of the estimated volume.
projections_est.show()

# %%

# For comparison, we view the first 10 source projections.
src.projections[0:10].show()
