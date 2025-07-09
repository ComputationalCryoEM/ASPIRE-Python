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

from aspire.downloader import emdb_2660

# Load 80s Ribosome as a ``Volume`` object.
original_vol = emdb_2660()

# During the preprocessing stages of the pipeline we will downsample
# the images to an image size of 64 pixels. Here, we also downsample the
# volume so we can compare to our reconstruction later.
res = 64
vol_ds = original_vol.downsample(res)

# %%
# .. note::
#     A ``Volume`` can be saved using the ``Volume.save()`` method as follows::
#
#         fn = f"downsampled_80s_ribosome_size{res}.mrc"
#         vol_ds.save(fn, overwrite=True)


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
    RadialCTFFilter(pixel_size=original_vol.pixel_size, defocus=d)
    for d in np.linspace(defocus_min, defocus_max, defocus_ct)
]

# %%
# Initialize Simulation Object
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We feed our ``Volume`` and filters into ``Simulation`` to generate
# the dataset of images.  When controlled white Gaussian noise is
# desired, ``WhiteNoiseAdder(var=VAR)`` can be used to generate a
# simulation data set around a specific noise variance.
#
# Alternatively, users can bring their own images using an
# ``ArrayImageSource``, or define their own custom noise functions via
# ``Simulation(..., noise_adder=CustomNoiseAdder(...))``.  Examples
# can be found in ``tutorials/class_averaging.py`` and
# ``experiments/simulated_abinitio_pipeline.py``.

from aspire.noise import WhiteNoiseAdder
from aspire.source import Simulation

# For this ``Simulation`` we set all 2D offset vectors to zero,
# but by default offset vectors will be randomly distributed.
# We cache the Simulation to prevent regenerating the projections
# for each preprocessing stage.
src = Simulation(
    n=2500,  # number of projections
    vols=original_vol,  # volume source
    offsets=0,  # Default: images are randomly shifted
    unique_filters=ctf_filters,
    noise_adder=WhiteNoiseAdder(var=0.0002),  # desired noise variance
).cache()

# %%
# .. note::
#   The noise variance value above was chosen based on other parameters for this
#   quick tutorial, and can be changed to adjust the power of the additive noise.
#   Alternatively, an SNR value can be prescribed as follows::
#
#       Simulation(..., noise_adder=WhiteNoiseAdder.from_snr(SNR))

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
# Image Preprocessing
# -------------------
# We apply some image preprocessing techniques to prepare the
# the images for denoising via Class Averaging.

# %%
# Downsampling
# ------------
# We downsample the images. Reducing the image size will improve the
# efficiency of subsequent pipeline stages. Metadata such as pixel size is
# scaled accordingly to correspond correctly with the image resolution.

src = src.downsample(res)
src.images[:10].show()

# %%
# CTF Correction
# --------------
# We apply ``phase_flip()`` to correct for CTF effects.

src = src.phase_flip()
src.images[:10].show()

# %%
# Normalize Background
# --------------------
# We apply ``normalize_background()`` to prepare the image class averaging.

src = src.normalize_background()
src.images[:10].show()

# %%
# Noise Whitening
# ---------------
# We apply ``whiten()`` to estimate and whiten the noise.

src = src.whiten()
src.images[:10].show()

# %%
# Contrast Inversion
# ------------------
# We apply ``invert_contrast()`` to ensure a positive valued signal.

src = src.invert_contrast()

# %%
# Caching
# -------
# We apply ``cache`` to store the results of the ``ImageSource``
# pipeline up to this point.  This is optional, but can provide
# benefit when used intently on machines with adequate memory.
src = src.cache()

# %%
# Class Averaging
# ---------------
# For this tutorial we use the ``DebugClassAvgSource`` to generate an ``ImageSource``
# of class averages. Internally, ``DebugClassAvgSource`` uses the ``RIRClass2D``
# object to classify the source images via the rotationally invariant representation
# (RIR) algorithm and the ``TopClassSelector`` object to select the first ``n_classes``
# images in the original order from the source. In practice, class selection is commonly
# done by sorting class averages based on some configurable notion of quality
# (contrast, neighbor distance etc) which can be accomplished by providing a custom
# class selector to ``ClassAverageSource``, which changes the ordering of the classes
# returned by ``ClassAverageSource``.

from aspire.denoising import DebugClassAvgSource

avgs = DebugClassAvgSource(src=src)

# We'll continue our pipeline using only the first ``n_classes`` from
# ``avgs``.  The ``cache()`` call is used here to precompute results
# for the ``:n_classes`` slice.  This avoids recomputing the same
# images twice when peeking in the next cell then requesting them in
# the following ``CLSyncVoting`` algorithm.  Outside of demonstration
# purposes, where we are repeatedly peeking at various stage results,
# such caching can be dropped allowing for more lazy evaluation.
n_classes = 250
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
from aspire.utils import Rotation

# Stash true rotations for later comparison
true_rotations = Rotation(src.rotations[:n_classes])

# Instantiate a ``CLSyncVoting`` object for estimating orientations.
orient_est = CLSyncVoting(avgs)

# Instantiate an ``OrientedSource``.
oriented_src = OrientedSource(avgs, orient_est)

# Estimate Rotations.
est_rotations = oriented_src.rotations

# %%
# Mean Error of Estimated Rotations
# ---------------------------------
# ASPIRE has the built-in utility function, ``mean_aligned_angular_distance``, which globally
# aligns the estimated rotations to the true rotations and computes the mean
# angular distance (in degrees).

from aspire.utils import mean_aligned_angular_distance

# Compare with known true rotations
mean_ang_dist = mean_aligned_angular_distance(est_rotations, true_rotations)
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
projections_est = estimated_volume.project(est_rotations[0:10])

# We view the first 10 projections of the estimated volume.
projections_est.show()

# %%

# For comparison, we view the first 10 source projections.
src.projections[0:10].show()


# %%
# Fourier Shell Correlation
# -------------------------
# Additionally, we can compare our reconstruction to the known source volume
# by performing a Fourier shell correlation (FSC). We first find a rotation
# matrix which best aligns the estimated rotations to the ground truth rotations
# using the ``find_registration`` method. We then use that rotation to align
# the reconstructed volume to the ground truth volume.

# `find_registration` returns the best aligning rotation, `Q`, as well as
# a `flag` which indicates if the volume needs to be reflected.
Q, flag = Rotation(est_rotations).find_registration(true_rotations)
aligned_vol = estimated_volume
if flag == 1:
    aligned_vol = aligned_vol.flip()
aligned_vol = aligned_vol.rotate(Rotation(Q.T))

# Compute the FSC.
vol_ds.fsc(aligned_vol, cutoff=0.143, plot=True)
