"""
================================
Ab-initio Pipeline Demonstration
================================

This tutorial demonstrates some key components of an ab-initio reconstruction pipeline using
synthetic data generated with ASPIRE's ``Simulation`` class of objects.
"""

# %%
# Download a Volume
# -----------------
# We begin by downloading a high resolution volume map of the 80S Ribosome, sourced from
# EMDB: https://www.ebi.ac.uk/emdb/EMD-2660.

import os

import numpy as np
import requests


# sphinx_gallery_start_ignore
# flake8: noqa
# sphinx_gallery_end_ignore
# Download volume
def download(url, save_path, chunk_size=1024 * 1024):
    r = requests.get(url, stream=True)
    with open(save_path, "wb") as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


if not os.path.exists("data/emd_2660.map"):
    url = "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-2660/map/emd_2660.map.gz"
    download(url, "data/emd_2660.map")

# %%
# Load a Volume
# -------------
# We use ASPIRE's ``Volume`` class to load and downsample the volume.

from aspire.volume import Volume

# Load 80s Ribosome
original_vol = Volume.load("data/emd_2660.map", dtype=np.float32)

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
# ASPIRE's ``Simulation`` class can be used to generate a synthetic dataset of projection images.
# A ``Simulation`` object produces random projections of a supplied Volume and applies noise and
# CTF filters. The resulting stack of 2D images is stored in an ``Image`` object.


# %%
# Noise and CTF Filters
# ^^^^^^^^^^^^^^^^^^^^^
# Let's start by creating noise and CTF filters. The ``operators`` package contains a collection
# of filter classes that can be supplied to a ``Simulation``. We use ``ScalarFilter`` to create
# Gaussian white noise and ``RadialCTFFilter`` to generate a set of CTF filters with various defocus values.

# Create noise and CTF filters
from aspire.noise import WhiteNoiseAdder
from aspire.operators import RadialCTFFilter

# Gaussian noise filter.
# Note, the value supplied to the ``WhiteNoiseAdder``, chosen based on other parameters
# for this quick tutorial, can be changed to adjust the power of the noise.
noise_adder = WhiteNoiseAdder(var=1e-5)

# Radial CTF Filter
defocus_min = 15000  # unit is angstroms
defocus_max = 25000
defocus_ct = 7

ctf_filters = [
    RadialCTFFilter(pixel_size=5, defocus=d)
    for d in np.linspace(defocus_min, defocus_max, defocus_ct)
]


# %%
# Initialize Simulation Object
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We feed our ``Volume`` and filters into ``Simulation`` to generate the dataset of images.

from aspire.source import Simulation

# set parameters
res = 41
n_imgs = 2500

# For this ``Simulation`` we set all 2D offset vectors to zero,
# but by default offset vectors will be randomly distributed.
src = Simulation(
    L=res,  # resolution
    n=n_imgs,  # number of projections
    vols=vol,  # volume source
    offsets=np.zeros((n_imgs, 2)),  # Default: images are randomly shifted
    noise_adder=noise_adder,
    unique_filters=ctf_filters,
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

src.phase_flip()
src.images[0:10].show()


# %%
# Class Averaging
# ---------------
# We use ``RIRClass2D`` object to classify the images via the rotationally invariant
# representation (RIR) algorithm. Class selection is customizable. The classification module
# also includes a set of protocols for selecting a set of images to be used for classification.
# Here we're using ``TopClassSelector``, which selects the first ``n_classes`` images from the source.

from aspire.classification import (
    BFSReddyChatterjiAverager2D,
    RIRClass2D,
    TopClassSelector,
)

# set parameters
n_classes = 200
n_nbor = 6

# Create a class averaging instance. Note that the ``fspca_components`` and
# ``bispectrum_components`` were selected for this small tutorial.
rir = RIRClass2D(
    src,
    fspca_components=40,
    bispectrum_components=30,
    n_nbor=n_nbor,
)

from aspire.basis import FFBBasis2D
from aspire.classification import BFSReddyChatterjiAverager2D

averager = BFSReddyChatterjiAverager2D(
    FFBBasis2D(src.L, dtype=src.dtype),
    src,
    num_procs=1,  # Change to "auto" if your machine has many processors
    dtype=rir.dtype,
)

from aspire.denoising import ClassAvgSource

avgs = ClassAvgSource(
    classification_src=src,
    classifier=rir,
    class_selector=TopClassSelector(),
    averager=averager,
)

# For a small example, it is more effective to just cache these now.
# avgs.cache()
from aspire.source import ArrayImageSource

avgs = ArrayImageSource(avgs.images[:n_classes])

# classify and average
# classes, reflections, distances = rir.classify()
# avgs = rir.averages(classes, reflections, distances)
# avgs = cls_src.images[:n_classes]

# %%
# View the Class Averages
# -----------------------

# Show class averages
avgs.images[0:10].show()

# %%

# Show original images corresponding to those classes. This 1:1 comparison is only expected to
# work because we used ``TopClassSelector`` to classify our images.
src.images[0:10].show()


# %%
# Orientation Estimation
# ----------------------
# We initialize a ``CLSyncVoting`` class instance for estimating the orientations of the images.
# The estimation employs the common lines method with synchronization and voting.

from aspire.abinitio import CLSyncVoting

# Stash true rotations for later comparison
true_rotations = src.rotations[:n_classes]

orient_est = CLSyncVoting(avgs, n_theta=72)

# Get the estimated rotations
orient_est.estimate_rotations()
rots_est = orient_est.rotations


# %%
# Mean Squared Error
# ------------------
# ASIPRE has some built-in utility functions for globally aligning the estimated rotations
# to the true rotations and computing the mean squared error.

from aspire.utils.coor_trans import (
    get_aligned_rotations,
    get_rots_mse,
    register_rotations,
)

# Compare with known true rotations
Q_mat, flag = register_rotations(rots_est, true_rotations)
regrot = get_aligned_rotations(rots_est, Q_mat, flag)
mse_reg = get_rots_mse(regrot, true_rotations)
mse_reg


# %%
# Volume Reconstruction
# ---------------------
# Now that we have our class averages and rotation estimates, we can estimate the
# mean volume by supplying the class averages and basis for back projection.

from aspire.basis import FFBBasis3D
from aspire.reconstruction import MeanEstimator

# Assign the estimated rotations to the class averages
avgs.rotations = rots_est

# Create a reasonable Basis for the 3d Volume
basis = FFBBasis3D(res, dtype=vol.dtype)

# Setup an estimator to perform the back projection.
estimator = MeanEstimator(avgs, basis)

# Perform the estimation and save the volume.
estimated_volume = estimator.estimate()


# %%
# Comparison of Estimated Volume with Source Volume
# -------------------------------------------------
# To get a visual confirmation that our results are sane, we rotate the
# estimated volume by the estimated rotations and project along the z-axis.
# These estimated projections should align with the original projection images.

from aspire.source import ArrayImageSource

# Get projections from the estimated volume using the estimated orientations.
# We instantiate the projections as an ``ArrayImageSource`` to access the ``Image.show()`` method.
projections_est = ArrayImageSource(estimated_volume.project(0, rots_est))

# We view the first 10 projections of the estimated volume.
projections_est.images[0:10].show()

# %%

# For comparison, we view the first 10 source projections.
src.projections[0:10].show()
