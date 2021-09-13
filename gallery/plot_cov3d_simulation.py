"""
Covariance 3D Estimation
========================

This script illustrates the example of Covariance 3D estimation using simulation images
generated from Gaussian blob volumes.
"""
import logging

import numpy as np
from scipy.cluster.vq import kmeans2

from aspire.basis import FBBasis3D
from aspire.covariance import CovarianceEstimator
from aspire.denoising import src_wiener_coords
from aspire.noise import WhiteNoiseEstimator
from aspire.operators import RadialCTFFilter
from aspire.reconstruction import MeanEstimator
from aspire.source.simulation import Simulation
from aspire.utils import eigs
from aspire.utils.random import Random
from aspire.volume import Volume

logger = logging.getLogger(__name__)

# %%
# Create Simulation Object
# ------------------------

# Specify parameters
num_vols = 2  # number of volumes
img_size = 8  # image size in square
num_imgs = 1024  # number of images
num_eigs = 16  # number of eigen-vectors to keep

# Create a simulation object with specified filters
sim = Simulation(
    L=img_size,
    n=num_imgs,
    C=num_vols,
    unique_filters=[RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)],
)

# Specify the normal FB basis method for expending the 2D images
basis = FBBasis3D((img_size, img_size, img_size))

# Estimate the noise variance. This is needed for the covariance estimation step below.
noise_estimator = WhiteNoiseEstimator(sim, batchSize=500)
noise_variance = noise_estimator.estimate()
logger.info(f"Noise Variance = {noise_variance}")

# %%
# Estimate Mean Volume and Covariance
# -----------------------------------
#
# Estimate the mean. This uses conjugate gradient on the normal equations for
# the least-squares estimator of the mean volume. The mean volume is represented internally
# using the basis object, but the output is in the form of an
# L-by-L-by-L array.

mean_estimator = MeanEstimator(sim, basis)
mean_est = mean_estimator.estimate()

# Passing in a mean_kernel argument to the following constructor speeds up some calculations
covar_estimator = CovarianceEstimator(sim, basis, mean_kernel=mean_estimator.kernel)
covar_est = covar_estimator.estimate(mean_est, noise_variance)

# %%
# Use Top Eigenpairs to Form a Basis
# ----------------------------------

# Extract the top eigenvectors and eigenvalues of the covariance estimate.
# Since we know the population covariance is low-rank, we are only interested
# in the top eigenvectors.

eigs_est, lambdas_est = eigs(covar_est, num_eigs)

# Eigs returns column-major, so we transpose and construct a volume.
eigs_est = Volume(np.transpose(eigs_est, (3, 0, 1, 2)))

# Truncate the eigendecomposition. Since we know the true rank of the
# covariance matrix, we enforce it here.

eigs_est_trunc = Volume(eigs_est[: num_vols - 1])  # hrmm not very convenient
lambdas_est_trunc = lambdas_est[: num_vols - 1, : num_vols - 1]

# Estimate the coordinates in the eigenbasis. Given the images, we find the
# coordinates in the basis that minimize the mean squared error, given the
# (estimated) covariances of the volumes and the noise process.
coords_est = src_wiener_coords(
    sim, mean_est, eigs_est_trunc, lambdas_est_trunc, noise_variance
)

# Cluster the coordinates using k-means. Again, we know how many volumes
# we expect, so we can use this parameter here. Typically, one would take
# the number of clusters to be one plus the number of eigenvectors extracted.

# Since kmeans2 relies on randomness for initialization, important to push random seed to context manager here.
with Random(0):
    centers, vol_idx = kmeans2(coords_est.T, num_vols)
    centers = centers.squeeze()

# %%
# Performance Evaluation
# ----------------------

# Evaluate performance of mean estimation.

mean_perf = sim.eval_mean(mean_est)


# Evaluate performance of covariance estimation. We also evaluate the truncated
# eigendecomposition. This is expected to be a closer approximation since it
# imposes an additional low-rank condition on the estimate.

covar_perf = sim.eval_covar(covar_est)
eigs_perf = sim.eval_eigs(eigs_est_trunc, lambdas_est_trunc)

# Evaluate clustering performance.

clustering_accuracy = sim.eval_clustering(vol_idx)

# Assign the cluster centroids to the different images. Since we expect a discrete distribution of volumes
# (and therefore of coordinates), we assign the centroid coordinate to each image that belongs to that cluster.
# Evaluate the coordinates estimated

clustered_coords_est = centers[vol_idx]
coords_perf = sim.eval_coords(mean_est, eigs_est_trunc, clustered_coords_est)

# %%
# Results
# -------

# Output estimated covariance spectrum.

logger.info(f"Population Covariance Spectrum = {np.diag(lambdas_est)}")


# Output performance results.

logger.info(f'Mean (rel. error) = {mean_perf["rel_err"]}')
logger.info(f'Mean (correlation) = {mean_perf["corr"]}')
logger.info(f'Covariance (rel. error) = {covar_perf["rel_err"]}')
logger.info(f'Covariance (correlation) = {covar_perf["corr"]}')
logger.info(f'Eigendecomposition (rel. error) = {eigs_perf["rel_err"]}')
logger.info(f"Clustering (accuracy) = {clustering_accuracy}")
logger.info(f'Coordinates (mean rel. error) = {coords_perf["rel_err"]}')
logger.info(f'Coordinates (mean correlation) = {np.mean(coords_perf["corr"])}')

# Basic Check
assert covar_perf["rel_err"] <= 0.60
assert np.mean(coords_perf["corr"]) >= 0.98
assert clustering_accuracy >= 0.99
