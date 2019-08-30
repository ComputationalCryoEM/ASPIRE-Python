import logging
import numpy as np
from scipy.cluster.vq import kmeans2
import click

from aspire.source.simulation import Simulation
from aspire.basis.fb_3d import FBBasis3D
from aspire.utils.filters import RadialCTFFilter
from aspire.estimation.noise import WhiteNoiseEstimator
from aspire.estimation.mean import MeanEstimator
from aspire.estimation.covar import CovarianceEstimator
from aspire.utils.matlab_compat import Random
from aspire.utils.matrix import eigs
from aspire.utils.misc import src_wiener_coords


logger = logging.getLogger('aspire')


@click.command()
@click.option('--num_volumes', default=2, type=int, help='No. of distinct volumes')
@click.option('--image_size', default=8, type=int, help='Image size in pixels')
@click.option('--num_images', default=1024, type=int, help='No. of images')
@click.option('--num_eigs', default=16, type=int, help='Number of EigenVectors to estimate')
def simulation(num_volumes, image_size, num_images, num_eigs):
    """Generate a Simulation and run Covariance estimation."""

    C = num_volumes
    L = image_size
    n = num_images

    sim = Simulation(
        n=n,
        C=C,
        filters=[RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)]
    )
    basis = FBBasis3D((L, L, L))

    noise_estimator = WhiteNoiseEstimator(sim, batchSize=500)
    # Estimate the noise variance. This is needed for the covariance estimation step below.
    noise_variance = noise_estimator.estimate()
    logger.info(f'Noise Variance = {noise_variance}')

    """
    Estimate the mean. This uses conjugate gradient on the normal equations for the least-squares estimator of the mean
    volume. The mean volume is represented internally using the basis object, but the output is in the form of an
    L-by-L-by-L array.
    """
    mean_estimator = MeanEstimator(sim, basis)
    mean_est = mean_estimator.estimate()

    # Passing in a mean_kernel argument to the following constructor speeds up some calculations
    covar_estimator = CovarianceEstimator(sim, basis, mean_kernel=mean_estimator.kernel)
    covar_est = covar_estimator.estimate(mean_est, noise_variance)

    """
    Extract the top eigenvectors and eigenvalues of the covariance estimate.
    Since we know the population covariance is low-rank, we are only interested
    in the top eigenvectors.
    """
    eigs_est, lambdas_est = eigs(covar_est, num_eigs)

    """
    Truncate the eigendecomposition. Since we know the true rank of the
    covariance matrix, we enforce it here.
    """
    eigs_est_trunc = eigs_est[:, :, :, :C-1]
    lambdas_est_trunc = lambdas_est[:C-1, :C-1]

    # Estimate the coordinates in the eigenbasis. Given the images, we find the coordinates in the basis that
    # minimize the mean squared error, given the (estimated) covariances of the volumes and the noise process.
    coords_est = src_wiener_coords(sim, mean_est, eigs_est_trunc, lambdas_est_trunc, noise_variance)

    # Cluster the coordinates using k-means. Again, we know how many volumes we expect, so we can use this parameter
    # here. Typically, one would take the number of clusters to be one plus the number of eigenvectors extracted.

    # Since kmeans2 relies on randomness for initialization, important to push random seed to context manager here.
    with Random(0):
        centers, vol_idx = kmeans2(coords_est.T, C)
        centers = centers.squeeze()

    """
    -------------------
    EVALUATION
    -------------------
    """

    """
    Evaluate performance of mean estimation.
    """
    mean_perf = sim.eval_mean(mean_est)

    """
    Evaluate performance of covariance estimation. We also evaluate the truncated
    eigendecomposition. This is expected to be a closer approximation since it
    imposes an additional low-rank condition on the estimate.
    """
    covar_perf = sim.eval_covar(covar_est)
    eigs_perf = sim.eval_eigs(eigs_est_trunc, lambdas_est_trunc)

    """
    Evaluate clustering performance.
    """
    clustering_accuracy = sim.eval_clustering(vol_idx)

    """
    Assign the cluster centroids to the different images. Since we expect a discrete distribution of volumes
    (and therefore of coordinates), we assign the centroid coordinate to each image that belongs to that cluster.
    Evaluate the coordinates estimated
    """
    clustered_coords_est = centers[vol_idx]
    coords_perf = sim.eval_coords(mean_est, eigs_est_trunc, clustered_coords_est)

    """
    Output estimated covariance spectrum.
    """
    logger.info(f'Population Covariance Spectrum = {np.diag(lambdas_est)}')

    """
    Output performance results.
    """
    logger.info(f'Mean (rel. error) = {mean_perf["rel_err"]}')
    logger.info(f'Mean (correlation) = {mean_perf["corr"]}')
    logger.info(f'Covariance (rel. error) = {covar_perf["rel_err"]}')
    logger.info(f'Covariance (correlation) = {covar_perf["corr"]}')
    logger.info(f'Eigendecomposition (rel. error) = {eigs_perf["rel_err"]}')
    logger.info(f'Clustering (accuracy) = {clustering_accuracy}')
    logger.info(f'Coordinates (mean rel. error) = {coords_perf["rel_err"]}')
    logger.info(f'Coordinates (mean correlation) = {np.mean(coords_perf["corr"])}')
