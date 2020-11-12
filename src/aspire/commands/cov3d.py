import logging

import click

from aspire.basis import FBBasis3D
from aspire.covariance import CovarianceEstimator
from aspire.noise import WhiteNoiseEstimator
from aspire.reconstruction import MeanEstimator
from aspire.source.relion import RelionSource

logger = logging.getLogger(__name__)


@click.command()
@click.option("--starfile", required=True, help="Path to starfile")
@click.option(
    "--data_folder", default=None, help="Path to mrcs files referenced in starfile"
)
@click.option(
    "--pixel_size", default=1, type=float, help="Pixel size of images in starfile"
)
@click.option(
    "--max_rows",
    default=None,
    type=int,
    help="Max. number of image rows to read from starfile",
)
@click.option(
    "--max_resolution",
    default=8,
    type=int,
    help="Resolution of downsampled images read from starfile",
)
@click.option("--cg_tol", default=1e-5, help="Tolerance for optimization convergence")
def cov3d(starfile, data_folder, pixel_size, max_rows, max_resolution, cg_tol):
    """Estimate mean volume and covariance from a starfile."""

    source = RelionSource(
        starfile, data_folder=data_folder, pixel_size=pixel_size, max_rows=max_rows
    )

    source.downsample(max_resolution)
    source.cache()

    source.whiten()
    basis = FBBasis3D((max_resolution, max_resolution, max_resolution))
    mean_estimator = MeanEstimator(source, basis, batch_size=8192)
    mean_est = mean_estimator.estimate()

    noise_estimator = WhiteNoiseEstimator(source, batchSize=500)
    # Estimate the noise variance. This is needed for the covariance estimation step below.
    noise_variance = noise_estimator.estimate()
    logger.info(f"Noise Variance = {noise_variance}")

    # Passing in a mean_kernel argument to the following constructor speeds up some calculations
    covar_estimator = CovarianceEstimator(
        source, basis, mean_kernel=mean_estimator.kernel
    )
    covar_estimator.estimate(mean_est, noise_variance, tol=cg_tol)
