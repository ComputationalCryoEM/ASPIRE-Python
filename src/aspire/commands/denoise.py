import logging

import click

from aspire.basis import FFBBasis2D
from aspire.commands import log_level_option
from aspire.denoising import DenoisedSource, DenoiserCov2D
from aspire.noise import AnisotropicNoiseEstimator, WhiteNoiseEstimator
from aspire.source.relion import RelionSource
from aspire.utils.logging import setConsoleLoggingLevel

logger = logging.getLogger(__name__)


@click.command()
@click.option("--data_folder", default=None, help="Path to data folder")
@click.option(
    "--starfile_in",
    required=True,
    help="Path to input starfile relative to project folder",
)
@click.option(
    "--starfile_out",
    required=True,
    help="Path to output starfile relative to project folder",
)
@click.option(
    "--pixel_size", default=1, type=float, help="Pixel size of images in starfile"
)
@click.option(
    "--max_rows",
    default=None,
    type=int,
    help="Max. no. of image rows to read from starfile",
)
@click.option(
    "--max_resolution",
    default=16,
    type=int,
    help="Resolution of downsampled images read from starfile",
)
@click.option(
    "--noise_type", default="White", type=str, help="Noise type for estimation"
)
@click.option(
    "--denoise_method",
    default="CWF",
    type=str,
    help="Specified method for denoising 2D images",
)
@log_level_option
def denoise(
    data_folder,
    starfile_in,
    starfile_out,
    pixel_size,
    max_rows,
    max_resolution,
    noise_type,
    denoise_method,
    loglevel,
):
    """
    Denoise the images and output the clean images using the default CWF method.
    """
    # Set desired logging option for the command line
    setConsoleLoggingLevel(loglevel)

    # Create a source object for 2D images
    logger.info(f"Read in images from {starfile_in} and preprocess the images.")
    source = RelionSource(
        starfile_in, data_folder, pixel_size=pixel_size, max_rows=max_rows
    )

    logger.info(f"Set the resolution to {max_resolution} X {max_resolution}")
    if max_resolution < source.L:
        # Downsample the images
        source = source.downsample(max_resolution)
    else:
        logger.warn(f"Unable to downsample to {max_resolution}, using {source.L}")
    source = source.cache()

    # Specify the fast FB basis method for expending the 2D images
    basis = FFBBasis2D((source.L, source.L))

    # Estimate the noise of images
    noise_estimator = None
    if noise_type == "White":
        logger.info("Estimate the noise of images using white noise method")
        noise_estimator = WhiteNoiseEstimator(source)
    elif noise_type == "Anisotropic":
        logger.info("Estimate the noise of images using anisotropic method")
        noise_estimator = AnisotropicNoiseEstimator(source)
    else:
        raise RuntimeError(f"Unsupported noise_type={noise_type}")

    # Whiten the noise of images
    logger.info("Whiten the noise of images from the noise estimator")
    source = source.whiten(noise_estimator)

    if denoise_method == "CWF":
        logger.info("Denoise the images using CWF cov2D method.")
        denoiser = DenoiserCov2D(source, basis)
        denoised_src = DenoisedSource(source, denoiser)
        denoised_src.save(
            starfile_out, batch_size=512, save_mode="single", overwrite=False
        )
    else:
        raise NotImplementedError(
            "Currently only covariance Wiener filtering method is supported"
        )
