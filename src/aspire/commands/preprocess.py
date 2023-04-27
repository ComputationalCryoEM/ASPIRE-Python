import logging

import click

from aspire.commands import log_level_option
from aspire.noise import WhiteNoiseEstimator
from aspire.source.relion import RelionSource
from aspire.utils.logging import setConsoleLoggingLevel

logger = logging.getLogger(__name__)


@click.command()
@click.option("--data_folder", default=None, help="Path to data folder")
@click.option(
    "--starfile_in",
    required=True,
    help="Path to input STAR file relative to data folder",
)
@click.option(
    "--starfile_out",
    required=True,
    help="Path to output STAR file relative to data folder",
)
@click.option(
    "--pixel_size", default=1, type=float, help="Pixel size of images in STAR file"
)
@click.option(
    "--max_rows",
    default=None,
    type=int,
    help="Max number of image rows to read from STAR file",
)
@click.option("--flip_phase", default=True, help="Perform phase flip or not")
@click.option(
    "--downsample",
    default=None,
    type=int,
    help="Downsample the images to this resolution prior to saving to starfile/.mrcs stack",
)
@click.option(
    "--normalize_bg",
    default=True,
    help="Normalize the images to have mean zero and variance one in the corners",
)
@click.option(
    "--whiten",
    default=True,
    help="Estimate the noise variance of the images and whiten",
)
@click.option(
    "--invert_contrast",
    default=True,
    help="Invert the contrast of the images to ensure that clean particles have positive intensity",
)
@click.option(
    "--batch_size", default=512, help="Batch size to load images from MRC files"
)
@click.option(
    "--save_mode",
    default="single",
    help="Option to save MRC file, if not single, saved to multiple files in batch size",
)
@click.option(
    "--overwrite",
    default=False,
    help="Whether to overwrite MRC files if they already exist",
)
@log_level_option
def preprocess(
    data_folder,
    starfile_in,
    starfile_out,
    pixel_size,
    max_rows,
    flip_phase,
    downsample,
    normalize_bg,
    whiten,
    invert_contrast,
    batch_size,
    save_mode,
    overwrite,
    loglevel,
):
    """
    Preprocess the raw images and output desired images for future analysis
    """
    # Set desired logging option for the command line
    setConsoleLoggingLevel(loglevel)

    # Create a source object for 2D images
    logger.info(f"Read in images from {starfile_in} and preprocess the images.")
    source = RelionSource(
        starfile_in, data_folder, pixel_size=pixel_size, max_rows=max_rows
    )

    if flip_phase:
        logger.info("Perform phase flip to input images")
        source = source.phase_flip()

    if downsample and downsample < source.L:
        logger.info(f"Downsample resolution to {downsample} X {downsample}")
        source = source.downsample(downsample)

    if normalize_bg:
        logger.info("Normalize images to noise background")
        source = source.normalize_background()

    if whiten:
        logger.info("Whiten noise of images")
        noise_estimator = WhiteNoiseEstimator(source)
        source = source.whiten(noise_estimator)

    if invert_contrast:
        logger.info("Invert global density contrast")
        source = source.invert_contrast()

    source.save(
        starfile_out, batch_size=batch_size, save_mode=save_mode, overwrite=overwrite
    )
