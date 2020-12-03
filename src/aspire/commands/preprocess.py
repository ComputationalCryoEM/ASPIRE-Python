import logging

import click

from aspire.noise import WhiteNoiseEstimator
from aspire.source.relion import RelionSource

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
    "--max_resolution",
    default=16,
    type=int,
    help="Resolution for downsampling images read from STAR file",
)
@click.option(
    "--normalize_background",
    default=True,
    help="Whether to normalize images to background noise",
)
@click.option("--whiten_noise", default=True, help="Whiten background noise")
@click.option(
    "--invert_contrast",
    default=True,
    help="Invert the contrast of images so molecules are shown in white",
)
@click.option(
    "--batch_size", default=512, help="Batch size to load images from MRC files."
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
def preprocess(
    data_folder,
    starfile_in,
    starfile_out,
    pixel_size,
    max_rows,
    flip_phase,
    max_resolution,
    normalize_background,
    whiten_noise,
    invert_contrast,
    batch_size,
    save_mode,
    overwrite,
):
    """
    Preprocess the raw images and output desired images for future analysis
    """
    # Create a source object for 2D images
    logger.info(f"Read in images from {starfile_in} and preprocess the images.")
    source = RelionSource(
        starfile_in, data_folder, pixel_size=pixel_size, max_rows=max_rows
    )

    if flip_phase:
        logger.info("Perform phase flip to input images")
        source.phase_flip()

    if max_resolution < source.L:
        logger.info(f"Downsample resolution to {max_resolution} X {max_resolution}")
        source.downsample(max_resolution)

    if normalize_background:
        logger.info("Normalize images to noise background")
        source.normalize_background()

    if whiten_noise:
        logger.info("Whiten noise of images")
        noise_estimator = WhiteNoiseEstimator(source)
        source.whiten(noise_estimator.filter)

    if invert_contrast:
        logger.info("Invert global density contrast")
        source.invert_contrast()

    source.save(
        starfile_out, batch_size=batch_size, save_mode=save_mode, overwrite=overwrite
    )
