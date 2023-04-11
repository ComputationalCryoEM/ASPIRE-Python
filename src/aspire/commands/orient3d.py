import logging

import click

from aspire.abinitio import CLSyncVoting
from aspire.commands import log_level_option
from aspire.source import OrientedSource, RelionSource
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
@click.option(
    "--n_rad",
    default=None,
    type=int,
    help="Number of points in the radial direction. If None, defaults to half the resolution of the source.",
)
@click.option(
    "--n_theta",
    default=360,
    type=int,
    help="Number of points in the theta direction",
)
@click.option(
    "--max_shift",
    default=0.15,
    type=float,
    help="Maximum range for shifts as a proportion of resolution",
)
@click.option(
    "--shift_step",
    default=1,
    type=int,
    help="Resolution for shift estimation in pixels",
)
@log_level_option
def orient3d(
    data_folder,
    starfile_in,
    starfile_out,
    pixel_size,
    max_rows,
    n_rad,
    n_theta,
    max_shift,
    shift_step,
    loglevel,
):
    """
    Input images from STAR file and estimate orientational angles
    """
    # Set desired logging option for the command line
    setConsoleLoggingLevel(loglevel)

    logger.info(f"Read in images from {starfile_in} and estimate orientational angles.")
    # Create a source object for 2D images
    source = RelionSource(
        starfile_in, data_folder, pixel_size=pixel_size, max_rows=max_rows
    )

    # Estimate rotation matrices
    logger.info("Estimate rotation matrices.")
    orient_est = CLSyncVoting(
        source, n_rad=n_rad, n_theta=n_theta, max_shift=max_shift, shift_step=shift_step
    )

    # Create new source object and save Estimate rotation matrices
    logger.info("Save Estimate rotation matrices.")

    orient_est_src = OrientedSource(
        source,
        orient_est,
    )

    orient_est_src.save_metadata(starfile_out)
