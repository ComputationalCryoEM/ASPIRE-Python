import logging

import click

from aspire.abinitio import CLSyncVoting
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
def orient3d(data_folder, starfile_in, starfile_out, pixel_size, max_rows):
    """
    Input images from STAR file and estimate orientational angles
    """
    logger.info(f"Read in images from {starfile_in} and estimate orientational angles.")
    # Create a source object for 2D images
    source = RelionSource(
        starfile_in, data_folder, pixel_size=pixel_size, max_rows=max_rows
    )

    # Estimate rotation matrices
    logger.info("Estimate rotation matrices.")
    orient_est = CLSyncVoting(source)
    orient_est.estimate_rotations()

    # Create new source object and save Estimate rotation matrices
    logger.info("Save Estimate rotation matrices.")
    orient_est_src = orient_est.save_rotations()

    orient_est_src.save_metadata(starfile_out, new_mrcs=False)
