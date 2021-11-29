import json
import logging
import os

import click

from aspire.apple.apple import Apple

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--mrc_path",
    help="Path to an mrc file or folder containing all mrcs for particle picking.",
)
@click.option("--create_jpg", is_flag=True, help="save JPG files for picked particles.")
@click.option(
    "--output_dir",
    help="Path to folder to save STAR files. If unspecified, no STAR files are created.",
    default=None,
    type=str,
)
@click.option(
    "--particle_size",
    help="Particle size in pixels.  Many other options have internal defaults based off of particle size.",
    required=True,
    type=int,
)
@click.option("--max_particle_size", help="", default=None, type=int)
@click.option("--min_particle_size", help="", default=None, type=int)
@click.option("--minimum_overlap_amount", help="", default=None, type=int)
@click.option("--query_image_size", help="", default=None)
@click.option("--tau1", help="", default=None, type=int)
@click.option("--tau2", help="", default=None, type=int)
@click.option("--container_size", help="", default=450)
@click.option("--model", help="", default="svm")
@click.option(
    "--model_opts",
    help="Optional JSON dictionary representing specific options corresponding to `model`.",
    default=None,
    type=str,
)
@click.option("--mrc_margin_left", help="", default=99)
@click.option("--mrc_margin_right", help="", default=100)
@click.option("--mrc_margin_top", help="", default=99)
@click.option("--mrc_margin_bottom", help="", default=100)
@click.option("--mrc_shrink_factor", help="", default=2)
@click.option("--mrc_gauss_filter_size", help="", default=15)
@click.option("--mrc_gauss_filter_sigma", help="", default=0.5)
@click.option("--response_thresh_norm_factor", help="", default=20)
@click.option("--conv_map_nthreads", help="", default=4)
@click.option(
    "--n_processes",
    help="Concurrent processes to spawn."
    "May improve performance on very large machines."
    "Otherwise use default.",
    default=1,
)
def apple(
    mrc_path,
    create_jpg,
    output_dir,
    particle_size,
    max_particle_size,
    min_particle_size,
    query_image_size,
    tau1,
    tau2,
    minimum_overlap_amount,
    container_size,
    model,
    model_opts,
    mrc_margin_left,
    mrc_margin_right,
    mrc_margin_top,
    mrc_margin_bottom,
    mrc_shrink_factor,
    mrc_gauss_filter_size,
    mrc_gauss_filter_sigma,
    response_thresh_norm_factor,
    conv_map_nthreads,
    n_processes,
):
    """Pick and save particles from one or more mrc files."""

    # Convert model_opts string to a dictionary
    if model_opts is not None:
        try:
            model_opts = json.loads(model_opts)
        except Exception as e:
            logger.error(
                f"Failed to parse `model_opts`={model_opts}",
                "  Ensure well formed JSON.",
            )
            raise e

    picker = Apple(
        particle_size,
        output_dir,
        min_particle_size,
        max_particle_size,
        query_image_size,
        minimum_overlap_amount,
        tau1,
        tau2,
        container_size,
        model,
        model_opts,
        mrc_margin_left,
        mrc_margin_right,
        mrc_margin_top,
        mrc_margin_bottom,
        mrc_shrink_factor,
        mrc_gauss_filter_size,
        mrc_gauss_filter_sigma,
        response_thresh_norm_factor,
        conv_map_nthreads,
        n_processes,
    )

    if not os.path.exists(mrc_path):
        raise RuntimeError(f"`mrc_path` does not exist: {mrc_path}")
    elif os.path.isdir(mrc_path):
        func = picker.process_folder
    else:
        func = picker.process_micrograph

    func(mrc_path, create_jpg=create_jpg)
