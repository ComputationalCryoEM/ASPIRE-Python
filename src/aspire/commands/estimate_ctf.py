import logging

import click

logger = logging.getLogger("aspire")

from aspire import ctf

@click.command()
@click.option("--data_folder", default=None, help="Path to mrc or mrcs files")
@click.option("--pixel_size", default=1, type=float, help="Pixel size in A")
@click.option("--cs", default=2.0, type=float, help="Spherical aberration")
@click.option(
    "--amplitude_contrast", default=0.07, type=float, help="Amplitude contrast"
)
@click.option(
    "--voltage", default=300, type=float, help="Voltage in electron microscope"
)
@click.option(
    "--num_tapers",
    default=2,
    type=int,
    help="Number of tapers to apply in PSD estimation",
)
@click.option(
    "--psd_size", default=512, type=int, help="Size of blocks for use in PSD estimation"
)
@click.option(
    "--g_min", default=30, type=float, help="Inverse of minimum resolution for PSD"
)
@click.option(
    "--g_max", default=5, type=float, help="Inverse of maximum resolution for PSD"
)
@click.option("--corr", default=1, type=float, help="Select method")
@click.option("--output_dir", default="results", help="Path to output files")
@click.option(
    "--repro/--no-repro",
    default=False,
    help="Set dtypes and linear programming methods to be more deterministic in exchange for speed.",
)

def estimate_ctf(
    data_folder,
    pixel_size,
    cs,
    amplitude_contrast,
    voltage,
    num_tapers,
    psd_size,
    g_min,
    g_max,
    corr,
    output_dir,
    repro
):
    """
    Given paramaters estimates CTF from experimental data
    and returns CTF as a mrc file.

    This is a Click command line interface wrapper for 
    the aspire.ctf module.
    """
    return ctf.estimate_ctf(
        data_folder,
        pixel_size,
        cs,
        amplitude_contrast,
        voltage,
        num_tapers,
        psd_size,
        g_min,
        g_max,
        corr,
        output_dir,
        repro
    )
