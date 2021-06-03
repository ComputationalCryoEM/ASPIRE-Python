import logging

import click

# Overrides click.option with ASPIRE global defaults, see ./__init__.py
import aspire.commands._config  # noqa: F401
from aspire import ctf

logger = logging.getLogger("aspire")


@click.command()
@click.option("--data_folder", default=None, help="Path to mrc or mrcs files")
@click.option(
    "--pixel_size", default=1, type=float, help="Pixel size in \u212b (Angstrom)"
)
@click.option("--cs", default=2.0, type=float, help="Spherical aberration")
@click.option(
    "--amplitude_contrast", default=0.07, type=float, help="Amplitude contrast"
)
@click.option(
    "--voltage",
    default=300,
    type=float,
    help="Electron microscope Voltage in kilovolts (kV)",
)
@click.option(
    "--num_tapers",
    default=2,
    type=int,
    help="Number of tapers to apply in PSD estimation",
)
@click.option(
    "--psd_size",
    default=512,
    type=int,
    help="Size of blocks for use in Power Spectrum estimation",
)
@click.option(
    "--g_min",
    default=30,
    type=float,
    help="Inverse of minimum resolution for Power Spectrum Distribution",
)
@click.option(
    "--g_max", default=5, type=float, help="Inverse of maximum resolution for PSD"
)
@click.option(
    "--dtype",
    default="float32",
    help="NumPy dtype to use in computation.  Example: 'float32' or 'float64'.",
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
    output_dir,
    dtype,
):
    """
    Given parameters estimates CTF from experimental data
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
        output_dir,
        dtype,
    )
