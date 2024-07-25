import os

import numpy as np
import pytest

from aspire.source import RelionSource, Simulation
from aspire.volume import Volume

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


STARFILE = [
    "rln_proj_65.star",
    "rln_proj_64.star",
    "rln_proj_65_shifted.star",
    "rln_proj_64_shifted.star",
]


@pytest.fixture(params=STARFILE, scope="module")
def sources(request):
    starfile = os.path.join(DATA_DIR, request.param)
    rln_src = RelionSource(starfile)

    # Generate Volume used for Relion projections.
    # Note, `downsample` is a no-op for resolution 65.
    vol_path = os.path.join(DATA_DIR, "clean70SRibosome_vol.npy")
    vol = Volume(np.load(vol_path), dtype=rln_src.dtype).downsample(rln_src.L)

    # Create Simulation source using Volume and angles from Relion projections.
    # Note, for odd resolution Relion projections are shifted by 1 pixel in x and y.
    offsets = rln_src.offsets
    if rln_src.L % 2 == 1:
        offsets -= np.ones((rln_src.n, 2), dtype=rln_src.dtype)

    sim_src = Simulation(
        n=rln_src.n,
        vols=vol,
        offsets=offsets,
        amplitudes=1,
        angles=rln_src.angles,
        dtype=rln_src.dtype,
    )
    return rln_src, sim_src


def test_projections_relative_error(sources):
    """Check the relative error between Relion and ASPIRE projection images."""
    rln_src, sim_src = sources

    # Work with numpy arrays.
    rln_np = rln_src.images[:].asnumpy()
    sim_np = sim_src.images[:].asnumpy()

    # Normalize images.
    rln_np = (rln_np - np.mean(rln_np)) / np.std(rln_np)
    sim_np = (sim_np - np.mean(sim_np)) / np.std(sim_np)

    # Check that relative error is less than 4%.
    error = np.linalg.norm(rln_np - sim_np, axis=(1, 2)) / np.linalg.norm(
        rln_np, axis=(1, 2)
    )
    np.testing.assert_array_less(error, 0.04)


def test_projections_frc(sources):
    """Compute the FRC between Relion and ASPIRE projection images."""
    rln_src, sim_src = sources

    # Compute the Fourier Ring Correlation.
    res, corr = rln_src.images[:].frc(sim_src.images[:], cutoff=0.143)

    # Check that estimated resolution is high (< 2.5 pixels) and correlation is close to 1.
    np.testing.assert_array_less(res, 2.5)
    np.testing.assert_array_less(1 - corr[:, -2], 0.025)
