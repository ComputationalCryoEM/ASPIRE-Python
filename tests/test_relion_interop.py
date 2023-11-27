import os

import numpy as np
import pytest

from aspire.source import RelionSource, Simulation
from aspire.volume import Volume

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


STARFILE = ["rln_proj_65.star", "rln_proj_64.star"]


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
    offsets = 0
    if rln_src.L % 2 == 1:
        offsets = -np.ones((rln_src.n, 2), dtype=rln_src.dtype)

    sim_src = Simulation(
        n=rln_src.n,
        vols=vol,
        offsets=offsets,
        amplitudes=1,
        angles=rln_src.angles,
        dtype=rln_src.dtype,
    )
    return rln_src, sim_src


def test_projections(sources):
    rln_src, sim_src = sources

    # Compute the Fourier Ring Correlation.
    res, corr = rln_src.images[:].frc(sim_src.images[:], cutoff=0.143)

    # Check that estimated resolution is high (< 2.5 pixels) and correlation is close to 1.
    np.testing.assert_array_less(res, 2.5)
    np.testing.assert_array_less(1 - corr[:, -2], 0.02)
