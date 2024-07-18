import os

import numpy as np
import pytest

from aspire.source import RelionSource, Simulation
from aspire.utils import utest_tolerance
from aspire.volume import Volume

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


STARFILE_ODD = [
    "rln_proj_65_centered.star",
    "rln_proj_65_shifted.star",
]

STARFILE_EVEN = [
    "rln_proj_64_centered.star",
    "rln_proj_64_shifted.star",
]


@pytest.fixture(params=STARFILE_ODD + STARFILE_EVEN, scope="module")
def sources(request):
    """
    Initialize RelionSource from starfile and generate corresponding ASPIRE
    Simulation source.
    """
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


@pytest.fixture(params=[STARFILE_ODD, STARFILE_EVEN], scope="module")
def rln_sources(request):
    """
    Initialize centered and shifted RelionSource's generated using the
    same viewing angles.
    """
    starfile_centered = os.path.join(DATA_DIR, request.param[0])
    starfile_shifted = os.path.join(DATA_DIR, request.param[1])

    rln_src_centered = RelionSource(starfile_centered)
    rln_src_shifted = RelionSource(starfile_shifted)

    return rln_src_centered, rln_src_shifted


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


def test_relion_source_centering(rln_sources):
    """Test that centering by using provided Relion shifts works."""
    rln_src_centered, rln_src_shifted = rln_sources
    ims_centered = rln_src_centered.images[:]
    ims_shifted = rln_src_shifted.images[:]

    offsets = rln_src_shifted.offsets
    np.testing.assert_allclose(
        ims_centered.asnumpy(),
        ims_shifted.shift(-offsets).asnumpy(),
        atol=utest_tolerance(rln_src_centered.dtype),
    )
