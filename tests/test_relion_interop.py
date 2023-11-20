import logging
import os

import numpy as np
import pytest

from aspire.source import RelionSource, Simulation
from aspire.volume import Volume

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


def test_projections():
    # Create RelionSource from Relion generated projection images.
    starfile = os.path.join(DATA_DIR, "rln_proj.star")
    rln_src = RelionSource(starfile)

    # Create Simulation source using same volume and angles.
    # Note, Relion projections are shifted by 1 pixel cmopared to ASPIRE.
    dtype = rln_src.dtype
    vol_path = os.path.join(DATA_DIR, "clean70SRibosome_vol.npy")
    vol = Volume(np.load(vol_path), dtype=dtype)
    sim_src = Simulation(
        n=rln_src.n,
        vols=vol,
        offsets=-np.ones((rln_src.n, 2), dtype=dtype),
        angles=rln_src.angles,
        dtype=dtype,
    )

    # Compute the Fourier Ring Correlation.
    res, corr = rln_src.images[:].frc(sim_src.images[:], cutoff=0.143)

    # Check that res is small and corr is close to 1.
    np.testing.assert_array_less(res, 2.5)
    np.testing.assert_array_less(1 - corr[:, -2], 0.0015)
