import itertools
import logging

import numpy as np
import pytest

from aspire.source.simulation import Simulation
from aspire.utils import Rotation, grid_3d
from aspire.volume import (
    AsymmetricVolume,
    CnSymmetricVolume,
    DnSymmetricVolume,
    LegacyVolume,
    OSymmetricVolume,
    TSymmetricVolume,
)

logger = logging.getLogger(__name__)

# dtype fixture to pass into volume fixture.
DTYPES = [np.float32, pytest.param(np.float64, marks=pytest.mark.expensive)]

# Pixel sized used to test assignment
PXSZ = 3.0


@pytest.fixture(params=DTYPES)
def dtype_fixture(request):
    dtype = request.param
    return dtype


# Parameter combinations for testing SyntheticVolumes with cyclic and dihedral symmetry.
# Each tuple represents (volume class, resolution in pixels, cyclic order)
PARAMS_Cn_Dn = [
    (CnSymmetricVolume, 20, 2),
    (CnSymmetricVolume, 21, 2),
    (CnSymmetricVolume, 30, 3),
    (CnSymmetricVolume, 31, 3),
    (CnSymmetricVolume, 40, 4),
    (CnSymmetricVolume, 41, 4),
    (CnSymmetricVolume, 52, 5),
    (CnSymmetricVolume, 53, 5),
    (CnSymmetricVolume, 64, 6),
    (CnSymmetricVolume, 65, 6),
    (DnSymmetricVolume, 20, 2),
    (DnSymmetricVolume, 21, 2),
    (DnSymmetricVolume, 40, 3),
    (DnSymmetricVolume, 41, 3),
    (DnSymmetricVolume, 42, 4),
    (DnSymmetricVolume, 43, 4),
    (DnSymmetricVolume, 55, 5),
    (DnSymmetricVolume, 56, 5),
    (DnSymmetricVolume, 64, 6),
    (DnSymmetricVolume, 65, 6),
]


# Parameters for tetrahedral, octahedral, asymmetric, and legacy volumes.
# These volumes do not have an `order` parameter.
VOL_CLASSES = [TSymmetricVolume, OSymmetricVolume, AsymmetricVolume, LegacyVolume]
RESOLUTIONS = [20, 21]
PARAMS = list(itertools.product(VOL_CLASSES, RESOLUTIONS))


def vol_fixture_id(params):
    vol_class = params[0]
    res = params[1]
    if len(params) > 2:
        order = params[2]
        return f"{vol_class.__name__}, res={res}, order={order}"
    else:
        return f"{vol_class.__name__}, res={res}"


# Create SyntheticVolume fixture for the set of parameters.
@pytest.fixture(params=PARAMS_Cn_Dn + PARAMS, ids=vol_fixture_id)
def vol_fixture(request, dtype_fixture):
    params = request.param
    vol_class = params[0]
    res = params[1]
    vol_kwargs = dict(
        L=res,
        C=1,
        seed=0,
        dtype=dtype_fixture,
    )
    if len(params) > 2:
        vol_kwargs["order"] = params[2]

    # Assign some volumes a pixel_size, leave others as default.
    if res % 2:
        vol_kwargs["pixel_size"] = PXSZ

    return vol_class(**vol_kwargs)


# SyntheticVolume tests:
def test_volume_repr(vol_fixture):
    """Test Synthetic Volume repr"""
    assert repr(vol_fixture).startswith(f"{vol_fixture.__class__.__name__}")
    logger.debug(f"Volume object: {repr(vol_fixture)}")


def test_volume_generate(vol_fixture):
    """
    Test that a volume is generated
    and stores pixel_size when provided.
    """
    v = vol_fixture.generate()

    # In vol_fixture, we assign pixel_size to volumes having odd voxel sizes.
    if vol_fixture.L % 2:
        np.testing.assert_approx_equal(v.pixel_size, PXSZ)


def test_simulation_init(vol_fixture):
    """Test that a Simulation initializes provided a synthetic Volume."""
    _ = Simulation(L=vol_fixture.L, vols=vol_fixture.generate())


def test_compact_support(vol_fixture):
    """Test that volumes have compact support."""
    if not isinstance(vol_fixture, LegacyVolume):
        # Mask to check support
        g_3d = grid_3d(vol_fixture.L, dtype=vol_fixture.dtype)
        inside = g_3d["r"] < (vol_fixture.L - 1) / vol_fixture.L
        outside = g_3d["r"] > 1
        vol = vol_fixture.generate()

        # Check that volume is zero outside of support and positive inside.
        assert vol.asnumpy()[0][outside].all() == 0
        assert (vol.asnumpy()[0][inside] > 0).all()


# Supress expected warnings due to rotation of symmetric volume.
@pytest.mark.filterwarnings("ignore:`symmetry_group` attribute is being set to `C1`")
def test_volume_symmetry(vol_fixture):
    """Test that volumes have intended symmetry."""
    vol = vol_fixture.generate()

    # Rotations in symmetry group, excluding the Identity.
    rots = vol_fixture.symmetry_group.matrices[1:]

    for rot in rots:
        # Rotate volume by an element of the symmetric group.
        rot_vol = vol.rotate(Rotation(rot), zero_nyquist=False)

        # Check that correlation is close to 1.
        corr = np.dot(rot_vol[0].flatten(), vol[0].flatten()) / np.dot(
            vol[0].flatten(), vol[0].flatten()
        )
        assert abs(corr - 1) < 1.1e-5
