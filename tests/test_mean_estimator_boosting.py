import numpy as np
import pytest

from aspire.basis import FFBBasis3D
from aspire.operators import IdentityFilter
from aspire.reconstruction import MeanEstimator
from aspire.source import Simulation
from aspire.volume import (
    AsymmetricVolume,
    CnSymmetricVolume,
    DnSymmetricVolume,
    OSymmetricVolume,
    TSymmetricVolume,
)

SEED = 23

RESOLUTION = [
    32,
    33,
]

DTYPE = [
    np.float32,
    pytest.param(np.float64, marks=pytest.mark.expensive),
]

# Symmetric volume parameters, (volume_type, symmetric_order).
VOL_PARAMS = [
    (AsymmetricVolume, None),
    (CnSymmetricVolume, 4),
    (CnSymmetricVolume, 5),
    (DnSymmetricVolume, 2),
    pytest.param((TSymmetricVolume, None), marks=pytest.mark.expensive),
    pytest.param((OSymmetricVolume, None), marks=pytest.mark.expensive),
]


# Fixtures.
@pytest.fixture(params=RESOLUTION, ids=lambda x: f"resolution={x}", scope="module")
def resolution(request):
    return request.param


@pytest.fixture(params=DTYPE, ids=lambda x: f"dtype={x}", scope="module")
def dtype(request):
    return request.param


@pytest.fixture(params=VOL_PARAMS, ids=lambda x: f"volume={x[0]}", scope="module")
def volume(request, resolution, dtype):
    Volume, order = request.param
    vol_kwargs = dict(
        L=resolution,
        C=1,
        seed=SEED,
        dtype=dtype,
    )
    if order:
        vol_kwargs["order"] = order

    return Volume(**vol_kwargs).generate()


@pytest.fixture(scope="module")
def source(volume):
    src = Simulation(
        n=200,
        vols=volume,
        offsets=0,
        amplitudes=1,
        seed=SEED,
        dtype=volume.dtype,
        unique_filters=[IdentityFilter()],  # Can remove after PR 1076
    )

    return src


@pytest.fixture(scope="module")
def estimated_volume(source):
    basis = FFBBasis3D(source.L, dtype=source.dtype)
    estimator = MeanEstimator(source, basis)
    estimated_volume = estimator.estimate()

    return estimated_volume


# MeanEstimator Tests.
def test_fsc(source, estimated_volume):
    """Compare estimated volume to source volume with FSC."""
    # Fourier Shell Correlation
    fsc_resolution, fsc = source.vols.fsc(estimated_volume, pixel_size=1, cutoff=0.5)

    # Check that resolution is less than 2.1 pixels.
    np.testing.assert_array_less(fsc_resolution, 2.1)

    # Check that second to last correlation value is high (>.90).
    np.testing.assert_array_less(0.90, fsc[0, -2])


def test_mse(source, estimated_volume):
    """Check the mean-squared error between source and estimated volumes."""
    mse = (
        np.sum((source.vols.asnumpy() - estimated_volume.asnumpy()) ** 2)
        / source.L**3
    )
    np.testing.assert_allclose(mse, 0, atol=1e-3)


def test_total_energy(source, estimated_volume):
    """Test that energy is preserved in reconstructed volume."""
    og_total_energy = np.sum(source.vols)
    recon_total_energy = np.sum(estimated_volume)
    np.testing.assert_allclose(og_total_energy, recon_total_energy, rtol=1e-3)
