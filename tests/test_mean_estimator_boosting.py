import numpy as np
import pytest

from aspire.reconstruction import MeanEstimator, WeightedVolumesEstimator
from aspire.source import ArrayImageSource, Simulation
from aspire.utils import Rotation, utest_tolerance
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
    pytest.param(33, marks=pytest.mark.expensive),
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


@pytest.fixture(
    params=VOL_PARAMS, ids=lambda x: f"volume={x[0]}, order={x[1]}", scope="module"
)
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
    )
    src = src.cache()  # precompute images

    return src


@pytest.fixture(scope="module")
def estimated_volume(source):
    estimator = MeanEstimator(source)
    estimated_volume = estimator.estimate()

    return estimated_volume


# Weighted volume fixture. Only tesing C1, C4, and C5.
@pytest.fixture(
    params=VOL_PARAMS[:3], ids=lambda x: f"volume={x[0]}, order={x[1]}", scope="module"
)
def weighted_volume(request, resolution, dtype):
    Volume, order = request.param
    vol_kwargs = dict(
        L=resolution,
        C=2,
        seed=SEED,
        dtype=dtype,
    )
    if order:
        vol_kwargs["order"] = order

    return Volume(**vol_kwargs).generate()


@pytest.fixture(scope="module")
def weighted_source(weighted_volume):
    src = Simulation(
        n=400,
        vols=weighted_volume,
        offsets=0,
        amplitudes=1,
        seed=SEED,
        dtype=weighted_volume.dtype,
    )

    return src


# MeanEstimator Tests.
def test_fsc(source, estimated_volume):
    """Compare estimated volume to source volume with FSC."""
    # Fourier Shell Correlation
    fsc_resolution, fsc = source.vols.fsc(estimated_volume, cutoff=0.5)

    # Check that resolution is less than 2.1 pixels.
    np.testing.assert_array_less(fsc_resolution, 2.1)

    # Check that second to last correlation value is high (>.90).
    np.testing.assert_array_less(0.90, fsc[0, -2])


def test_mse(source, estimated_volume):
    """Check the mean-squared error between source and estimated volumes."""
    mse = np.mean((source.vols.asnumpy() - estimated_volume.asnumpy()) ** 2)
    np.testing.assert_allclose(mse, 0, atol=1e-3)


def test_total_energy(source, estimated_volume):
    """Test that energy is preserved in reconstructed volume."""
    og_total_energy = np.sum(source.vols)
    recon_total_energy = np.sum(estimated_volume)
    np.testing.assert_allclose(og_total_energy, recon_total_energy, rtol=1e-3)


def test_boost_flag(source, estimated_volume):
    """Manually boost a source and reconstruct without boosting."""
    ims = source.projections[:]
    rots = source.rotations
    sym_rots = source.symmetry_group.matrices
    sym_order = len(sym_rots)

    # Manually boosted images and rotations.
    ims_boosted = np.tile(ims, (sym_order, 1, 1))
    rots_boosted = np.zeros((sym_order * source.n, 3, 3), dtype=source.dtype)
    for i, sym_rot in enumerate(sym_rots):
        rots_boosted[i * source.n : (i + 1) * source.n] = sym_rot @ rots
    rots_boosted = Rotation(rots_boosted)

    # Manually boosted source.
    boosted_source = ArrayImageSource(ims_boosted, angles=rots_boosted.angles)

    # Estimate volume with boosting OFF.
    estimator = MeanEstimator(boosted_source, boost=False)
    est_vol = estimator.estimate()

    # Check reconstructions are equal.
    mse = np.mean((estimated_volume.asnumpy() - est_vol.asnumpy()) ** 2)
    np.testing.assert_allclose(mse, 0, atol=utest_tolerance(source.dtype))


# WeightVolumesEstimator Tests.
def test_weighted_volumes(weighted_source):
    """
    Test WeightedVolumeEstimator reconstructs multiple volumes using symmetry boosting.
    """
    src = weighted_source

    # Use source states to assign weights to volumes.
    weights = np.zeros((src.n, src.C), dtype=src.dtype)
    weights[:, 0] = abs(src.states - 1.99)  # sends states [1, 2] to weights [.99, .01]
    weights[:, 1] = 1 - weights[:, 0]  # sets weights for states [1, 2] as [.01, .99]

    # Scale weights
    n0 = np.count_nonzero(src.states == 1)  # number of images from vol[0]
    n1 = np.count_nonzero(src.states == 2)  # number of images from vol[1]
    weights[:, 0] = weights[:, 0] / weights[:, 0].sum() * np.sqrt(n0)
    weights[:, 1] = weights[:, 1] / weights[:, 1].sum() * np.sqrt(n1)

    # Initialize estimator.
    estimator = WeightedVolumesEstimator(src=src, weights=weights)
    est_vols = estimator.estimate()

    # Check FSC (scaling may not be close enough to match mse)
    _, corr = src.vols.fsc(est_vols)
    np.testing.assert_array_less(0.91, corr[:, -2])
