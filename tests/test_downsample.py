from itertools import product

import numpy as np
import pytest

from aspire.image import Image
from aspire.source import Simulation
from aspire.utils import utest_tolerance
from aspire.utils.matrix import anorm
from aspire.utils.misc import gaussian_3d
from aspire.volume import Volume

N = 27
DTYPE = np.float32


def createImages(L, L_ds):
    # generate a 3D Gaussian volume
    sigma = 0.1
    vol = gaussian_3d(L, sigma=(L * sigma,) * 3, dtype=DTYPE)
    # initialize a Simulation object to generate projections of the volume
    sim = Simulation(L, N, vols=Volume(vol), offsets=0.0, amplitudes=1.0, dtype=DTYPE)

    # get images before downsample
    imgs_org = sim.images[:N]

    # get images after downsample
    sim = sim.downsample(L_ds)
    imgs_ds = sim.images[:N]
    return imgs_org, imgs_ds


def createVolumes(L, L_ds):
    # generate a set of volumes with various anisotropic spreads
    sigmas = list(product([L * 0.1, L * 0.2, L * 0.3], repeat=3))
    # get volumes before downsample
    vols_org = Volume(np.array([gaussian_3d(L, sigma=s, dtype=DTYPE) for s in sigmas]))

    # get volumes after downsample
    vols_ds = vols_org.downsample(L_ds)

    return vols_org, vols_ds


def checkCenterPoint(data_org, data_ds):
    # Check that center point is the same after ds
    L = data_org.shape[-1]
    L_ds = data_ds.shape[-1]
    # grab the center point via tuple of length 2 or 3 (image or volume)
    center_org, center_ds = (L // 2, L // 2), (L_ds // 2, L_ds // 2)
    # different tolerances for 2d vs 3d ...
    tolerance = utest_tolerance(DTYPE)
    if isinstance(data_org, Volume):
        center_org += (L // 2,)
        center_ds += (L_ds // 2,)
        # indeterminacy for 3D
        tolerance = 5e-2
    return np.allclose(
        data_org.asnumpy()[(..., *center_org)],
        data_ds.asnumpy()[(..., *center_ds)],
        atol=tolerance,
    )


def checkSignalEnergy(data_org, data_ds):
    # check conservation of energy after downsample
    L = data_org.shape[-1]
    L_ds = data_ds.shape[-1]
    if isinstance(data_org, Image):
        return np.allclose(
            anorm(data_org.asnumpy(), axes=(1, 2)) / L,
            anorm(data_ds.asnumpy(), axes=(1, 2)) / L_ds,
            atol=utest_tolerance(DTYPE),
        )
    elif isinstance(data_org, Volume):
        return np.allclose(
            anorm(data_org.asnumpy(), axes=(1, 2, 3)) / (L ** (3 / 2)),
            anorm(data_ds.asnumpy(), axes=(1, 2, 3)) / (L_ds ** (3 / 2)),
            atol=utest_tolerance(DTYPE),
        )


@pytest.mark.parametrize("L", [65, 66])
@pytest.mark.parametrize("L_ds", [32, 33])
def test_downsample_2d_case(L, L_ds):
    # downsampling from size L to L_ds
    imgs_org, imgs_ds = createImages(L, L_ds)
    # check resolution is correct
    assert (N, L_ds, L_ds) == imgs_ds.shape
    # check center points for all images
    assert checkCenterPoint(imgs_org, imgs_ds)


@pytest.mark.parametrize("L", [65, 66])
@pytest.mark.parametrize("L_ds", [32, 33])
def test_downsample_3d_case(L, L_ds):
    # downsampling from size L to L_ds
    vols_org, vols_ds = createVolumes(L, L_ds)
    # check resolution is correct
    assert (N, L_ds, L_ds, L_ds) == vols_ds.shape
    # check center points for all volumes
    assert checkCenterPoint(vols_org, vols_ds)
    # check signal energy is conserved
    assert checkSignalEnergy(vols_org, vols_ds)


def test_integer_offsets():
    sim = Simulation(offsets=0)
    _ = sim.downsample(3)
