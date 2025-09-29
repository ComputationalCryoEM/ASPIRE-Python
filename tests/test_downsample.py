import os
import tempfile
from itertools import product

import numpy as np
import pytest

from aspire.downloader import emdb_2660
from aspire.image import Image
from aspire.noise import WhiteNoiseAdder
from aspire.operators import RadialCTFFilter
from aspire.source import RelionSource, Simulation
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
    # Confirm default `pixel_size`
    assert np.allclose(imgs_org.pixel_size, 1.0)
    assert np.allclose(imgs_ds.pixel_size, imgs_org.pixel_size * (L / L_ds))


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
    # Confirm default `pixel_size`
    assert vols_org.pixel_size is None
    assert vols_ds.pixel_size is None


def test_integer_offsets():
    sim = Simulation(offsets=0)
    _ = sim.downsample(3)


# Test that vol.downsample.project == vol.project.downsample.
DTYPES = [np.float32, pytest.param(np.float64, marks=pytest.mark.expensive)]
RES = [65, 66]
RES_DS = [32, 33]
LEGACY = [True, False]

# These parameters are defined in the top level `ImageSource.legacy_downsample` wrapper.
im_ds_legacy_flags = {
    "zero_nyquist": False,
    "centered_fft": False,
}


@pytest.fixture(params=DTYPES, ids=lambda x: f"dtype={x}", scope="module")
def dtype(request):
    return request.param


@pytest.fixture(params=RES, ids=lambda x: f"resolution={x}", scope="module")
def res(request):
    return request.param


@pytest.fixture(params=RES_DS, ids=lambda x: f"resolution_ds={x}", scope="module")
def res_ds(request):
    return request.param


@pytest.fixture(params=LEGACY, ids=lambda x: f"legacy={x}", scope="module")
def legacy(request):
    return request.param


@pytest.fixture(scope="module")
def emdb_vol():
    return emdb_2660()


@pytest.fixture(scope="module")
def volume(emdb_vol, res, dtype):
    vol = emdb_vol.astype(dtype, copy=False)
    vol = vol.downsample(res)
    return vol


def test_downsample_project(volume, res_ds, legacy):
    """
    Test that vol.downsample.project == vol.project.downsample.
    """
    rot = np.eye(3, dtype=volume.dtype)  # project along z-axis
    im_ds_proj = volume.downsample(res_ds, legacy=legacy).project(rot)
    if legacy:
        im_proj_ds = volume.project(rot).downsample(res_ds, **im_ds_legacy_flags)
    else:
        im_proj_ds = volume.project(rot).downsample(res_ds)

    tol = 1e-09
    if volume.dtype == np.float32:
        tol = 1e-07
    if legacy:
        # project does not enforce legacy centering convention,
        # so this property will not hold up to allclose tolerance.
        tol = 1e-03

    np.testing.assert_allclose(im_ds_proj, im_proj_ds, atol=tol)


def test_downsample_legacy(volume, res_ds):
    """
    The legay Matlab downsample method differs from ASPIRE-Python
    downsample in that is uses a different centering convention,
    off by a half pixel for odd images, and does not zero out
    the nyquist frequency. By making these alterations to the
    ASPIRE-Python downsampled images we can match legacy downsample
    upt to `allclose`.
    """
    n_img = 10
    dtype = volume.dtype
    src = Simulation(
        n=n_img,
        vols=volume,
        amplitudes=1,
        dtype=dtype,
        seed=1980,
    )
    ims = src.images[:]

    # Legacy downsampled images.
    #   Params are defined in `ImageSource.legacy_downsample`
    ims_ds_legacy = ims.downsample(res_ds, **im_ds_legacy_flags)

    # ASPIRE-Python downsample with centering adjustments for odd resolution images.
    shifts = 0.5 * np.ones((n_img, 2), dtype=dtype)
    if src.L % 2 == 1:
        ims = ims.shift(shifts)
    ims_ds_py = ims.downsample(res_ds, zero_nyquist=False)
    if res_ds % 2 == 1:
        ims_ds_py = ims_ds_py.shift(-shifts)

    np.testing.assert_allclose(ims_ds_legacy, ims_ds_py, atol=1e-08)


def test_simulation_relion_downsample():
    """
    Test that Simulation.downsample corresponds to RelionSource.downsample
    with respect to images and source attributes.
    """
    # Generate CTF and noise filters.
    defocus_min = 15000  # unit is angstroms
    defocus_max = 25000
    defocus_ct = 7

    ctf_filters = [
        RadialCTFFilter(defocus=d)
        for d in np.linspace(defocus_min, defocus_max, defocus_ct)
    ]

    # Generate Simulation source and downsampled simulation.
    src = Simulation(
        L=64,
        n=10,
        C=1,
        unique_filters=ctf_filters,
        noise_adder=WhiteNoiseAdder.from_snr(snr=1),
        pixel_size=1,
    )
    src_ds = src.downsample(src.L // 2)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save the simulation object into STAR and MRCS files
        starfile = os.path.join(tmpdir, "save_test.star")
        src.save(starfile)

        # Load Simulation source as a RelionSource
        rln_src = RelionSource(starfile)

        # Downsample and test that images and attributes correspond to src_ds
        rln_src_ds = rln_src.downsample(src.L // 2)

        np.testing.assert_allclose(
            src_ds.images[:], rln_src_ds.images[:], atol=utest_tolerance(src.dtype)
        )
        np.testing.assert_allclose(src_ds.offsets, rln_src_ds.offsets)


def test_downsample_offsets(dtype, res):
    """
    Test that `offsets` are appropriately scaled after downsample, that `sim_offsets`
    remain unchanged by downsample, and that centering a downsampled image works properly.
    """
    L = res
    n = 10
    ds_scale = 2

    offsets = np.random.choice([L // 8, -L // 8], size=(n, 2)).astype(dtype, copy=False)
    src = Simulation(
        L=L,
        n=n,
        offsets=offsets,
        seed=1234,
        dtype=dtype,
    )

    src_centered = Simulation(
        L=L,
        n=n,
        offsets=0,
        seed=1234,
        dtype=dtype,
    )

    src_ds = src.downsample(L // ds_scale)
    src_centered_ds = src_centered.downsample(L // ds_scale)
    offset_scale = L / (L // ds_scale)

    # Check `offsets` and `sim_offsets` attributes are as expected, ie.
    # `sim_offsets` are same as original while `offsets` are scaled by downsample.
    np.testing.assert_array_equal(src_ds.sim_offsets, offsets)
    np.testing.assert_array_equal(src_ds.offsets, offsets / offset_scale)

    # Check that centering works for original and downsampled images.
    np.testing.assert_allclose(
        src.images[:].shift(-src.offsets),
        src_centered.images[:],
        atol=utest_tolerance(dtype),
    )

    np.testing.assert_allclose(
        src_ds.images[:].shift(-src_ds.offsets),
        src_centered_ds.images[:],
        atol=utest_tolerance(dtype),
    )


def test_pixel_size():
    """
    Test downsampling is rescaling the `pixel_size` attribute for both Images
    and ImageSource objects.
    """
    # Image sizes in pixels
    L = 8  # original
    dsL = 5  # downsampled
    n = 10  # num images

    # Construct small test ImageSource and Image instances
    src = Simulation(L=L, n=n)
    img = src.images[:]

    # Confirm the pixel size is scaled properly in both cases
    np.testing.assert_approx_equal(
        src.downsample(dsL).pixel_size,
        src.pixel_size * L / dsL,
        err_msg="Incorrect pixel size.",
    )

    np.testing.assert_approx_equal(
        img.downsample(dsL).pixel_size,
        img.pixel_size * L / dsL,
        err_msg="Incorrect pixel size.",
    )
