import logging
import os
import tempfile

import numpy as np
import pytest

from aspire.noise import BlueNoiseAdder
from aspire.numeric import fft
from aspire.source import Simulation
from aspire.utils import FourierRingCorrelation, FourierShellCorrelation, Rotation
from aspire.volume import Volume

from .test_utils import matplotlib_no_gui

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")

logger = logging.getLogger(__name__)

IMG_SIZES = [
    64,
    65,
]
DTYPES = [
    np.float64,
    np.float32,
]
METHOD = ["fft", "nufft"]


@pytest.fixture(params=DTYPES, ids=lambda x: f"dtype={x}")
def dtype(request):
    return request.param


@pytest.fixture(params=METHOD, ids=lambda x: f"method={x}")
def method(request):
    return request.param


@pytest.fixture(params=IMG_SIZES, ids=lambda x: f"img_size={x}")
def img_size(request):
    return request.param


@pytest.fixture
def image_fixture(img_size, dtype):
    """
    Serve up images with prescribed parameters.
    """
    # Load sample molecule volume
    v = Volume(
        np.load(os.path.join(DATA_DIR, "clean70SRibosome_vol.npy")), dtype=dtype
    ).downsample(img_size)

    # Instantiate ASPIRE's Rotation class with a set of angles.
    thetas = [0, 0.123]
    rots = Rotation.about_axis("z", thetas, dtype=dtype)

    # Contruct the Simulation source.
    src = Simulation(
        L=img_size,
        n=2,
        vols=v,
        offsets=0,
        amplitudes=1,
        C=1,
        angles=rots.angles,
        dtype=dtype,
    )

    img, img_rot = src.images[:]

    noisy_src = Simulation(
        L=img_size,
        n=2,
        vols=v,
        offsets=0,
        amplitudes=1,
        C=1,
        angles=rots.angles,
        noise_adder=BlueNoiseAdder(var=np.var(img.asnumpy() * 0.5)),
        dtype=dtype,
    )
    img_noisy = noisy_src.images[0]

    return img, img_rot, img_noisy


@pytest.fixture
def volume_fixture(img_size, dtype):
    """
    Serve up volumes with prescribed parameters.
    """
    # Load sample molecule volume
    vol = Volume(
        np.load(os.path.join(DATA_DIR, "clean70SRibosome_vol.npy")), dtype=dtype
    ).downsample(img_size)

    # Invert correlation for some high frequency content
    #   Convert volume to Fourier space.
    vol_trunc_f = fft.centered_fftn(vol.asnumpy()[0])
    #   Get a frequency index.
    trunc_frq = img_size // 3
    #   Negate the power for some frequencies higher than `trunc_frq`.
    vol_trunc_f[-trunc_frq:, :, :] *= -1.0
    vol_trunc_f[:, -trunc_frq:, :] *= -1.0
    vol_trunc_f[:, :, -trunc_frq:] *= -1.0
    #   Convert volume from Fourier space to real space Volume.
    vol_trunc = Volume(fft.centered_ifftn(vol_trunc_f).real)

    return vol, vol_trunc


# FRC


def test_frc_id(image_fixture, method):
    img, _, _ = image_fixture

    frc_resolution, frc = img.frc(img, pixel_size=1, method=method)
    assert np.isclose(frc_resolution[0][0], 1, rtol=0.02)
    assert np.allclose(frc, 1)


def test_frc_rot(image_fixture, method):
    img_a, img_b, _ = image_fixture
    assert img_a.dtype == img_b.dtype
    frc_resolution, frc = img_a.frc(img_b, pixel_size=1, method=method)
    assert np.isclose(frc_resolution[0][0], 3.78 / 2, rtol=0.1)


def test_frc_noise(image_fixture, method):
    img_a, _, img_n = image_fixture

    frc_resolution, frc = img_a.frc(img_n, pixel_size=1, method=method)
    assert np.isclose(frc_resolution[0][0], 3.5 / 2, rtol=0.2)


def test_frc_plot(image_fixture, method):
    """
    Smoke test the plots.

    Also tests resetting the cutoff.
    """
    img_a, img_b, _ = image_fixture

    frc = FourierRingCorrelation(
        img_a.asnumpy(), img_b.asnumpy(), pixel_size=1, method=method, cutoff=0.5
    )

    with matplotlib_no_gui():
        frc.plot()

        with tempfile.TemporaryDirectory() as tmp_input_dir:
            file_path = os.path.join(tmp_input_dir, "frc_curve.png")
            frc.plot(save_to_file=file_path)


# FSC


def test_fsc_id(volume_fixture, method):
    vol, _ = volume_fixture

    fsc_resolution, fsc = vol.fsc(vol, pixel_size=1, method=method)
    assert np.isclose(fsc_resolution[0][0], 1, rtol=0.02)
    assert np.allclose(fsc, 1)


def test_fsc_trunc(volume_fixture, method):
    vol_a, vol_b = volume_fixture

    fsc_resolution, fsc = vol_a.fsc(vol_b, pixel_size=1, method=method)
    assert fsc_resolution[0][0] > 1.5

    # The follow should correspond to the test_fsc_plot below.
    fsc_resolution, fsc = vol_a.fsc(vol_b, pixel_size=1, method=method, cutoff=0.5)
    assert fsc_resolution[0][0] > 2.0


def test_fsc_plot(volume_fixture, method):
    """
    Smoke test the plots.

    Also tests resetting the cutoff.
    """
    vol_a, vol_b = volume_fixture

    fsc = FourierShellCorrelation(
        vol_a.asnumpy(), vol_b.asnumpy(), pixel_size=1, method=method, cutoff=0.5
    )

    with matplotlib_no_gui():
        fsc.plot()

        # Reset cutoff
        fsc.cutoff = 0.143

        with tempfile.TemporaryDirectory() as tmp_input_dir:
            file_path = os.path.join(tmp_input_dir, "fsc_curve.png")
            fsc.plot(save_to_file=file_path)
