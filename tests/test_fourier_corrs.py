import logging
import os

import numpy as np
import pytest

from aspire.noise import BlueNoiseAdder
from aspire.numeric import fft
from aspire.source import Simulation
from aspire.utils import Rotation, grid_3d
from aspire.volume import Volume

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


@pytest.fixture(params=DTYPES, ids=lambda x: f"dtype={x}")
def dtype(request):
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
        L=img_size, n=2, vols=v, offsets=0, amplitudes=1, C=1, angles=rots.angles
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

    # Instantiate ASPIRE's Rotation class with a set of angles.
    thetas = [0.12]
    rots = Rotation.about_axis("z", thetas, dtype=dtype)

    vol_rot = vol.rotate(rots)

    # Scale gaussian noise radially
    noise = np.random.normal(loc=0, scale=1, size=vol.shape)
    noise = noise * (1.0 + grid_3d(img_size, normalized=False)["r"]) * 0.33
    vol_noise = Volume(
        np.real(fft.centered_ifftn(fft.centered_fftn(vol.asnumpy()[0]) * (1 + noise)))
    )

    return vol, vol_rot, vol_noise


# FRC


def test_frc_id(image_fixture):
    img, _, _ = image_fixture

    frc_resolution, frc = img.frc(img, pixel_size=1)
    assert np.isclose(frc_resolution[0][0], 2, rtol=0.02)
    assert np.allclose(frc, 1)


def test_frc_rot(image_fixture):
    img_a, img_b, _ = image_fixture

    frc_resolution, frc = img_a.frc(img_b, pixel_size=1)
    assert np.isclose(frc_resolution[0][0], 3.76, rtol=0.01)


def test_frc_noise(image_fixture):
    img_a, _, img_n = image_fixture

    frc_resolution, frc = img_a.frc(img_n, pixel_size=1)
    assert np.isclose(frc_resolution[0][0], 1 / 0.3, rtol=0.3)


# FSC


def test_fsc_id(volume_fixture):
    vol, _, _ = volume_fixture

    fsc_resolution, fsc = vol.fsc(vol, pixel_size=1)
    assert np.isclose(fsc_resolution[0][0], 2.0, rtol=0.02)
    assert np.allclose(fsc, 1)


def test_fsc_rot(volume_fixture):
    vol_a, vol_b, _ = volume_fixture

    fsc_resolution, fsc = vol_a.fsc(vol_b, pixel_size=1)
    assert np.isclose(fsc_resolution[0][0], 3.2, rtol=0.01)


def test_fsc_noise(volume_fixture):
    vol_a, _, vol_n = volume_fixture

    fsc_resolution, fsc = vol_a.fsc(vol_n, pixel_size=1)
    assert np.isclose(fsc_resolution[0][0], 1 / 0.38, rtol=0.3)
