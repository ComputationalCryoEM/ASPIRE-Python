import logging
import os
import tempfile

import numpy as np
import pytest

from aspire.image import Image
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
    noisy_src = Simulation(
        L=img_size,
        n=2,
        vols=v,
        offsets=0,
        amplitudes=1,
        C=1,
        angles=rots.angles,
        noise_adder=BlueNoiseAdder.from_snr(2),
        dtype=dtype,
    )
    img, img_rot = noisy_src.clean_images[:]
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

    frc_resolution, frc = img.frc(img, pixel_size=1, cutoff=0.143, method=method)
    assert np.isclose(frc_resolution[0][0], 1, rtol=0.02)
    assert np.allclose(frc, 1)


def test_frc_rot(image_fixture, method):
    img_a, img_b, _ = image_fixture
    assert img_a.dtype == img_b.dtype
    frc_resolution, frc = img_a.frc(img_b, pixel_size=1, cutoff=0.143, method=method)
    assert np.isclose(frc_resolution[0][0], 1.89, rtol=0.1)


def test_frc_noise(image_fixture, method):
    img_a, _, img_n = image_fixture

    frc_resolution, frc = img_a.frc(img_n, pixel_size=1, cutoff=0.143, method=method)
    assert np.isclose(frc_resolution[0][0], 1.75, rtol=0.2)


def test_frc_img_plot(image_fixture):
    """
    Smoke test Image.frc(plot=) passthrough.
    """
    img_a, _, img_n = image_fixture

    # Plot to screen
    with matplotlib_no_gui():
        _ = img_a.frc(img_n, pixel_size=1, cutoff=0.143, plot=True)

    # Plot to file
    with tempfile.TemporaryDirectory() as tmp_input_dir:
        file_path = os.path.join(tmp_input_dir, "img_frc_curve.png")
        img_a.frc(img_n, pixel_size=1, cutoff=0.143, plot=file_path)
        assert os.path.exists(file_path)


def test_frc_plot(image_fixture, method):
    """
    Smoke test the plot.

    Also tests resetting the cutoff.
    """
    img_a, img_b, _ = image_fixture

    frc = FourierRingCorrelation(
        img_a.asnumpy(), img_b.asnumpy(), pixel_size=1, method=method
    )

    with matplotlib_no_gui():
        frc.plot(cutoff=0.5)

    with tempfile.TemporaryDirectory() as tmp_input_dir:
        file_path = os.path.join(tmp_input_dir, "frc_curve.png")
        frc.plot(cutoff=0.143, save_to_file=file_path)


# FSC


def test_fsc_id(volume_fixture, method):
    vol, _ = volume_fixture

    fsc_resolution, fsc = vol.fsc(vol, pixel_size=1, cutoff=0.143, method=method)
    assert np.isclose(fsc_resolution[0][0], 1, rtol=0.02)
    assert np.allclose(fsc, 1)


def test_fsc_trunc(volume_fixture, method):
    vol_a, vol_b = volume_fixture

    fsc_resolution, fsc = vol_a.fsc(vol_b, pixel_size=1, cutoff=0.143, method=method)
    assert fsc_resolution[0][0] > 1.5

    # The follow should correspond to the test_fsc_plot below.
    fsc_resolution, fsc = vol_a.fsc(vol_b, pixel_size=1, cutoff=0.5, method=method)
    assert fsc_resolution[0][0] > 2.0


def test_fsc_vol_plot(volume_fixture):
    """
    Smoke test Image.frc(plot=) passthrough.
    """
    vol_a, vol_b = volume_fixture

    # Plot to screen
    with matplotlib_no_gui():
        _ = vol_a.fsc(vol_b, pixel_size=1, cutoff=0.5, plot=True)

    # Plot to file
    with tempfile.TemporaryDirectory() as tmp_input_dir:
        file_path = os.path.join(tmp_input_dir, "img_fsc_curve.png")
        vol_a.fsc(vol_b, pixel_size=1, cutoff=0.143, plot=file_path)
        assert os.path.exists(file_path)


def test_fsc_plot(volume_fixture, method):
    """
    Smoke test the plot.
    """
    vol_a, vol_b = volume_fixture

    fsc = FourierShellCorrelation(
        vol_a.asnumpy(), vol_b.asnumpy(), pixel_size=1, method=method
    )

    with matplotlib_no_gui():
        fsc.plot(cutoff=0.5)

    with tempfile.TemporaryDirectory() as tmp_input_dir:
        file_path = os.path.join(tmp_input_dir, "fsc_curve.png")
        fsc.plot(cutoff=0.143, save_to_file=file_path)


# Check the error checks.


def test_dtype_mismatch():
    a = np.empty((8, 8), dtype=np.float32)
    b = a.astype(np.float64)

    with pytest.raises(TypeError, match=r"Mismatched input types"):
        _ = FourierRingCorrelation(a, b)


def test_type_mismatch():
    a = np.empty((8, 8), dtype=np.float32)
    b = a.tolist()

    with pytest.raises(TypeError, match=r".*is not a Numpy array"):
        _ = FourierRingCorrelation(a, b)


def test_data_shape_mismatch():
    a = np.empty((8, 8), dtype=np.float32)
    b = np.empty((8, 9), dtype=np.float32)

    with pytest.raises(RuntimeError, match=r".*different data axis shapes"):
        _ = FourierRingCorrelation(a, b)


def test_method_na():
    a = np.empty((8, 8), dtype=np.float32)

    with pytest.raises(
        RuntimeError, match=r"Requested method.*not in available methods"
    ):
        _ = FourierRingCorrelation(a, a, method="man")


def test_cutoff_range():
    a = np.empty((8, 8), dtype=np.float32)

    with pytest.raises(ValueError, match=r"Supplied correlation `cutoff` not in"):
        _ = FourierRingCorrelation(a, a).analyze_correlations(cutoff=2)


def test_2d_stack_plot_raise():
    a = np.random.random((2, 3, 8, 8)).astype(np.float32)

    with pytest.raises(
        RuntimeError, match=r"Unable to plot figure tables with more than 2 dim"
    ):
        FourierRingCorrelation(a, a).plot(cutoff=0.143)


def test_multiple_stack_plot_raise():
    a = np.random.random((3, 8, 8)).astype(np.float32)

    with pytest.raises(
        RuntimeError, match=r"Unable to plot figure tables with more than 1 figure"
    ):
        FourierRingCorrelation(a, a).plot(cutoff=0.143)


def test_img_type_mismatch():
    a = Image(np.empty((8, 8), dtype=np.float32))
    b = a.asnumpy()

    with pytest.raises(TypeError, match=r"`other` image must be an `Image` instance"):
        _ = a.frc(b, pixel_size=1, cutoff=0.143)


def test_vol_type_mismatch():
    a = Volume(np.empty((8, 8, 8), dtype=np.float32))
    b = a.asnumpy()

    with pytest.raises(TypeError, match=r"`other` volume must be an `Volume` instance"):
        _ = a.fsc(b, pixel_size=1, cutoff=0.143)
