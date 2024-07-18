import logging
import os
import tempfile

import numpy as np
import pytest

from aspire.image import Image
from aspire.noise import BlueNoiseAdder
from aspire.numeric import fft
from aspire.source import Simulation
from aspire.utils import (
    FourierRingCorrelation,
    FourierShellCorrelation,
    grid_2d,
    grid_3d,
)
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

    # Contruct the Simulation source to generate a noisy image.
    noisy_src = Simulation(
        L=img_size,
        n=1,
        vols=v,
        offsets=0,
        amplitudes=1,
        C=1,
        noise_adder=BlueNoiseAdder.from_snr(2),
        dtype=dtype,
    )
    img = noisy_src.clean_images[0]
    img_noisy = noisy_src.images[0]

    # Invert correlation for some high frequency content
    #   Convert image to Fourier space.
    img_trunc_f = fft.centered_fftn(img.asnumpy()[0])
    #   Get high frequency indices
    trunc_frq = grid_2d(img_size, normalized=True)["r"] > 1 / 2
    #   Negate the power for high freq content
    img_trunc_f[trunc_frq] *= -1.0
    #   Convert imgume from Fourier space to real space Imgume.
    img_trunc = Image(fft.centered_ifftn(img_trunc_f).real)

    return img, img_trunc, img_noisy


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
    #   Get high frequency indices
    trunc_frq = grid_3d(img_size, normalized=True)["r"] > 1 / 2
    #   Negate the power for high freq content
    vol_trunc_f[trunc_frq] *= -1.0
    #   Convert volume from Fourier space to real space Volume.
    vol_trunc = Volume(fft.centered_ifftn(vol_trunc_f).real)

    return vol, vol_trunc


# FRC


def test_frc_id(image_fixture, method):
    img, _, _ = image_fixture

    frc_resolution, frc = img.frc(img, cutoff=0.143, method=method)
    assert np.isclose(frc_resolution[0], 2, rtol=0.02)
    assert np.allclose(frc, 1, rtol=0.01)


def test_frc_trunc(image_fixture, method):
    img_a, img_b, _ = image_fixture
    assert img_a.dtype == img_b.dtype
    frc_resolution, frc = img_a.frc(img_b, cutoff=0.143, method=method)
    assert frc_resolution[0] > 3.0


def test_frc_noise(image_fixture, method):
    img_a, _, img_n = image_fixture

    frc_resolution, frc = img_a.frc(img_n, cutoff=0.143, method=method)
    assert frc_resolution[0] > 3.5


def test_frc_img_plot(image_fixture):
    """
    Smoke test Image.frc(plot=) passthrough.
    """
    img_a, _, img_n = image_fixture

    # Plot to screen
    with matplotlib_no_gui():
        _ = img_a.frc(img_n, cutoff=0.143, plot=True)

    # Plot to file
    # Also tests `cutoff=None`
    with tempfile.TemporaryDirectory() as tmp_input_dir:
        file_path = os.path.join(tmp_input_dir, "img_frc_curve.png")
        img_a.frc(img_n, cutoff=None, plot=file_path)
        assert os.path.exists(file_path)


def test_frc_plot(image_fixture, method):
    """
    Smoke test the plot.

    Also tests resetting the cutoff.
    """
    img_a, img_b, _ = image_fixture

    frc = FourierRingCorrelation(img_a.asnumpy(), img_b.asnumpy(), method=method)

    with matplotlib_no_gui():
        frc.plot(cutoff=0.5)

    with tempfile.TemporaryDirectory() as tmp_input_dir:
        file_path = os.path.join(tmp_input_dir, "frc_curve.png")
        frc.plot(cutoff=0.143, save_to_file=file_path)


# FSC


def test_fsc_id(volume_fixture, method):
    vol, _ = volume_fixture

    fsc_resolution, fsc = vol.fsc(vol, cutoff=0.143, method=method)
    assert np.isclose(fsc_resolution[0], 2, rtol=0.02)
    assert np.allclose(fsc, 1, rtol=0.01)


def test_fsc_trunc(volume_fixture, method):
    vol_a, vol_b = volume_fixture

    fsc_resolution, fsc = vol_a.fsc(vol_b, cutoff=0.143, method=method)
    assert fsc_resolution[0] > 3.0

    # The follow should correspond to the test_fsc_plot below.
    fsc_resolution, fsc = vol_a.fsc(vol_b, cutoff=0.5, method=method)
    assert fsc_resolution[0] > 3.9


def test_fsc_vol_plot(volume_fixture):
    """
    Smoke test Image.frc(plot=) passthrough.
    """
    vol_a, vol_b = volume_fixture

    # Plot to screen
    with matplotlib_no_gui():
        _ = vol_a.fsc(vol_b, cutoff=0.5, plot=True)

    # Plot to file
    # Also tests `cutoff=None`
    with tempfile.TemporaryDirectory() as tmp_input_dir:
        file_path = os.path.join(tmp_input_dir, "vol_fsc_curve.png")
        vol_a.fsc(vol_b, cutoff=None, plot=file_path)
        assert os.path.exists(file_path)


def test_fsc_plot(volume_fixture, method):
    """
    Smoke test the plot.
    """
    vol_a, vol_b = volume_fixture

    fsc = FourierShellCorrelation(vol_a.asnumpy(), vol_b.asnumpy(), method=method)

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
        RuntimeError, match=r"Unable to plot figure tables with more than 1 reference"
    ):
        FourierRingCorrelation(a, a).plot(cutoff=0.143)


def test_3d_stack_plot_raise():
    a = np.random.random((2, 3, 4, 8, 8)).astype(np.float32)

    with pytest.raises(
        RuntimeError, match=r"Unable to plot figure tables with more than 2 dim"
    ):
        FourierRingCorrelation(a, a).plot(cutoff=0.143)


def test_multiple_stack_plot_raise():
    a = np.random.random((3, 8, 8)).astype(np.float32)
    b = np.reshape(a, (3, 1, 8, 8))

    with pytest.raises(
        RuntimeError, match=r"Unable to plot figure tables with more than 1 reference"
    ):
        FourierRingCorrelation(a, b).plot(cutoff=0.143)


def test_img_type_mismatch():
    a = Image(np.empty((8, 8), dtype=np.float32))
    b = a.asnumpy()

    with pytest.raises(TypeError, match=r"`other` image must be an `Image` instance"):
        _ = a.frc(b, cutoff=0.143)


def test_vol_type_mismatch():
    a = Volume(np.empty((8, 8, 8), dtype=np.float32))
    b = a.asnumpy()

    with pytest.raises(TypeError, match=r"`other` volume must be an `Volume` instance"):
        _ = a.fsc(b, cutoff=0.143)


# Broadcasting


def test_frc_id_bcast(image_fixture, method):
    """
    Test FRC for (1) x (3),  (1) x (1,3) , (1) x (3,1).
    """
    img, _, _ = image_fixture

    k = 3
    img_b = Image(np.tile(img, (3, 1, 1)))

    frc_resolution, frc = img.frc(img_b, cutoff=0.143, method=method)
    assert np.allclose(
        frc_resolution,
        [
            2.0,
        ]
        * k,
        rtol=0.02,
    )
    assert np.allclose(frc, 1.0, rtol=0.01)
    assert frc_resolution.shape == (3,)

    # (1) x (1,3)
    img_b = img_b.stack_reshape(1, 3)

    frc_resolution, frc = img.frc(img_b, cutoff=0.143, method=method)
    assert np.allclose(
        frc_resolution,
        [
            2.0,
        ]
        * k,
        rtol=0.02,
    )
    assert np.allclose(frc, 1.0, rtol=0.01)
    assert frc_resolution.shape == (1, 3)

    # (1) x (3,1)
    img_b = img_b.stack_reshape(3, 1)

    frc_resolution, frc = img.frc(img_b, cutoff=0.143, method=method)
    assert np.allclose(
        frc_resolution,
        [
            2.0,
        ]
        * k,
        rtol=0.02,
    )
    assert np.allclose(frc, 1.0, rtol=0.01)
    assert frc_resolution.shape == (3, 1)


def test_fsc_id_bcast(volume_fixture, method):
    vol, _ = volume_fixture

    k = 3
    vol_b = Volume(np.tile(vol.asnumpy(), (3, 1, 1, 1)))

    fsc_resolution, fsc = vol.fsc(vol_b, cutoff=0.143, method=method)
    assert np.allclose(
        fsc_resolution,
        [
            2.0,
        ]
        * k,
        rtol=0.02,
    )
    assert np.allclose(fsc, 1.0, rtol=0.01)


def test_frc_img_plot_bcast(image_fixture):
    """
    Smoke test Image.frc(plot=) passthrough.
    """
    img_a, img_b, img_n = image_fixture

    img_b = Image(np.vstack((img_a, img_b, img_n)))

    # Plot to screen, one:many
    with matplotlib_no_gui():
        _ = img_a.frc(img_b, cutoff=0.143, plot=True)

    # Plot to file, many elementwise
    with tempfile.TemporaryDirectory() as tmp_input_dir:
        file_path = os.path.join(tmp_input_dir, "img_frc_curve.png")
        img_b.frc(img_b, cutoff=0.143, plot=file_path)
        assert os.path.exists(file_path)


def test_plot_bad_bcast(image_fixture):
    """
    When reference is a stack, we should raise when attempting to plot
    anything other than 1:1 elementwise.
    """
    img_a, img_b, img_n = image_fixture
    img_b = np.vstack((img_a, img_b, img_n))

    # many:many, all pairs for (3,) x (2,1)
    with pytest.raises(RuntimeError, match="Unable to plot figure tables"):
        FourierRingCorrelation(img_b, img_b[:2].reshape(2, 1, *img_b.shape[-2:])).plot(
            cutoff=0.143
        )

    # many:one
    with pytest.raises(RuntimeError, match="Unable to plot figure tables"):
        FourierRingCorrelation(
            img_b,
            img_a.asnumpy(),
        ).plot(cutoff=0.143)


def test_plot_labels(image_fixture):
    """
    When reference is a stack, we should raise when attempting to plot
    anything other than 1:1 elementwise.
    """
    img_a, img_b, img_n = image_fixture
    img_b = np.vstack((img_a, img_b, img_n))

    frc = FourierRingCorrelation(img_a.asnumpy(), img_b)
    with matplotlib_no_gui():
        frc.plot(cutoff=0.143, labels=["abc", "easyas", "123"])

    with pytest.raises(ValueError, match="Check `labels`"):
        frc.plot(cutoff=0.143, labels=["abc", "easyas", "123", "toomany"])

    with pytest.raises(ValueError, match="Check `labels`"):
        frc.plot(cutoff=0.143, labels=["toofew"])
