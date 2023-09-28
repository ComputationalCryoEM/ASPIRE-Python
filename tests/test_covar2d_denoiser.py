import numpy as np
import pytest

from aspire.basis.ffb_2d import FFBBasis2D
from aspire.denoising import DenoisedSource, DenoiserCov2D
from aspire.noise import WhiteNoiseAdder
from aspire.operators.filters import RadialCTFFilter
from aspire.source.simulation import Simulation

# TODO, parameterize these further.
dtype = np.float32
img_size = 64
num_imgs = 1024
noise_var = 0.1848
noise_adder = WhiteNoiseAdder(var=noise_var)
filters = [
    RadialCTFFilter(5, 200, defocus=d, Cs=2.0, alpha=0.1)
    for d in np.linspace(1.5e4, 2.5e4, 7)
]
basis = FFBBasis2D((img_size, img_size), dtype=dtype)


@pytest.fixture(scope="module")
def sim():
    """Create a reusable Simulation source."""
    return Simulation(
        L=img_size,
        n=num_imgs,
        unique_filters=filters,
        offsets=0.0,
        amplitudes=1.0,
        dtype=dtype,
        noise_adder=noise_adder,
    )


def test_batched_rotcov2d_MSE(sim):
    """
    Check calling `DenoiserCov2D` via `DenoiserSource` framework yields acceptable error.
    """
    # need larger numbers of images and higher resolution for good MSE
    imgs_clean = sim.projections[:]

    # Specify the fast FB basis method for expending the 2D images
    denoiser = DenoiserCov2D(sim, basis, noise_var)
    imgs_denoised = denoiser.denoise[:]

    # Calculate the normalized RMSE of the estimated images.
    nrmse_ims = (imgs_denoised - imgs_clean).norm() / imgs_clean.norm()
    np.testing.assert_array_less(nrmse_ims, 0.25)

    # Additionally test the `DenoisedSource` and lazy-eval-cache
    # of the cov2d estimator.
    src = DenoisedSource(sim, denoiser)
    np.testing.assert_allclose(imgs_denoised, src.images[:], rtol=1e-05, atol=1e-08)


def test_source_mismatch(sim):
    """
    Assert mismatched sources raises an error.
    """
    # Create a denoiser.
    denoiser = DenoiserCov2D(sim, basis, noise_var)

    # Create a different source.
    src2 = sim[: sim.n - 1]

    # Raise because src2 not identical to denoiser.src (sim)
    with pytest.raises(NotImplementedError, match=r".*must match.*"):
        _ = DenoisedSource(src2, denoiser)
