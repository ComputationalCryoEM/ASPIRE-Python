import numpy as np
import pytest

from aspire.basis import FBBasis2D, FFBBasis2D, FLEBasis2D, FPSWFBasis2D, PSWFBasis2D
from aspire.denoising import DenoisedSource, DenoiserCov2D
from aspire.noise import WhiteNoiseAdder
from aspire.operators import IdentityFilter, RadialCTFFilter
from aspire.source import Simulation

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
BASIS = [
    pytest.param(FBBasis2D, marks=pytest.mark.expensive),
    FFBBasis2D,
    FLEBasis2D,
    pytest.param(PSWFBasis2D, marks=pytest.mark.expensive),
    FPSWFBasis2D,
]


@pytest.fixture(params=BASIS, scope="module", ids=lambda x: f"basis={x}")
def basis(request):
    """
    Construct and return a 2D Basis.
    """
    cls = request.param
    return cls(img_size, dtype=dtype)


@pytest.fixture(scope="module")
def sim():
    """Create a reusable Simulation source."""
    sim = Simulation(
        L=img_size,
        n=num_imgs,
        unique_filters=filters,
        offsets=0.0,
        amplitudes=1.0,
        dtype=dtype,
        noise_adder=noise_adder,
    )
    sim = sim.cache()
    return sim


@pytest.fixture(scope="module")
def coef(sim, basis):
    """Generate small set of reference coefficients."""
    return basis.expand(sim.images[:3])


def test_batched_rotcov2d_MSE(sim, basis):
    """
    Check calling `DenoiserCov2D` via `DenoiserSource` framework yields acceptable error.
    """
    # Smoke test reference values (chosen by experimentation).
    refs = {
        "FBBasis2D": 0.25,
        "FFBBasis2D": 0.25,
        "PSWFBasis2D": 0.75,
        "FPSWFBasis2D": 0.75,
        "FLEBasis2D": 0.32,
    }

    # need larger numbers of images and higher resolution for good MSE
    imgs_clean = sim.projections[:]

    # Specify the fast FB basis method for expending the 2D images
    denoiser = DenoiserCov2D(sim, basis, noise_var)
    imgs_denoised = denoiser.denoise[:]

    # Calculate the normalized RMSE of the estimated images.
    nrmse_ims = (imgs_denoised - imgs_clean).norm() / imgs_clean.norm()
    ref = refs[basis.__class__.__name__]
    np.testing.assert_array_less(
        nrmse_ims,
        ref,
        err_msg=f"Comparison failed for {basis}. Achieved: {nrmse_ims} expected: {ref}.",
    )

    # Additionally test the `DenoisedSource` and lazy-eval-cache
    # of the cov2d estimator.
    src = DenoisedSource(sim, denoiser)
    np.testing.assert_allclose(imgs_denoised, src.images[:], rtol=1e-05, atol=1e-08)


def test_source_mismatch(sim, basis):
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


def test_filter_to_basis_mat_id(coef, basis):
    """
    Test `basis.filter_to_basis_mat` operator performance against
    manual sequence of evaluate->filter->expand for `IdentifyFilter`.
    """

    refs = {
        "FBBasis2D": 0.025,
        "FFBBasis2D": 8e-7,
        "PSWFBasis2D": 0.12,
        "FPSWFBasis2D": 0.12,
        "FLEBasis2D": 4e-7,
    }

    # IdentityFilter should produce id
    filt = IdentityFilter()

    # Apply the basis filter operator.
    # Note transpose because `apply` expects and returns column vectors.
    # coef_ftbm = basis.filter_to_basis_mat(filt).apply(coef.asnumpy().T).T
    coef_ftbm = (basis.filter_to_basis_mat(filt) @ coef.asnumpy().T).T

    # Apply evaluate->filter->expand manually
    imgs = coef.evaluate()
    imgs_manual = imgs.filter(filt)
    coef_manual = basis.expand(imgs_manual)

    # Compare coefs from using ftbm operator with coef from eval->filter->exp
    rms = np.sqrt(np.mean(np.square(coef_ftbm - coef_manual)))
    ref = refs[basis.__class__.__name__]
    np.testing.assert_array_less(
        rms,
        ref,
        err_msg=f"Comparison failed for {basis}. Achieved: {rms} expected: {ref}",
    )


def test_filter_to_basis_mat_ctf(coef, basis):
    """
    Test `basis.filter_to_basis_mat` operator performance against
    manual sequence of evaluate->filter->expand for `RadialCTFFilter`.
    """

    refs = {
        "FBBasis2D": 0.11,
        "FFBBasis2D": 0.36,
        "PSWFBasis2D": 0.07,
        "FPSWFBasis2D": 0.07,
        "FLEBasis2D": 0.4,
    }

    # Create a RadialCTFFilter
    filt = RadialCTFFilter(pixel_size=1)

    # Apply the basis filter operator.
    # Note transpose because `apply` expects and returns column vectors.
    # coef_ftbm = basis.filter_to_basis_mat(filt).apply(coef.asnumpy().T).T
    coef_ftbm = (basis.filter_to_basis_mat(filt) @ coef.asnumpy().T).T

    # Apply evaluate->filter->expand manually
    imgs = coef.evaluate()
    imgs_manual = imgs.filter(filt)
    coef_manual = basis.expand(imgs_manual)

    # Compare coefs from using ftbm operator with coef from eval->filter->exp
    rms = np.sqrt(np.mean(np.square(coef_ftbm - coef_manual)))
    ref = refs[basis.__class__.__name__]
    np.testing.assert_array_less(
        rms,
        ref,
        err_msg=f"Comparison failed for {basis}. Achieved: {rms} expected: {ref}",
    )
