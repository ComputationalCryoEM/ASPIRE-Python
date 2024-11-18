import logging
import os.path

import numpy as np
import pytest

from aspire.noise import AnisotropicNoiseEstimator, CustomNoiseAdder
from aspire.operators.filters import FunctionFilter, RadialCTFFilter
from aspire.source import ArrayImageSource
from aspire.source.simulation import Simulation
from aspire.utils import grid_2d, utest_tolerance
from aspire.utils.matrix import anorm

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")

params = [(64, np.float32), (64, np.float64), (63, np.float32), (63, np.float64)]

num_images = 128


def get_sim_object(L, dtype):
    noise_adder = CustomNoiseAdder(
        noise_filter=FunctionFilter(lambda x, y: np.exp(-(x**2 + y**2) / 2))
    )
    sim = Simulation(
        L=L,
        n=num_images,
        unique_filters=[
            RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)
        ],
        noise_adder=noise_adder,
        dtype=dtype,
    )
    return sim


@pytest.mark.parametrize("L, dtype", params)
def testPhaseFlip(L, dtype):
    sim = get_sim_object(L, dtype)
    imgs_org = sim.images[:num_images]
    sim = sim.phase_flip()
    imgs_pf = sim.images[:num_images]

    # check energy conservation
    assert np.allclose(
        anorm(imgs_org.asnumpy(), axes=(1, 2)),
        anorm(imgs_pf.asnumpy(), axes=(1, 2)),
    )

    # dtype of returned images should be the same
    assert dtype == imgs_pf.dtype


def testEmptyPhaseFlip(caplog):
    """
    Attempting phase_flip without CTFFilters should warn.
    """
    # this test doesn't depend on dtype, not parametrized
    # Create a Simulation without any CTFFilters
    sim = Simulation(
        L=8,
        n=num_images,
        dtype=np.float32,
    )
    # assert we log a warning to the user
    with caplog.at_level(logging.WARNING):
        _ = sim.phase_flip()
        assert "No Filters found" in caplog.text


@pytest.mark.parametrize("L, dtype", params)
def testNormBackground(L, dtype):
    sim = get_sim_object(L, dtype)
    bg_radius = 1.0
    grid = grid_2d(sim.L, indexing="yx")
    mask = grid["r"] > bg_radius
    sim = sim.normalize_background()
    imgs_nb = sim.images[:num_images].asnumpy()
    new_mean = np.mean(imgs_nb[:, mask])
    new_variance = np.var(imgs_nb[:, mask])

    # new mean of noise should be close to zero and variance should be close to 1
    assert new_mean < utest_tolerance(dtype) and abs(
        new_variance - 1
    ) < utest_tolerance(dtype)

    # dtype of returned images should be the same
    assert dtype == imgs_nb.dtype


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def testWhiten(dtype):
    # Note this atol holds only for L even. Odd tested in testWhiten2.
    L = 64
    sim = get_sim_object(L, dtype)
    noise_estimator = AnisotropicNoiseEstimator(sim)
    sim = sim.whiten(noise_estimator)
    imgs_wt = sim.images[:num_images].asnumpy()

    # calculate correlation between two neighboring pixels from background
    corr_coef = np.corrcoef(imgs_wt[:, L - 1, L - 1], imgs_wt[:, L - 2, L - 1])

    # correlation matrix should be close to identity
    np.testing.assert_allclose(np.eye(2), corr_coef, atol=1e-1)
    # dtype of returned images should be the same
    assert dtype == imgs_wt.dtype


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def testWhiten2(dtype):
    # Excercises missing cases using odd image resolutions with filter.
    #  Relates to GitHub issue #401.
    # Otherwise this is the similar to testWhiten, though the accuracy
    #  (atol) for odd resolutions seems slightly worse.
    # Note, we also use this test to excercise calling `whiten`
    #  directly with a `Filter`.
    L = 63
    sim = get_sim_object(L, dtype)
    noise_estimator = AnisotropicNoiseEstimator(sim)
    sim = sim.whiten(noise_estimator.filter)
    imgs_wt = sim.images[:num_images].asnumpy()

    corr_coef = np.corrcoef(imgs_wt[:, L - 1, L - 1], imgs_wt[:, L - 2, L - 1])

    # Correlation matrix should be close to identity
    np.testing.assert_allclose(np.eye(2), corr_coef, atol=2e-1)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_whiten_safeguard(dtype):
    """Test that whitening safeguard works as expected."""
    L = 25
    epsilon = 0.02
    sim = get_sim_object(L, dtype)
    noise_estimator = AnisotropicNoiseEstimator(sim)
    sim = sim.whiten(noise_estimator.filter, epsilon=epsilon)

    # Get whitening_filter from generation pipeline.
    whiten_filt = sim.generation_pipeline.xforms[0].filter.evaluate_grid(sim.L)

    # Generate whitening_filter without safeguard directly from noise_estimator.
    filt_vals = noise_estimator.filter.xfer_fn_array
    whiten_filt_unsafe = filt_vals**-0.5

    # Get indices where safeguard should be applied
    # and assert that they are not empty.
    ind = np.where(filt_vals < epsilon)
    np.testing.assert_array_less(0, len(ind[0]))

    # Check that whiten_filt and whiten_filt_unsafe agree up to safeguard indices.
    disagree = np.where(whiten_filt != whiten_filt_unsafe)
    np.testing.assert_array_equal(ind, disagree)

    # Check that whiten_filt is zero at safeguard indices.
    np.testing.assert_allclose(whiten_filt[ind], 0.0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_whiten_safeguard_default(dtype):
    """
    This test catches the case found in the simulated_abinitio_pipeline.py
    of images being zeroed out by to strict of a safeguard on the whiten filter.
    """
    L = 25
    sim = get_sim_object(L, dtype)
    noise_estimator = AnisotropicNoiseEstimator(sim)

    # Alter noise_estimator filter values to be below machine eps.
    # These are values comparable to those in
    # gallery/experiments/simulated_abinitio_pipeline.py.
    mach_eps = np.finfo(dtype).eps
    noise_estimator.filter.xfer_fn_array *= mach_eps
    assert noise_estimator.filter.xfer_fn_array.min() < mach_eps

    # Whiten images
    sim = sim.whiten(noise_estimator.filter)

    # Get whitening_filter from generation pipeline.
    whiten_filt = sim.generation_pipeline.xforms[0].filter.evaluate_grid(sim.L)

    # Check that no values in the whiten filter have been zeroed out by safeguard.
    assert np.count_nonzero(whiten_filt == 0) == 0


@pytest.mark.parametrize("L, dtype", params)
def testInvertContrast(L, dtype):
    sim1 = get_sim_object(L, dtype)
    imgs_org = sim1.images[:num_images]
    sim1 = sim1.invert_contrast()
    imgs1_rc = sim1.images[:num_images]
    # need to set the negative images to the second simulation object
    sim2 = ArrayImageSource(-imgs_org)
    sim2 = sim2.invert_contrast()
    imgs2_rc = sim2.images[:num_images]

    # all images should be the same after inverting contrast
    np.testing.assert_allclose(
        imgs1_rc.asnumpy(), imgs2_rc.asnumpy(), rtol=1e-05, atol=1e-06
    )
    # dtype of returned images should be the same
    assert dtype == imgs1_rc.dtype
    assert dtype == imgs2_rc.dtype
