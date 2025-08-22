import logging
import os.path

import numpy as np
import pytest

from aspire.noise import (
    AnisotropicNoiseEstimator,
    CustomNoiseAdder,
    LegacyNoiseEstimator,
)
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


@pytest.mark.parametrize("L, dtype", params)
def test_norm_background_legacy_outofcore(L, dtype):
    """
    This executes normalize_background with the parameters found to reproduce MATLAB's "outofcore" method.
    """
    # Legacy "outofcore" normalize_background defaults to a shifted grid, a different
    # mask radius, disabled ramping, and N - 1 degrees of freedom
    # when computing standard deviation.
    norm_bg_outofcore_flags = {
        "bg_radius": 2 * (L // 2) / L,
        "do_ramp": False,
        "shifted": True,
        "ddof": 1,
    }

    sim = get_sim_object(L, dtype)
    grid = grid_2d(sim.L, shifted=True, indexing="yx", dtype=dtype)
    mask = grid["r"] > norm_bg_outofcore_flags["bg_radius"]
    sim = sim.normalize_background(**norm_bg_outofcore_flags)
    imgs_nb = sim.images[:].asnumpy()
    new_mean = np.mean(imgs_nb[:, mask])
    new_variance = np.var(imgs_nb[:, mask], ddof=1)

    # new mean of noise should be close to zero and variance should be close to 1
    np.testing.assert_array_less(new_mean, utest_tolerance(dtype))
    np.testing.assert_array_less(abs(new_variance - 1), 2e-3)

    # dtype of returned images should be the same
    np.testing.assert_equal(dtype, imgs_nb.dtype)


@pytest.mark.parametrize("L, dtype", params)
def test_legacy_normalize_background(L, dtype):
    """
    This executes legacy_normalize_background.
    """
    # Legacy normalize_background defaults to a shifted grid, a different
    # 0.45 mask radius, disabled ramping, and N - 1 degrees of freedom
    # when computing standard deviation.
    norm_bg_legacy_flags = {
        "bg_radius": 2 * np.floor(L * 0.45) / L,
        "do_ramp": False,
        "shifted": True,
        "ddof": 1,
    }

    sim = get_sim_object(L, dtype)
    grid = grid_2d(sim.L, shifted=True, indexing="yx", dtype=dtype)
    mask = grid["r"] > norm_bg_legacy_flags["bg_radius"]
    sim = sim.legacy_normalize_background()
    imgs_nb = sim.images[:].asnumpy()
    new_mean = np.mean(imgs_nb[:, mask])
    new_variance = np.var(imgs_nb[:, mask], ddof=1)

    # new mean of noise should be close to zero and variance should be close to 1
    np.testing.assert_array_less(new_mean, 3e-4)
    np.testing.assert_array_less(abs(new_variance - 1), 2e-3)

    # dtype of returned images should be the same
    np.testing.assert_equal(dtype, imgs_nb.dtype)


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


@pytest.mark.parametrize(
    "dtype", [np.float32, pytest.param(np.float64, marks=pytest.mark.expensive)]
)
def test_legacy_whiten(dtype):
    """
    Test `legacy_whiten` method.
    """
    L = 64
    sim = get_sim_object(L, dtype)
    sim = sim.legacy_whiten()
    imgs_wt = sim.images[:num_images].asnumpy()

    # calculate correlation between two neighboring pixels from background
    corr_coef = np.corrcoef(imgs_wt[:, L - 1, L - 1], imgs_wt[:, L - 2, L - 1])

    # correlation matrix should be close to identity
    np.testing.assert_allclose(np.eye(2), corr_coef, atol=1e-1)
    # dtype of returned images should be the same
    assert dtype == imgs_wt.dtype


@pytest.mark.parametrize(
    "dtype", [np.float32, pytest.param(np.float64, marks=pytest.mark.expensive)]
)
def test_legacy_whiten_2(dtype):
    """
    Test `legacy_whiten` method with alternate invocation.
    """
    L = 63
    sim = get_sim_object(L, dtype)
    noise_estimator = LegacyNoiseEstimator(sim)
    sim = sim.legacy_whiten(noise_estimator)
    imgs_wt = sim.images[:num_images].asnumpy()

    # calculate correlation between two neighboring pixels from background
    corr_coef = np.corrcoef(imgs_wt[:, L - 1, L - 1], imgs_wt[:, L - 2, L - 1])

    # correlation matrix should be close to identity
    np.testing.assert_allclose(np.eye(2), corr_coef, atol=1e-1)
    # dtype of returned images should be the same
    assert dtype == imgs_wt.dtype


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


@pytest.mark.parametrize("L, dtype", params)
def test_crop(L, dtype):
    """
    Test cropping and document convention via code.
    """

    sim1 = get_sim_object(L, dtype)
    ref_images = sim1.images[:].asnumpy()

    if L % 2:  # L odd
        # Cropping odd by one should remove last row and last col.
        crop_odd_to_even_one = sim1.crop_pad(L - 1)
        np.testing.assert_allclose(
            crop_odd_to_even_one.images[:], ref_images[..., :-1, :-1]
        )

        # Cropping by two should remove first+last row and first+last col.
        crop_odd_to_odd_two = sim1.crop_pad(L - 2)
        np.testing.assert_allclose(
            crop_odd_to_odd_two.images[:], ref_images[..., 1:-1, 1:-1]
        )

        # Cropping by many even should remove equal first+last rows and first+last cols.
        many_even = 32
        k = many_even // 2
        crop_odd_to_odd_many_even = sim1.crop_pad(L - many_even)
        np.testing.assert_allclose(
            crop_odd_to_odd_many_even.images[:], ref_images[..., k:-k, k:-k]
        )

        # Cropping by many odd should remove remove equal first+(last+1) rows and first+(last+1) cols.
        many_odd = 33
        k = many_odd // 2
        crop_odd_to_even_many_odd = sim1.crop_pad(L - many_odd)
        np.testing.assert_allclose(
            crop_odd_to_even_many_odd.images[:], ref_images[..., k : -k - 1, k : -k - 1]
        )
    else:  # L even
        # Cropping even by one should remove first row and first col.
        crop_even_to_odd_one = sim1.crop_pad(L - 1)
        np.testing.assert_allclose(
            crop_even_to_odd_one.images[:], ref_images[..., 1:, 1:]
        )

        # Cropping by two should remove first+last row and first+last col.
        crop_even_to_even_two = sim1.crop_pad(L - 2)
        np.testing.assert_allclose(
            crop_even_to_even_two.images[:], ref_images[..., 1:-1, 1:-1]
        )

        # Cropping by many even should remove equal first+last rows and first+last cols.
        many_even = 32
        k = many_even // 2
        crop_even_to_even_many_even = sim1.crop_pad(L - many_even)
        np.testing.assert_allclose(
            crop_even_to_even_many_even.images[:], ref_images[..., k:-k, k:-k]
        )

        # Cropping by many odd should remove remove equal (first+1)+last) rows and (first+1)+last cols.
        many_odd = 33
        k = many_odd // 2
        crop_even_to_even_many_odd = sim1.crop_pad(L - many_odd)
        np.testing.assert_allclose(
            crop_even_to_even_many_odd.images[:],
            ref_images[..., k + 1 : -k, k + 1 : -k],
        )


@pytest.mark.parametrize("L, dtype", params)
def test_pad(L, dtype):
    """
    Test pad and document convention via code.
    """

    sim1 = get_sim_object(L, dtype)
    ref_images = sim1.images[:].asnumpy()

    if L % 2:  # L odd
        # Padding odd by one should zero pad first row and first col.
        pad_odd_to_even_one = sim1.crop_pad(L + 1)
        # Test image content
        ref = np.pad(ref_images, ((0, 0), (1, 0), (1, 0)))
        np.testing.assert_allclose(pad_odd_to_even_one.images[:], ref)

        # Padding odd by two should zero pad first row and first col.
        pad_odd_to_odd_two = sim1.crop_pad(L + 2)
        # Test image content
        ref = np.pad(ref_images, ((0, 0), (1, 1), (1, 1)))
        np.testing.assert_allclose(pad_odd_to_odd_two.images[:], ref)

        # Padding odd to even by many should zero pad the first+1 and last cols equally.
        many_odd = 33
        pad_odd_to_even_many = sim1.crop_pad(L + many_odd)
        k = many_odd // 2
        # Test image content
        ref = np.pad(ref_images, ((0, 0), (k + 1, k), (k + 1, k)))
        np.testing.assert_allclose(pad_odd_to_even_many.images[:], ref)

        # Padding odd to odd by many should pad the first and last cols equally.
        # This test will also excecise `fill_value`
        many_even = 32
        fill = -1
        pad_odd_to_odd_many = sim1.crop_pad(L + many_even, fill_value=fill)
        k = many_even // 2
        # Test image content
        ref = np.pad(ref_images, ((0, 0), (k, k), (k, k)), constant_values=fill)
        np.testing.assert_allclose(pad_odd_to_odd_many.images[:], ref)
    else:  # L even
        # Padding even by one should zero pad last row and last col.
        pad_even_to_odd_one = sim1.crop_pad(L + 1)
        # Test image content
        ref = np.pad(ref_images, ((0, 0), (0, 1), (0, 1)))
        np.testing.assert_allclose(pad_even_to_odd_one.images[:], ref)

        # Padding even by two should zero pad first row and first col.
        pad_even_to_even_two = sim1.crop_pad(L + 2)
        # Test image content
        ref = np.pad(ref_images, ((0, 0), (1, 1), (1, 1)))
        np.testing.assert_allclose(pad_even_to_even_two.images[:], ref)

        # Padding even to even by many should zero pad the first and last cols equally.
        many_even = 32
        pad_even_to_even_many = sim1.crop_pad(L + many_even)
        k = many_even // 2
        # Test image content
        ref = np.pad(ref_images, ((0, 0), (k, k), (k, k)))
        np.testing.assert_allclose(pad_even_to_even_many.images[:], ref)

        # Padding even to odd by many should pad the first and last+1 cols equally.
        # This test will also excecise `fill_value`
        many_odd = 33
        fill = -1
        pad_even_to_odd_many = sim1.crop_pad(L + many_odd, fill_value=fill)
        k = many_odd // 2
        # Test image content
        ref = np.pad(ref_images, ((0, 0), (k, k + 1), (k, k + 1)), constant_values=fill)
        np.testing.assert_allclose(pad_even_to_odd_many.images[:], ref)
