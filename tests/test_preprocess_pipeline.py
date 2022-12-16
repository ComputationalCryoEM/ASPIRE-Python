import logging
import os.path

import numpy as np
import pytest

from aspire.noise import AnisotropicNoiseEstimator
from aspire.operators.filters import FunctionFilter, RadialCTFFilter
from aspire.source import ArrayImageSource
from aspire.source.simulation import Simulation
from aspire.utils import grid_2d, utest_tolerance
from aspire.utils.matrix import anorm

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")

params = [(64, np.float32), (64, np.float64), (63, np.float32), (63, np.float64)]

num_images = 128


def get_sim_object(L, dtype):
    noise_filter = FunctionFilter(lambda x, y: np.exp(-(x**2 + y**2) / 2))
    sim = Simulation(
        L=L,
        n=num_images,
        unique_filters=[
            RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)
        ],
        noise_filter=noise_filter,
        dtype=dtype,
    )
    return sim


@pytest.mark.parametrize("L, dtype", params)
def testPhaseFlip(L, dtype):
    sim = get_sim_object(L, dtype)
    imgs_org = sim.images[:num_images]
    sim.phase_flip()
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
        sim.phase_flip()
        assert "No Filters found" in caplog.text


@pytest.mark.parametrize("L, dtype", params)
def testNormBackground(L, dtype):
    sim = get_sim_object(L, dtype)
    bg_radius = 1.0
    grid = grid_2d(sim.L, indexing="yx")
    mask = grid["r"] > bg_radius
    sim.normalize_background()
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
    sim.whiten(noise_estimator.filter)
    imgs_wt = sim.images[:num_images].asnumpy()

    # calculate correlation between two neighboring pixels from background
    corr_coef = np.corrcoef(imgs_wt[:, L - 1, L - 1], imgs_wt[:, L - 2, L - 1])

    # correlation matrix should be close to identity
    assert np.allclose(np.eye(2), corr_coef, atol=1e-1)
    # dtype of returned images should be the same
    assert dtype == imgs_wt.dtype


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def testWhiten2(dtype):
    # Excercises missing cases using odd image resolutions with filter.
    #  Relates to GitHub issue #401.
    # Otherwise this is the same as testWhiten, though the accuracy
    #  (atol) for odd resolutions seems slightly worse.
    L = 63
    sim = get_sim_object(L, dtype)
    noise_estimator = AnisotropicNoiseEstimator(sim)
    sim.whiten(noise_estimator.filter)
    imgs_wt = sim.images[:num_images].asnumpy()

    corr_coef = np.corrcoef(imgs_wt[:, L - 1, L - 1], imgs_wt[:, L - 2, L - 1])

    # Correlation matrix should be close to identity
    assert np.allclose(np.eye(2), corr_coef, atol=2e-1)


@pytest.mark.parametrize("L, dtype", params)
def testInvertContrast(L, dtype):
    sim1 = get_sim_object(L, dtype)
    imgs_org = sim1.images[:num_images]
    sim1.invert_contrast()
    imgs1_rc = sim1.images[:num_images]
    # need to set the negative images to the second simulation object
    sim2 = ArrayImageSource(-imgs_org)
    sim2.invert_contrast()
    imgs2_rc = sim2.images[:num_images]

    # all images should be the same after inverting contrast
    assert np.allclose(imgs1_rc.asnumpy(), imgs2_rc.asnumpy())
    # dtype of returned images should be the same
    assert dtype == imgs1_rc.dtype
    assert dtype == imgs2_rc.dtype
