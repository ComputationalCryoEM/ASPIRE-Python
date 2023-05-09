import itertools
import logging
import os.path

import numpy as np
import pytest

from aspire.image import Image
from aspire.noise import (
    AnisotropicNoiseEstimator,
    BlueNoiseAdder,
    CustomNoiseAdder,
    PinkNoiseAdder,
    WhiteNoiseAdder,
    WhiteNoiseEstimator,
)
from aspire.operators import FunctionFilter, ScalarFilter
from aspire.source.simulation import Simulation
from aspire.volume import AsymmetricVolume

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")

logger = logging.getLogger(__name__)

RESOLUTIONS = [64, 65]
DTYPES = [np.float32, np.float64]
# Check one case for each of the above, but other combinations can be checked as part of the `expensive` suite.
VARS = [0.1] + [
    pytest.param(10 ** (-x), marks=pytest.mark.expensive) for x in range(2, 5)
]


def _noise_function(x, y):
    f = np.exp(-(x * x + y * y) / (2 * 0.3**2))
    m = np.mean(f)
    return f / m


def sim_fixture_id(params):
    res = params[0]
    dtype = params[1]
    return f"res={res}, dtype={dtype.__name__}"


@pytest.fixture(params=itertools.product(RESOLUTIONS, DTYPES), ids=sim_fixture_id)
def sim_fixture(request):
    resolution, dtype = request.param
    # Setup a sim with no noise, no ctf, no shifts,
    #   using a compactly supported volume.
    # ie, clean centered projections.
    return Simulation(
        vols=AsymmetricVolume(L=resolution, C=1, dtype=dtype).generate(),
        n=128,
        amplitudes=1,
        offsets=0,
        dtype=dtype,
    )


@pytest.fixture(
    params=[
        WhiteNoiseAdder(var=1),
        CustomNoiseAdder(noise_filter=ScalarFilter(dim=2, value=1)),
    ],
    ids=lambda param: str(param),
)
def adder(request):
    return request.param


def test_white_noise_estimator_clean_corners(sim_fixture):
    """
    Tests that a clean image yields a noise estimate that is virtually zero.
    """
    noise_estimator = WhiteNoiseEstimator(sim_fixture, batchSize=512)
    noise_variance = noise_estimator.estimate()
    # Using a compactly supported volume should yield
    #   virtually no noise in the image corners.
    assert np.isclose(noise_variance, 0, atol=1e-6)
    # 1e-6 was chosen by running the unit test suite 10k (x4 fixture expansion) times.
    #   min=7.544584748608862e-12 max=1.6798412007391812e-07
    #   mean=1.0901885456753326e-09 var=5.27460957578949e-18
    # To reduce the randomness would require a increasing simulation `n`
    # and/or increasing resolution. Both would increase test time.


def test_adder_reprs(adder):
    """Test __repr__ does not crash."""
    logger.info(f"Example repr:\n{repr(adder)}")


def test_adder_strs(adder):
    """Test __str__ does not crash."""
    logger.info(f"Example str:\n{str(adder)}")


@pytest.mark.parametrize(
    "target_noise_variance", VARS, ids=lambda param: f"var={param}"
)
def test_white_noise_adder(sim_fixture, target_noise_variance):
    """
    Test `noise_var` property is set exactly
    and the variance estimated by WhiteNoiseEstimator is within 1%
    for a variety of variances, resolutions and dtypes.
    """

    sim_fixture.noise_adder = WhiteNoiseAdder(var=target_noise_variance)

    # Assert we have passed through the var exactly
    assert sim_fixture.noise_adder.noise_var == target_noise_variance

    # Create an estimator from the source
    noise_estimator = WhiteNoiseEstimator(sim_fixture, batchSize=512)

    # Match estimate within 1%
    assert np.isclose(target_noise_variance, noise_estimator.estimate(), rtol=0.01)


@pytest.mark.parametrize(
    "target_noise_variance", VARS, ids=lambda param: f"var={param}"
)
def test_custom_noise_adder(sim_fixture, target_noise_variance):
    """
    Custom Noise adder uses custom `Filter`.
    Test `noise_var` property is near target_noise_variance used during construction.
    For custom noise adders the default behavior is to estimate `noise_var`
    by generating a sample of the noise.
    """

    custom_filter = FunctionFilter(f=_noise_function) * ScalarFilter(
        value=target_noise_variance
    )

    # Create the CustomNoiseAdder
    sim_fixture.noise_adder = CustomNoiseAdder(noise_filter=custom_filter)

    # Estimate the noise_variance
    estimated_noise_var = sim_fixture.noise_adder.noise_var

    # Check we are achieving an estimate near the target
    logger.debug(f"Estimated Noise Variance {estimated_noise_var}")
    assert np.isclose(estimated_noise_var, target_noise_variance, rtol=0.1)

    # Check sampling yields an estimate near target.
    sample_n = 16
    sample_res = 32
    im_zeros = Image(np.zeros((sample_n, sample_res, sample_res)))
    im_noise_sample = sim_fixture.noise_adder._forward(im_zeros, range(sample_n))
    sampled_noise_var = np.var(im_noise_sample.asnumpy())

    logger.debug(f"Sampled Noise Variance {sampled_noise_var}")
    assert np.isclose(sampled_noise_var, target_noise_variance, rtol=0.1)


@pytest.mark.parametrize(
    "target_noise_variance", VARS, ids=lambda param: f"var={param}"
)
def test_from_snr_white(sim_fixture, target_noise_variance):
    """
    Test that prescribing noise directly by var and  by `from_snr`,
    are close for a variety of paramaters.
    """

    # First add an explicit amount of noise to the base simulation,
    sim_fixture.noise_adder = WhiteNoiseAdder(var=target_noise_variance)
    # and compute the resulting snr of the sim.
    target_snr = sim_fixture.estimate_snr()

    # Compute the `true_snr` of the sim.
    computed_true_snr = sim_fixture.true_snr()
    # Compare the `estimate_snr()` with `true_snr()`
    assert np.isclose(target_snr, computed_true_snr, rtol=0.05)

    # Attempt to create a new simulation at this `target_snr`
    # For unit testing, we will use `sim_fixture`'s volume,
    #   but the new Simulation instance should yield different projections.
    sim_from_snr = Simulation(
        vols=sim_fixture.vols,  # Force the previously generated volume.
        n=sim_fixture.n,
        offsets=0,
        amplitudes=1.0,
        noise_adder=WhiteNoiseAdder.from_snr(target_snr),
    )

    # Check we're within 5% of explicit target
    logger.info(
        "sim_from_snr.noise_adder.noise_var, target_noise_variance ="
        f" {sim_from_snr.noise_adder.noise_var}, {target_noise_variance}"
    )
    assert np.isclose(
        sim_from_snr.noise_adder.noise_var, target_noise_variance, rtol=0.05
    )

    # Compare with WhiteNoiseEstimator consuming sim_from_snr
    noise_estimator = WhiteNoiseEstimator(sim_from_snr, batchSize=512)
    est_noise_variance = noise_estimator.estimate()
    logger.info(
        "est_noise_variance, target_noise_variance ="
        f" {est_noise_variance}, {target_noise_variance}"
    )

    # Check we're within 5%
    assert np.isclose(est_noise_variance, target_noise_variance, rtol=0.05)


@pytest.mark.parametrize(
    "target_noise_variance", VARS, ids=lambda param: f"var={param}"
)
def test_blue_iso_noise_estimation(sim_fixture, target_noise_variance):
    """
    Test that prescribing isotropic blue-ish noise
    is close to target for a variety of paramaters.
    """

    # Create the CustomNoiseAdder
    sim_fixture.noise_adder = BlueNoiseAdder(var=target_noise_variance)

    # TODO, potentially remove or change to Isotropic after #842
    # Compare with AnisotropicNoiseEstimator consuming sim_from_snr
    noise_estimator = AnisotropicNoiseEstimator(sim_fixture, batchSize=512)
    est_noise_variance = noise_estimator.estimate()
    logger.info(
        "est_noise_variance, target_noise_variance ="
        f" {est_noise_variance}, {target_noise_variance}"
    )

    # Check we're within 5%
    assert np.isclose(est_noise_variance, target_noise_variance, rtol=0.05)


@pytest.mark.parametrize(
    "target_noise_variance", VARS, ids=lambda param: f"var={param}"
)
def test_pink_iso_noise_estimation(sim_fixture, target_noise_variance):
    """
    Test that prescribing isotropic pink-ish noise
    is close to target for a variety of paramaters.
    """

    # Create the CustomNoiseAdder
    sim_fixture.noise_adder = PinkNoiseAdder(var=target_noise_variance)

    # TODO, potentially remove or change to Isotropic after #842
    # Compare with AnisotropicNoiseEstimator consuming sim_from_snr
    noise_estimator = AnisotropicNoiseEstimator(sim_fixture, batchSize=512)
    est_noise_variance = noise_estimator.estimate()
    logger.info(
        "est_noise_variance, target_noise_variance ="
        f" {est_noise_variance}, {target_noise_variance}"
    )

    # Check we're within 5%
    assert np.isclose(est_noise_variance, target_noise_variance, rtol=0.05)


@pytest.mark.parametrize(
    "target_noise_variance", VARS, ids=lambda param: f"var={param}"
)
def test_pink_aniso_noise_estimation(sim_fixture, target_noise_variance):
    """
    Test that prescribing anisotropic pink-ish noise
    is close to target for a variety of paramaters.
    """

    # Create the custom noise function and associated Filter
    def aniso_spectrum(x, y):
        s = x[-1] - x[-2]
        f = 4 * s / (np.hypot(x, 2 * y) + s)
        m = np.mean(f)
        return f * target_noise_variance / m

    custom_filter = FunctionFilter(f=aniso_spectrum)

    # Create the CustomNoiseAdder
    sim_fixture.noise_adder = CustomNoiseAdder(noise_filter=custom_filter)

    # TODO, potentially remove after #842
    # Compare with AnisotropicNoiseEstimator consuming sim_from_snr
    noise_estimator = AnisotropicNoiseEstimator(sim_fixture, batchSize=512)
    est_noise_variance = noise_estimator.estimate()
    logger.info(
        "est_noise_variance, target_noise_variance ="
        f" {est_noise_variance}, {target_noise_variance}"
    )

    # Check we're within 5%
    assert np.isclose(est_noise_variance, target_noise_variance, rtol=0.05)
