import logging
import os
from unittest import TestCase

import numpy as np
import pytest

from aspire.basis import FFBBasis2D
from aspire.classification import (
    AligningAverager2D,
    Averager2D,
    BFRAverager2D,
    BFSRAverager2D,
    BFSReddyChatterjiAverager2D,
    ReddyChatterjiAverager2D,
)
from aspire.operators import PolarFT
from aspire.source import Simulation
from aspire.utils import Rotation
from aspire.volume import Volume

logger = logging.getLogger(__name__)


DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


def check_angle_diff(est, ref, tol):
    """
    Helper function to check if a set of estimated rotation angles are within
    a tolerance of a set of reference angles.

    :return: boolean
    """
    all_within_tol = np.all(abs(est % (2 * np.pi) - ref % (2 * np.pi)) <= tol)
    return all_within_tol


class Averager2DBase:
    """
    Configure and setup a unit test case bypassing pytest execution.

    Base class will become inherited into concrete TestCase.
    """

    # Subclasses should override `averager` with a different class.
    averager = Averager2D

    def setUp(self):
        self.vols = Volume(
            np.load(os.path.join(DATA_DIR, "clean70SRibosome_vol.npy"))
        ).downsample(64)

        self.resolution = self.vols.resolution
        self.n_img = 3
        self.dtype = np.float64

        # Create a Basis to use in averager.
        self.basis = FFBBasis2D((self.resolution, self.resolution), dtype=self.dtype)

        # This sets up a trivial class, where there is one group having all images.
        self.classes = np.arange(self.n_img, dtype=int).reshape(1, self.n_img)
        self.reflections = np.zeros(self.classes.shape, dtype=bool)

    # This is a workaround to use a `pytest` fixture with `unittest` style cases.
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        self._caplog = caplog

    def tearDown(self):
        pass

    def _getSrc(self):
        # Base Averager2D does not require anything from source.
        # Subclasses implement specific src
        return None

    def testTypeMismatch(self):
        # Work around ABC, which won't let us test the unimplemented base case.
        self.averager.__abstractmethods__ = set()

        # Intentionally mismatch Basis and Averager dtypes
        if self.dtype == np.float32:
            test_dtype = np.float64
        else:
            test_dtype = np.float32

        with self._caplog.at_level(logging.WARN):
            self.averager(self.basis, self._getSrc(), dtype=test_dtype)
            assert "does not match dtype" in self._caplog.text

    def _construct_rotations(self):
        """
        Constructs a `Rotation` object which can yield `angles` as used by `Source`s.
        """

        # Get a list of angles to test
        self.thetas, self.step = np.linspace(
            0, 2 * np.pi, num=self.n_img, endpoint=False, retstep=True, dtype=self.dtype
        )

        # Generate rotations to be used by `Simulation`. Since `Simulation` rotates
        # the coordinate grid and the averager aligns by rotating the projection images,
        # we negate the angles fed into `Simulation` for direct comparison later.
        self.rotations = Rotation.about_axis(
            "z", -self.thetas, dtype=self.dtype, gimble_lock_warnings=False
        )


class Averager2DTestCase(Averager2DBase, TestCase):
    """
    Concrete TestCase
    """


class AligningAverager2DBase(Averager2DBase):
    """
    Configure and setup a unit test case bypassing pytest execution.

    Base class will become inherited into concrete TestCase.

    Aligning Averagers are expected to expose

    `.rotations`
    `.shifts`
    `.correlations`
    """

    averager = AligningAverager2D

    def setUp(self):
        super().setUp()

        # We'll construct our Rotations now
        self._construct_rotations()

        # Create a `src` to feed our tests
        self.src = self._getSrc()

        # Get the image coef
        self.coefs = self.basis.evaluate_t(self.src.images[: self.n_img])

    def _call_averager(self):
        # Construct the Averager
        avgr = self.averager(self.basis, self._getSrc())
        # Call the `align` method
        _ = avgr.align(self.classes, self.reflections, self.coefs)
        _ = avgr.average(self.classes, self.reflections, self.coefs)
        return avgr

    def test_attributes(self):
        avgr = self._call_averager()

        self.assertTrue(hasattr(avgr, "rotations"))

        self.assertTrue(hasattr(avgr, "shifts"))

        self.assertTrue(hasattr(avgr, "correlations"))

    def _getSrc(self):
        if not hasattr(self, "shifts"):
            self.shifts = np.zeros((self.n_img, 2))

        return Simulation(
            vols=self.vols,
            L=self.resolution,
            n=self.n_img,
            C=1,
            angles=self.rotations.angles,
            offsets=self.shifts,
            amplitudes=np.ones(self.n_img),
            seed=12345,
            dtype=self.dtype,
        )


class BFRAverager2DTestCase(AligningAverager2DBase, TestCase):
    averager = BFRAverager2D
    n_search_angles = 360

    def testNoRot(self):
        """
        Test we raise an error when our basis does not provide `rotate` method.
        """
        # DiracBasis does not provide `rotate`,
        basis = PolarFT((self.resolution, self.resolution), dtype=self.dtype)

        # and that should raise an error during instantiation.
        with pytest.raises(RuntimeError, match=r".* must provide a `rotate` method."):
            _ = self.averager(basis, self._getSrc())

    def testAverager(self):
        """
        Construct a stack of images with known rotations.

        Rotationally averager the stack and compare output with known rotations.
        """

        # Construct the Averager and then call the `align` method
        avgr = self.averager(
            self.basis,
            self._getSrc(),
            n_angles=self.n_search_angles,
        )
        _rotations, _shifts, _ = avgr.align(self.classes, self.reflections, self.coefs)

        self.assertIsNone(_shifts)
        # Crude check that we are closer to known angle than the next rotation
        self.assertTrue(check_angle_diff(_rotations, self.thetas, self.step / 2))

        # Fine check that we are within n_angles.
        self.assertTrue(
            check_angle_diff(_rotations, self.thetas, 2 * np.pi / self.n_search_angles)
        )


class BFSRAverager2DTestCase(BFRAverager2DTestCase):
    averager = BFSRAverager2D

    def setUp(self):
        # Inherit basic params from the base class
        super(BFRAverager2DTestCase, self).setUp()

        # Setup shifts, don't shift the base image
        self.shifts = np.zeros((self.n_img, 2))
        self.shifts[1:, 0] = 1
        self.shifts[1:, 1] = 2

        # Execute the remaining setup from BFRAverager2DTestCase
        super().setUp()

    def testAverager(self):
        """
        Construct a stack of images with known rotations.

        Rotationally averager the stack and compare output with known rotations.
        """

        # Construct the Averager and then call the main `align` method
        avgr = self.averager(
            self.basis,
            self._getSrc(),
            n_angles=self.n_search_angles,
            radius=3,
        )
        _rotations, _shifts, _ = avgr.align(self.classes, self.reflections, self.coefs)

        # Crude check that we are closer to known angle than the next rotation
        self.assertTrue(check_angle_diff(_rotations, self.thetas, self.step / 2))

        # Fine check that we are within n_angles.
        self.assertTrue(
            check_angle_diff(_rotations, self.thetas, 2 * np.pi / self.n_search_angles)
        )

        # Check that we are _not_ shifting the base image
        self.assertTrue(np.all(_shifts[0][0] == 0))
        # Check that we produced estimated shifts away from origin
        #  Note that Simulation's rot+shift is generally not equal to shift+rot.
        #  Instead we check that some combination of
        #  non zero shift+rot improved corr.
        #  Perhaps in the future should check more details.
        self.assertTrue(np.all(np.hypot(*_shifts[0][1:].T) >= 1))


class ReddyChatterjiAverager2DTestCase(BFSRAverager2DTestCase):
    averager = ReddyChatterjiAverager2D

    def testAverager(self):
        """
        Construct a stack of images with known rotations.

        Rotationally averager the stack and compare output with known rotations.
        """

        # Construct the Averager and then call the main `align` method
        avgr = self.averager(
            composite_basis=self.basis,
            src=self._getSrc(),
            dtype=self.dtype,
        )
        _rotations, _shifts, _ = avgr.align(self.classes, self.reflections, self.coefs)

        # Crude check that we are closer to known angle than the next rotation
        self.assertTrue(check_angle_diff(_rotations, self.thetas, self.step / 2))

        # Fine check that we are within 4 degrees.
        self.assertTrue(check_angle_diff(_rotations, self.thetas, np.pi / 45))

        # Check that we are _not_ shifting the base image
        self.assertTrue(np.all(_shifts[0][0] == 0))
        # Check that we produced estimated shifts away from origin
        #  Note that Simulation's rot+shift is generally not equal to shift+rot.
        #  Instead we check that some combination of
        #  non zero shift+rot improved corr.
        #  Perhaps in the future should check more details.
        self.assertTrue(np.all(np.hypot(*_shifts[0][1:].T) >= 1))


class BFSReddyChatterjiAverager2DTestCase(ReddyChatterjiAverager2DTestCase):
    averager = BFSReddyChatterjiAverager2D
