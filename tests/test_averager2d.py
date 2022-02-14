import logging
import os
from unittest import TestCase

import numpy as np
import pytest

from aspire.basis import DiracBasis, FFBBasis2D
from aspire.classification import (
    Averager2D,
    BFRAverager2D,
    BFSRAverager2D,
    BFSReddyChatterjiAverager2D,
    ReddyChatterjiAverager2D,
)
from aspire.source import Simulation
from aspire.utils import Rotation
from aspire.volume import Volume

logger = logging.getLogger(__name__)


DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


# Ignore Gimbal lock warning for our in plane rotations.
@pytest.mark.filterwarnings("ignore:Gimbal lock detected")
class Averager2DTestCase(TestCase):
    # Subclasses should override `averager` with a different class.
    averager = Averager2D

    def setUp(self):

        self.vols = Volume(
            np.load(os.path.join(DATA_DIR, "clean70SRibosome_vol.npy"))
        ).downsample(17)

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

        # Common 3D rotation matrix, about z.
        def r(theta):
            return np.array(
                [
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1],
                ],
                dtype=self.dtype,
            )

        # Construct a sequence of rotation matrices using thetas
        _rots = np.empty((self.n_img, 3, 3), dtype=self.dtype)
        for n, theta in enumerate(self.thetas):
            # Note we negate theta to match Rotation convention.
            _rots[n] = r(-theta)

        # Use our Rotation class (maybe it should be able to do this one day?)
        self.rots = Rotation.from_matrix(_rots)


@pytest.mark.filterwarnings("ignore:Gimbal lock detected")
class BFRAverager2DTestCase(Averager2DTestCase):

    averager = BFRAverager2D

    def setUp(self):

        self.n_search_angles = 360

        super().setUp()

        # We'll construct our Rotations now
        self._construct_rotations()

        # Create a `src` to feed our tests
        self.src = self._getSrc()

        # Get the image coef
        self.coefs = self.basis.evaluate_t(self.src.images(0, self.n_img))

    def _getSrc(self):
        if not hasattr(self, "shifts"):
            self.shifts = np.zeros((self.n_img, 2))

        return Simulation(
            vols=self.vols,
            L=self.resolution,
            n=self.n_img,
            C=1,
            angles=self.rots.angles,
            offsets=self.shifts,
            amplitudes=np.ones(self.n_img),
            seed=12345,
            dtype=self.dtype,
        )

    def testNoRot(self):
        """
        Test we raise an error when our basis does not provide `rotate` method.
        """
        # DiracBasis does not provide `rotate`,
        basis = DiracBasis((self.resolution, self.resolution), dtype=self.dtype)

        # and that should raise an error during instantiation.
        with pytest.raises(RuntimeError, match=r".* must provide a `rotate` method."):
            _ = self.averager(basis, self._getSrc())

    def testAverager(self):
        """
        Construct a stack of images with known rotations.

        Rotationally averager the stack and compare output with known rotations.
        """

        # Construct the Averager and then call the `align` method
        avgr = self.averager(self.basis, self._getSrc(), n_angles=self.n_search_angles)
        _rotations, _shifts, _ = avgr.align(self.classes, self.reflections, self.coefs)

        self.assertIsNone(_shifts)

        # Crude check that we are closer to known angle than the next rotation
        self.assertTrue(np.all((_rotations - self.thetas) <= (self.step / 2)))

        # Fine check that we are within n_angles.
        self.assertTrue(
            np.all((_rotations - self.thetas) <= (2 * np.pi / self.n_search_angles))
        )


@pytest.mark.filterwarnings("ignore:Gimbal lock detected")
class BFSRAverager2DTestCase(BFRAverager2DTestCase):

    averager = BFSRAverager2D

    def setUp(self):
        # Inherit basic params from the base class
        super(BFRAverager2DTestCase, self).setUp()

        # Setup shifts, don't shift the base image
        self.shifts = np.zeros((self.n_img, 2))
        self.shifts[1:, 0] = 2
        self.shifts[1:, 1] = 4

        # Execute the remaining setup from BFRAverager2DTestCase
        super().setUp()

    def testNoShift(self):
        """
        Test we raise an error when our basis does not provide `shift` method.
        """

        # DiracBasis does not provide `rotate` or `shift`.
        basis = DiracBasis((self.resolution, self.resolution), dtype=self.dtype)

        # The missing `rotate` case was already covered by (inherited) NoRot.
        # Add a dummy rotate method; we will still be missing `shift`,
        basis.rotate = lambda x: x

        # and that should raise an error during instantiation.
        with pytest.raises(RuntimeError, match=r".* must provide a `shift` method."):
            _ = self.averager(basis, self._getSrc())

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
            n_x_shifts=1,
            n_y_shifts=1,
        )
        _rotations, _shifts, _ = avgr.align(self.classes, self.reflections, self.coefs)

        # Crude check that we are closer to known angle than the next rotation
        self.assertTrue(np.all((_rotations - self.thetas) <= (self.step / 2)))

        # Fine check that we are within n_angles.
        self.assertTrue(
            np.all((_rotations - self.thetas) <= (2 * np.pi / self.n_search_angles))
        )

        # Check that we are _not_ shifting the base image
        self.assertTrue(np.all(_shifts[0][0] == 0))
        # Check that we produced estimated shifts away from origin
        #  Note that Simulation's rot+shift is generally not equal to shift+rot.
        #  Instead we check that some combination of
        #  non zero shift+rot improved corr.
        #  Perhaps in the future should check more details.
        self.assertTrue(np.all(np.hypot(*_shifts[0][1:].T) >= 1))


@pytest.mark.filterwarnings("ignore:Gimbal lock detected")
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
            source=self._getSrc(),
            dtype=self.dtype,
        )
        _rotations, _shifts, _ = avgr.align(self.classes, self.reflections, self.coefs)

        # Crude check that we are closer to known angle than the next rotation
        self.assertTrue(np.all((_rotations - self.thetas) <= (self.step / 2)))

        # Fine check that we are within one degree.
        self.assertTrue(np.all((_rotations - self.thetas) <= (2 * np.pi / 360.0)))

        # Check that we are _not_ shifting the base image
        self.assertTrue(np.all(_shifts[0][0] == 0))
        # Check that we produced estimated shifts away from origin
        #  Note that Simulation's rot+shift is generally not equal to shift+rot.
        #  Instead we check that some combination of
        #  non zero shift+rot improved corr.
        #  Perhaps in the future should check more details.
        self.assertTrue(np.all(np.hypot(*_shifts[0][1:].T) >= 1))


@pytest.mark.filterwarnings("ignore:Gimbal lock detected")
class BFSReddyChatterjiAverager2DTestCase(ReddyChatterjiAverager2DTestCase):

    averager = BFSReddyChatterjiAverager2D
