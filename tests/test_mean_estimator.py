import os.path
import tempfile
from unittest import TestCase

import numpy as np
from pytest import raises

from aspire.basis import Coef, FBBasis3D
from aspire.operators import RadialCTFFilter
from aspire.reconstruction import MeanEstimator
from aspire.source.simulation import Simulation
from aspire.utils import grid_3d

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class MeanEstimatorTestCase(TestCase):
    def setUp(self):
        self.dtype = np.float32
        self.L = 8
        self.sim = Simulation(
            n=512,
            C=1,  # single volume
            unique_filters=[
                RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)
            ],
            dtype=self.dtype,
            seed=1616,
        )
        # Todo, swap for default FFB
        self.basis = FBBasis3D((self.L,) * 3, dtype=self.dtype)

        self.estimator = MeanEstimator(
            self.sim, basis=self.basis, preconditioner="none"
        )

        self.estimator_with_preconditioner = MeanEstimator(
            self.sim, basis=self.basis, preconditioner="circulant"
        )
        self.mask = grid_3d(self.L)["r"] < 1

    def tearDown(self):
        pass

    def testEstimateResolutionError(self):
        """
        Test mismatched resolutions yields a relevant error message.
        """

        with raises(ValueError, match=r".*resolution.*"):
            # This basis is intentionally the wrong resolution.
            incorrect_basis = FBBasis3D((2 * self.L,) * 3, dtype=self.dtype)

            _ = MeanEstimator(self.sim, basis=incorrect_basis, preconditioner="none")

    def testEstimate(self):
        estimate = self.estimator.estimate()

        est = estimate.asnumpy() * self.mask
        vol = self.sim.vols.asnumpy() * self.mask

        np.testing.assert_allclose(
            est / np.linalg.norm(est), vol / np.linalg.norm(vol), atol=0.1
        )

    def testAdjoint(self):
        # Mean coefs formed by backprojections
        mean_b_coef = self.estimator.src_backward()

        # Evaluate mean coefs into a volume
        est = Coef(self.basis, mean_b_coef).evaluate()

        # Mask off corners of volume
        vol = self.sim.vols.asnumpy() * self.mask

        # Assert the mean volume is close to original volume
        np.testing.assert_allclose(
            est / np.linalg.norm(est), vol / np.linalg.norm(vol), atol=0.1
        )

    def testOptimize1(self):
        """
        x = self.estimator.conj_grad(mean_b_coef)
        """

    def testOptimize2(self):
        """
        x = self.estimator_with_preconditioner.conj_grad(mean_b_coef)
        """

    def testCheckpoint(self):
        """Exercise the checkpointing and max iterations branches."""
        test_iter = 2
        with tempfile.TemporaryDirectory() as tmp_input_dir:
            prefix = os.path.join(tmp_input_dir, "new", "dirs", "chk")
            estimator = MeanEstimator(
                self.sim,
                basis=self.basis,
                preconditioner="none",
                checkpoint_iterations=test_iter,
                maxiter=test_iter + 1,
                checkpoint_prefix=prefix,
            )

            # Assert we raise when reading `maxiter`.
            with raises(RuntimeError, match="Unable to converge!"):
                _ = estimator.estimate()

            # Load the checkpoint coefficients while tmp_input_dir exists.
            b_chk = np.load(f"{prefix}_iter{test_iter:04d}.npy")

        # Restart estimate from checkpoint
        _ = self.estimator.estimate(b_coef=b_chk)

    def testCheckpointArgs(self):
        with tempfile.TemporaryDirectory() as tmp_input_dir:
            prefix = os.path.join(tmp_input_dir, "chk")

            for junk in [-1, 0, "abc"]:
                # Junk `checkpoint_iterations` values
                with raises(
                    ValueError, match=r".*iterations.*should be a positive integer.*"
                ):
                    _ = MeanEstimator(
                        self.sim,
                        basis=self.basis,
                        preconditioner="none",
                        checkpoint_iterations=junk,
                        checkpoint_prefix=prefix,
                    )
                # Junk `maxiter` values
                with raises(
                    ValueError, match=r".*maxiter.*should be a positive integer.*"
                ):
                    _ = MeanEstimator(
                        self.sim,
                        basis=self.basis,
                        preconditioner="none",
                        maxiter=junk,
                        checkpoint_prefix=prefix,
                    )
