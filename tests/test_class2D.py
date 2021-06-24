import logging
import os
from unittest import TestCase

import numpy as np
import pytest
from sklearn import datasets

from aspire.basis import FFBBasis2D, FSPCABasis
from aspire.classification import Class2D, RIRClass2D
from aspire.classification.legacy_implementations import (
    bispec_2drot_large,
    icfft2,
    pca_y,
)
from aspire.operators import ScalarFilter
from aspire.source import Simulation
from aspire.utils import utest_tolerance
from aspire.volume import Volume

logger = logging.getLogger(__name__)


DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class FSPCATestCase(TestCase):
    def setUp(self):
        self.resolution = 16
        self.dtype = np.float32

        # Get a volume
        v = Volume(
            np.load(os.path.join(DATA_DIR, "clean70SRibosome_vol.npy")).astype(
                self.dtype
            )
        )
        v = v.downsample(self.resolution)

        # Create a src from the volume
        self.src = Simulation(
            L=self.resolution,
            n=321,
            vols=v,
            dtype=self.dtype,
        )

        # Calculate some projection images
        self.imgs = self.src.images(0, self.src.n)

        # Configure an FSPCA basis
        self.fspca_basis = FSPCABasis(self.src, noise_var=0)
        # Compute the FSPCA basis
        self.fspca_basis.build()

    def testExpandEval(self):
        coef = self.fspca_basis.expand_from_image_basis(self.imgs)
        recon = self.fspca_basis.evaluate_to_image_basis(coef)

        # Check recon is close to imgs
        rmse = np.sqrt(np.mean(np.square(self.imgs.asnumpy() - recon.asnumpy())))
        logger.info(f"FSPCA Expand Eval Image Round Trupe RMSE: {rmse}")
        self.assertTrue(rmse < utest_tolerance(self.dtype))

    def testComplexConversionErrors(self):
        """
        Test we raise when passed incorrect dtypes.

        Also checks we can handle 0d vector in `to_real`.

        Most other cases covered by classification unit tests.
        """

        with pytest.raises(
            TypeError, match="coef provided to to_complex should be real."
        ):
            _ = self.fspca_basis.to_complex(
                np.arange(self.fspca_basis.count, dtype=np.complex64)
            )

        with pytest.raises(
            TypeError, match="coef provided to to_real should be complex."
        ):
            _ = self.fspca_basis.to_real(
                np.arange(self.fspca_basis.count, dtype=np.float32).flatten()
            )

    def testRotate(self):
        """
        Trivial test of rotation in FSPCA Basis.

        Also covers to_real and to_complex conversions in FSPCA Basis.
        """
        coef = self.fspca_basis.expand_from_image_basis(self.imgs)
        # rotate by pi
        rot_coef = self.fspca_basis.rotate(coef, radians=np.pi)
        rot_imgs = self.fspca_basis.evaluate_to_image_basis(rot_coef)

        for i, img in enumerate(self.imgs):
            rmse = np.sqrt(np.mean(np.square(np.flip(img) - rot_imgs[i])))
            self.assertTrue(rmse < 10 * utest_tolerance(self.dtype))


class RIRClass2DTestCase(TestCase):
    def setUp(self):
        self.resolution = 16
        self.dtype = np.float64

        # Create some projections
        v = Volume(
            np.load(os.path.join(DATA_DIR, "clean70SRibosome_vol.npy")).astype(
                self.dtype
            )
        )
        v = v.downsample(self.resolution)

        # Clean
        self.clean_src = Simulation(
            L=self.resolution,
            n=321,
            vols=v,
            dtype=self.dtype,
        )

        # With Noise
        noise_var = 0.01 * np.var(np.sum(v[0], axis=0))
        noise_filter = ScalarFilter(dim=2, value=noise_var)
        self.noisy_src = Simulation(
            L=self.resolution,
            n=123,
            vols=v,
            dtype=self.dtype,
            noise_filter=noise_filter,
        )

        # Set up FFB
        # Setup a Basis
        self.basis = FFBBasis2D((self.resolution, self.resolution), dtype=self.dtype)
        # Calculate Fourier Bessel Coefs
        self.clean_coefs = self.basis.evaluate_t(
            self.clean_src.images(0, self.clean_src.n)
        )
        self.noisy_coefs = self.basis.evaluate_t(
            self.noisy_src.images(0, self.clean_src.n)
        )

        # Create Basis
        self.clean_fspca_basis = FSPCABasis(
            self.clean_src, self.basis, noise_var=0
        )  # Note noise_var assigned zero, skips eigval filtering.
        self.clean_fspca_basis.build(self.clean_coefs)

        self.noisy_fspca_basis = FSPCABasis(self.noisy_src, self.basis)
        self.noisy_fspca_basis.build(self.noisy_coefs)

    def testClass2DBase(self):
        """
        Make sure the base class doesn't crash when using arguments.
        """
        _ = Class2D(self.clean_src)  # Default dtype
        _ = Class2D(self.clean_src, dtype=self.dtype)  # Consistent dtype
        _ = Class2D(self.clean_src, dtype=np.float16)  # Different dtype

    def testRIRLegacy(self):
        """
        Currently just tests for runtime errors.
        """
        rir = RIRClass2D(
            self.clean_src,
            self.clean_fspca_basis,
            large_pca_implementation="legacy",
            nn_implementation="legacy",
            bispectrum_implementation="legacy",
        )

        result = rir.classify()
        _ = rir.output(*result[:3], include_refl=False)

    def testRIRsk(self):
        """
        Excercises the eigenvalue based filtering,
        along with other swappable components.

        Currently just tests for runtime errors.
        """
        rir = RIRClass2D(
            self.noisy_src,
            self.noisy_fspca_basis,
            bispectrum_componenents=100,
            sample_n=42,
            large_pca_implementation="sklearn",
            nn_implementation="sklearn",
            bispectrum_implementation="devel",
        )

        result = rir.classify()
        _ = rir.output(*result[:3], include_refl=True)


class LegacyImplementationTestCase(TestCase):
    """
    Cover branches of Legacy code not taken by the classification unit tests.
    """

    def setUp(self):
        pass

    def test_icfft2_ndim2(self):
        """
        Check icfft2 works with 2D shapes.
        3D (stack of one) is checked in the classification unit tests.
        """
        fin = np.mgrid[:5, :5][0]
        # Note we fftshift in
        F = np.fft.fftshift(np.fft.fft2(fin))
        # and ifftshift out
        fout = np.fft.ifftshift(icfft2(F))

        self.assertTrue(np.allclose(fin, fout))

    def test_pca_y(self):
        """
        We want to check that real inputs and differing input matrix shapes work.

        Most of pca_y is covered by the classificiation unit tests.
        """

        # The iris dataset is a small 150 sample by 5 feature dataset in float64
        iris = datasets.load_iris()

        # Extract the data matrix, run once as is (150, 5),
        # and once tranposed  so shape[0] < shape[1] (5, 150)
        for x in (iris.data, iris.data.T):
            # Run pca_y and check reconstruction holds
            lsvec, svals, rsvec = pca_y(x, 5)

            # svd ~~> A = U S V = (U S) V
            recon = np.dot(lsvec * svals, rsvec)

            self.assertTrue(np.allclose(x, recon))

    def testBispectOverflow(self):
        """
        A zero value coeff will cause a div0 error in log call.
        Check it is raised.
        """

        with pytest.raises(ValueError, match="coeff_norm should not be -inf"):
            bispec_2drot_large(
                coeff=np.arange(10),
                freqs=np.arange(1, 11),
                eigval=np.arange(10),
                alpha=1 / 3,
                sample_n=4000,
            )
