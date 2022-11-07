import logging
import os
from unittest import TestCase

import numpy as np
import pytest
from sklearn import datasets

from aspire.basis import FFBBasis2D, FSPCABasis
from aspire.classification import (
    BFRAverager2D,
    ClassSelector,
    RIRClass2D,
    TopClassSelector,
)
from aspire.classification.legacy_implementations import bispec_2drot_large, pca_y
from aspire.operators import ScalarFilter
from aspire.source import Simulation
from aspire.utils import utest_tolerance
from aspire.volume import Volume

from .test_averager2d import xfail_ray_dev

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
        self.src.cache()  # Precompute image stack

        # Calculate some projection images
        self.imgs = self.src.images[:]

        # Configure an FSPCA basis
        self.fspca_basis = FSPCABasis(self.src, noise_var=0)

    def testExpandEval(self):
        coef = self.fspca_basis.expand_from_image_basis(self.imgs)
        recon = self.fspca_basis.evaluate_to_image_basis(coef)

        # Check recon is close to imgs
        rmse = np.sqrt(np.mean(np.square(self.imgs.asnumpy() - recon.asnumpy())))
        logger.info(f"FSPCA Expand Eval Image Round True RMSE: {rmse}")
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

    def testBasisTooSmall(self):
        """
        When number of components is more than basis functions raise with descriptive error.
        """
        fb_basis = FFBBasis2D((self.resolution, self.resolution), dtype=self.dtype)

        with pytest.raises(ValueError, match=r".*Reduce components.*"):
            # Configure an FSPCA basis
            _ = FSPCABasis(
                self.src, basis=fb_basis, components=fb_basis.count * 2, noise_var=0
            )


class RIRClass2DTestCase(TestCase):
    def setUp(self):
        self.n_classes = 5
        self.resolution = 16
        self.dtype = np.float64
        self.n_img = 150

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
            n=self.n_img,
            vols=v,
            dtype=self.dtype,
        )

        # With Noise
        noise_var = 0.01 * np.var(np.sum(v[0], axis=0))
        noise_filter = ScalarFilter(dim=2, value=noise_var)
        self.noisy_src = Simulation(
            L=self.resolution,
            n=self.n_img,
            vols=v,
            dtype=self.dtype,
            noise_filter=noise_filter,
        )

        # Set up FFB
        # Setup a Basis
        self.basis = FFBBasis2D((self.resolution, self.resolution), dtype=self.dtype)

        # Create Basis, use precomputed Basis
        self.clean_fspca_basis = FSPCABasis(
            self.clean_src, self.basis, noise_var=0
        )  # Note noise_var assigned zero, skips eigval filtering.

        self.clean_fspca_basis_compressed = FSPCABasis(
            self.clean_src, self.basis, components=101, noise_var=0
        )  # Note noise_var assigned zero, skips eigval filtering.

        # Ceate another fspca_basis, use autogeneration FFB2D Basis
        self.noisy_fspca_basis = FSPCABasis(self.noisy_src)

    def testSourceTooSmall(self):
        """
        When number of images in source is less than requested bispectrum components,
        raise with descriptive error.
        """

        with pytest.raises(
            RuntimeError, match=r".*Increase number of images or reduce components.*"
        ):
            _ = RIRClass2D(
                self.clean_src,
                fspca_components=self.clean_src.n * 4,
                bispectrum_components=self.clean_src.n * 2,
            )

    def testIncorrectComponents(self):
        """
        Check we raise with inconsistent configuration of FSPCA components.
        """

        with pytest.raises(
            RuntimeError, match=r"`pca_basis` components.*provided by user."
        ):
            _ = RIRClass2D(
                self.clean_src,
                self.clean_fspca_basis,  # 400 components
                fspca_components=100,
                large_pca_implementation="legacy",
                nn_implementation="legacy",
                bispectrum_implementation="legacy",
            )

        # Explicitly providing the same number should be okay.
        _ = RIRClass2D(
            self.clean_src,
            self.clean_fspca_basis,  # 400 components
            fspca_components=self.clean_fspca_basis.components,
            bispectrum_components=100,
            large_pca_implementation="legacy",
            nn_implementation="legacy",
            bispectrum_implementation="legacy",
        )

    def testRIRLegacy(self):
        """
        Currently just tests for runtime errors.
        """

        clean_fspca_basis = FSPCABasis(
            self.clean_src, self.basis, noise_var=0, components=100
        )  # Note noise_var assigned zero, skips eigval filtering.

        rir = RIRClass2D(
            self.clean_src,
            clean_fspca_basis,
            n_classes=5,
            bispectrum_components=42,
            large_pca_implementation="legacy",
            nn_implementation="legacy",
            bispectrum_implementation="legacy",
            selector=TopClassSelector(),
            num_procs=1 if xfail_ray_dev() else None,
        )

        classification_results = rir.classify()
        _ = rir.averages(*classification_results)

    def testRIRDevelBisp(self):
        """
        Currently just tests for runtime errors.
        """

        # Use the basis class setup, only requires a Source.
        rir = RIRClass2D(
            self.clean_src,
            fspca_components=self.clean_fspca_basis.components,
            bispectrum_components=self.clean_fspca_basis.components - 1,
            large_pca_implementation="legacy",
            nn_implementation="legacy",
            bispectrum_implementation="devel",
            num_procs=1 if xfail_ray_dev() else None,
        )

        _ = rir.classify()

    def testRIRsk(self):
        """
        Excercises the eigenvalue based filtering,
        along with other swappable components.

        Currently just tests for runtime errors.
        """
        rir = RIRClass2D(
            self.noisy_src,
            self.noisy_fspca_basis,
            bispectrum_components=100,
            sample_n=42,
            n_classes=self.n_classes,
            large_pca_implementation="sklearn",
            nn_implementation="sklearn",
            bispectrum_implementation="devel",
            averager=BFRAverager2D(
                self.noisy_fspca_basis.basis,  # FFB basis
                self.noisy_src,
                n_angles=100,
            ),
        )

        classification_results = rir.classify()
        _ = rir.averages(*classification_results)

    def testEigenImages(self):
        """
        Test we can return eigenimages.
        """

        # Get the eigenimages from an FSPCA basis for testing
        eigimg_uncompressed = self.clean_fspca_basis.eigen_images()

        # Get the eigenimages from a compressed FSPCA basis for testing
        eigimg_compressed = self.clean_fspca_basis_compressed.eigen_images()

        # Check they are close.
        # Note it is expected the compression reorders the eigvecs,
        #  and thus the eigimages.
        # We sum over all the eigimages to yield an "average" for comparison
        self.assertTrue(
            np.allclose(
                np.sum(eigimg_uncompressed.asnumpy(), axis=0),
                np.sum(eigimg_compressed.asnumpy(), axis=0),
            )
        )

    def testComponentSize(self):
        """
        Tests we raise when number of components are too small.

        Also tests dtype mismatch behavior.
        """

        with pytest.raises(RuntimeError, match=r".*Reduce bispectrum_components.*"):
            _ = RIRClass2D(
                self.clean_src,
                self.clean_fspca_basis,
                bispectrum_components=self.clean_src.n + 1,
                dtype=np.float64,
            )

    def testImplementations(self):
        """
        Test optional implementations handle bad inputs with a descriptive error.
        """

        # Nearest Neighbhor component
        with pytest.raises(ValueError, match=r"Provided nn_implementation.*"):
            _ = RIRClass2D(
                self.clean_src,
                self.clean_fspca_basis,
                bispectrum_components=int(0.75 * self.clean_fspca_basis.basis.count),
                nn_implementation="badinput",
            )

        # Large PCA component
        with pytest.raises(ValueError, match=r"Provided large_pca_implementation.*"):
            _ = RIRClass2D(
                self.clean_src,
                self.clean_fspca_basis,
                large_pca_implementation="badinput",
            )

        # Bispectrum component
        with pytest.raises(ValueError, match=r"Provided bispectrum_implementation.*"):
            _ = RIRClass2D(
                self.clean_src,
                self.clean_fspca_basis,
                bispectrum_implementation="badinput",
            )

        # Legacy Bispectrum implies legacy bispectrum (they're integrated).
        with pytest.raises(
            ValueError, match=r'"legacy" bispectrum_implementation implies.*'
        ):
            _ = RIRClass2D(
                self.clean_src,
                self.clean_fspca_basis,
                bispectrum_implementation="legacy",
                large_pca_implementation="sklearn",
            )

        # Currently we only FSPCA Basis in RIRClass2D
        with pytest.raises(
            RuntimeError,
            match="RIRClass2D has currently only been developed for pca_basis as a FSPCABasis.",
        ):
            _ = RIRClass2D(self.clean_src, self.basis)

    def testSelectionImplementations(self):
        """
        Test optional implementations handle bad inputs with a descriptive error.
        """

        class CustomClassSelector(ClassSelector):
            def __init__(self, x):
                self.x = x

            def _select(self, n, classes, reflections, distances):
                return self.x

        # lower bound
        with pytest.raises(ValueError, match=r".*out of bounds.*"):
            rir = RIRClass2D(
                self.clean_src,
                self.clean_fspca_basis,
                n_classes=self.n_classes,
                bispectrum_components=self.clean_fspca_basis.components - 1,
                selector=CustomClassSelector(np.arange(self.n_classes) - 1),
            )
            _ = rir.averages(*rir.classify())

        # upper bound
        with pytest.raises(ValueError, match=r".*out of bounds.*"):
            rir = RIRClass2D(
                self.clean_src,
                self.clean_fspca_basis,
                n_classes=self.n_classes,
                bispectrum_components=self.clean_fspca_basis.components - 1,
                selector=CustomClassSelector(
                    np.arange(self.n_classes) + self.clean_src.n
                ),
            )
            _ = rir.averages(*rir.classify())

        # too short
        with pytest.raises(ValueError, match=r".*must be len.*"):
            rir = RIRClass2D(
                self.clean_src,
                self.clean_fspca_basis,
                n_classes=self.n_classes,
                bispectrum_components=self.clean_fspca_basis.components - 1,
                selector=CustomClassSelector(np.arange(self.n_classes - 1)),
            )
            _ = rir.averages(*rir.classify())

        # too long
        with pytest.raises(ValueError, match=r".*must be len.*"):
            rir = RIRClass2D(
                self.clean_src,
                self.clean_fspca_basis,
                n_classes=self.n_classes,
                bispectrum_components=self.clean_fspca_basis.components - 1,
                selector=CustomClassSelector(np.arange(self.n_classes + 1)),
            )
            _ = rir.averages(*rir.classify())

    def testIncorrectSelectorClass(self):
        """
        Test passing incorect ClassSelector raises with a descriptive error.
        """

        with pytest.raises(RuntimeError, match=r".*must be subclass of.*"):
            rir = RIRClass2D(
                self.clean_src,
                self.clean_fspca_basis,
                n_classes=self.n_classes,
                selector=range(self.n_classes),
            )
            _ = rir.averages(*rir.classify())


class LegacyImplementationTestCase(TestCase):
    """
    Cover branches of Legacy code not taken by the classification unit tests.
    """

    def setUp(self):
        pass

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
            # This should emit a warning before raising
            with self.assertWarns(RuntimeWarning):
                bispec_2drot_large(
                    coeff=np.arange(10),
                    freqs=np.arange(1, 11),
                    eigval=np.arange(10),
                    alpha=1 / 3,
                    sample_n=4000,
                )
