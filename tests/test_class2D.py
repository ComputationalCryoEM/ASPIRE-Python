import logging
import os
from unittest import TestCase

import mrcfile
import numpy as np

from aspire.basis import FFBBasis2D, FSPCABasis
from aspire.classification import RIRClass2D
from aspire.image import Image
from aspire.operators import ScalarFilter
from aspire.source import Simulation
from aspire.volume import Volume

logger = logging.getLogger(__name__)


DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


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
        self.clean_src = src = Simulation(
            L=self.resolution,
            n=321,
            vols=v,
            dtype=self.dtype,
        )

        # With Noise
        noise_var = 0.01 * np.var(np.sum(v[0], axis=0))
        noise_filter = ScalarFilter(dim=2, value=noise_var)
        self.noisy_src = src = Simulation(
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
            bispectrum_implementation="legacy",
        )

        result = rir.classify()
        _ = rir.output(*result[:3], include_refl=True)
