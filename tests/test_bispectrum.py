import os
from unittest import TestCase

import numpy as np

from aspire.basis import FFBBasis2D, FSPCABasis
from aspire.image import Image
from aspire.source import Simulation
from aspire.volume import Volume

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class BispectrumTestCase(TestCase):
    def setUp(self):
        self.dtype = np.float32

        # Test Volume
        v = Volume(
            np.load(os.path.join(DATA_DIR, "clean70SRibosome_vol.npy")).astype(
                self.dtype
            )
        )

        # Create Sim object then extract a projection.
        self.src = Simulation(L=v.resolution, n=10, vols=v, dtype=v.dtype)

        # Original projection image to transform.
        self.orig_img = self.src.images(0, 1)

        # Rotate 90 degrees in cartesian coordinates.
        self.rt90_img = Image(np.rot90(self.orig_img.asnumpy(), axes=(1, 2)))

        # Prepare a Fourier Bessel Basis
        self.basis = FFBBasis2D((self.orig_img.res,) * 2, dtype=self.dtype)
        self.v1 = self.basis.evaluate_t(self.orig_img)
        self.v2 = self.basis.evaluate_t(self.rt90_img)

        # Prepare a FSPCA Basis too.
        self.fspca_basis = FSPCABasis(self.src, self.basis)

    def testRotationalInvarianceFB(self):
        """
        Compare FB/FFB bispectrums before and after rotation.
        Compares a slab (by freq_cutoff) to reduce size.
        """

        b1 = self.basis.calculate_bispectrum(self.v1, freq_cutoff=3)
        b2 = self.basis.calculate_bispectrum(self.v2, freq_cutoff=3)

        self.assertTrue(np.allclose(b1, b2))

    def testRotationalInvarianceFSPCA(self):
        """
        Compare FSPCA bispctrum before and after rotation.
        """

        self.fspca_basis.build(self.v1)
        self.fspca_basis.build(self.v2)

        # Convert to complex
        cv1 = self.basis.to_complex(self.v1)
        cv2 = self.basis.to_complex(self.v2)

        # Compute Bispect
        w1 = self.fspca_basis.calculate_bispectrum(cv1)
        w2 = self.fspca_basis.calculate_bispectrum(cv2)

        self.assertTrue(np.allclose(w1, w2))
