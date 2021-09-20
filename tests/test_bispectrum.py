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
        ).downsample(32)

        # Create Sim object.
        # Creates 10 projects so there is something to feed FSPCABasis.
        self.src = Simulation(L=v.resolution, n=10, vols=v, dtype=v.dtype)

        # Original projection image to transform.
        self.orig_img = self.src.images(0, 1)

        # Rotate 90 degrees in cartesian coordinates using third party tool.
        self.rt90_img = Image(np.rot90(self.orig_img.asnumpy(), axes=(1, 2)))

        # Prepare a Fourier Bessel Basis
        self.basis = FFBBasis2D((self.orig_img.res,) * 2, dtype=self.dtype)
        self.v1 = self.basis.evaluate_t(self.orig_img)
        self.v2 = self.basis.evaluate_t(self.rt90_img)
        # These should _not_ be equal or the test is pointless.
        self.assertFalse(np.allclose(self.v1, self.v2))

        # Prepare a FSPCA Basis too.
        self.fspca_basis = FSPCABasis(self.src, self.basis)

    def testRotationalInvarianceFB(self):
        """
        Compare FB/FFB bispectrums before and after rotation.
        Compares a slab of 3d bispectrum (k1 q1) x (k2 q2) x q3,
        by freq_cutoff of q3 to reduce size.
        """

        # Compute Bispectrum
        q = 3  # slab cutoff
        # We'll also excercise some other options to check they don't crash
        b1 = self.basis.calculate_bispectrum(
            self.v1, freq_cutoff=q, flatten=True, filter_nonzero_freqs=True
        )
        b2 = self.basis.calculate_bispectrum(
            self.v2, freq_cutoff=q, flatten=True, filter_nonzero_freqs=True
        )

        # Bispectrum should be equivalent
        self.assertTrue(np.allclose(b1, b2))

    def testRotationalInvarianceFSPCA(self):
        """
        Compare FSPCA bispctrum before and after rotation.
        """

        # Compute Bispectrum
        w1 = self.fspca_basis.calculate_bispectrum(self.v1)
        w2 = self.fspca_basis.calculate_bispectrum(self.v2)

        # Bispectrum should be equivalent
        self.assertTrue(np.allclose(w1, w2))

    def testRotationalInvarianceFSPCACompressed(self):
        """
        Compare Compressed FSPCA bispctrum before and after rotation.

        This is most like what is used in practice for RIR.
        """

        # Create a reduced rank (compressed) FSPCABasis, top 100 components.
        components = 100
        compressed_fspca = FSPCABasis(self.src, self.basis, components=components)

        # Compress using representation in the compressed FSPCA
        cv1_r = compressed_fspca.expand(self.v1)
        cv2_r = compressed_fspca.expand(self.v2)

        # Check we are really compressed
        self.assertTrue(compressed_fspca.complex_count == components)

        # Compute Bispectrum
        w1 = compressed_fspca.calculate_bispectrum(cv1_r)
        w2 = compressed_fspca.calculate_bispectrum(cv2_r)

        # Bispectrum should be equivalent
        self.assertTrue(np.allclose(w1, w2))
