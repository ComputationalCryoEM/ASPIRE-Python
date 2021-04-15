import logging

import numpy as np
from tqdm import tqdm

from aspire.basis import FSPCABasis
from aspire.classification import Class2D
from aspire.covariance import RotCov2D
from aspire.utils import complex_type

logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt


class RIRClass2D(Class2D):
    def __init__(self, src, pca_basis, fspca_components=400, dtype=None):
        """
        Constructor of an object for classifying 2D images using
        Rotationally Invariant Representation (RIP) algorithm.

        :param src: Source instance
        :param basis: (Fast) Fourier Bessel Basis instance
        :param k_components: Components (top eigvals) to keep, default 400.
        :param dtype: optional dtype, otherwise taken from src.
        :return: RIRClass2D instance to be used to compute bispectrum-like rotationally invariant 2D classification.
        """
        super().__init__(src=src, dtype=dtype)

        self.pca_basis = pca_basis
        self.fb_basis = self.pca_basis.basis
        self.fspca_components = fspca_components

        # Type checks
        if self.dtype != self.fb_basis.dtype:
            logger.warning(
                f"RIRClass2D basis.dtype {self.basis.dtype} does not match self.dtype {self.dtype}."
            )

        # Sanity Checks
        assert hasattr(self.pca_basis, "calculate_bispectrum")

        # For now, only run with FSPCA
        assert isinstance(self.pca_basis, FSPCABasis)

    def classify(self):
        """
        Perform the 2D images classification.
        """

        ## Stage 1: Compute coef and reduce dimensionality.
        # Memioze/batch this later when result is working

        # Initial round of component truncation is before bispectrum.
        #  default of 400 components taken from legacy code.
        #  Take minumum here in case we have less than k coefs already.
        if self.fb_basis.complex_count > self.fspca_components:
            # Instantiate a new truncated (compressed) basis.
            self.pca_basis = self.pca_basis.truncate(self.fspca_components)

        # Expand into the compressed FSPCA space.
        fb_coef = self.fb_basis.evaluate_t(self.src.images(0, self.src.n))
        coef = self.pca_basis.expand(fb_coef)
        ## should be equiv, make a ut
        # coef = self.pca_basis.expand_from_image_basis(self.src.images(0, self.src.n))

        ### Compute Bispectrum
        for i in range(self.src.n):
            self.pca_basis.calculate_bispectrum(coef[i])

            ### Truncate Bispectrum (Probably optional?)

        ### Reduce dimensionality of Bispectrum (PCA/SVD)

        ## Stage 2: Compute Nearest Neighbors

        ## Stage 3: Align

    def nn_classification(self):
        """
        Perform nearest neighbor classification and alignment.
        """
        pass

    def output(self):
        """
        Return class averages.
        """
        pass
