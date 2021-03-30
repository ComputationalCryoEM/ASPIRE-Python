import logging

import numpy as np
from tqdm import tqdm

from aspire.classification import Class2D
from aspire.covariance import RotCov2D
from aspire.utils import complex_type

logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt


class RIRClass2D(Class2D):
    def __init__(self, src, basis, k_components=400, dtype=None):
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
        self.basis = basis
        if self.dtype != self.basis.dtype:
            logger.warning(
                f"RIRClass2D basis.dtype {self.basis.dtype} does not match self.dtype {self.dtype}."
            )

        # Sanity Check
        assert hasattr(basis, "calculate_bispectrum")

    def classify(self):
        """
        Perform the 2D images classification.
        """

        ## Stage 1: Compute Coefficients
        # Memioze/batch this later when result is working

        self.coef = self.basis.evaluate_t(
            src.images(0, self.src.n)
        )  # basis coefficients

        ### Compute Bispectrum
        for i in range(self.src.n):
            basis.calculate_bispectrum(self.coef[i])

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
