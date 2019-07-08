import numpy as np
from aspyre.denoise import Denoise

import logging
logger = logging.getLogger(__name__)


class RotCov2D(Denoise):
    """
    Define a derived class for denoising 2D images using Cov2D method
    """

    def __init__(self, src, basis, as_type='single'):
        """
          constructor of an object for 2D covariance analysis
        """
        pass

    def get_mean(self, coeffs=None):
        """
        Calculate the mean vector from the expansion coefficient.
        param coeffs: A coefficient vector (or an array of coefficient vectors) to be evaluated.
        :return: The mean value vector for all images.
        """
        pass

    def get_covar(self, coeffs=None, mean_coeff=None,  do_refl=false):
        """
        Calculate the covariance matrix from the expansion coefficients.
        param mean_coeff: The mean vector calculated from the `b_coeff`.
        param b_coeffs: A coefficient vector (or an array of coefficient vectors) to be evaluated.
        param do_refl: If true, enforce invariance to reflection (default false).
        :return: The covariance matrix of coefficients for all images.

        """
        pass

    def get_coeffs(self):
        """
        Apply adjoint mapping to 2D images and obtain the coefficients

        :return: The coefficients of `basis` after the adjoint mapping is applied to the images.
        """
        pass
