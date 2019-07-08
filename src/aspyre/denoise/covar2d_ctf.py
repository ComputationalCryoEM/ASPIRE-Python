from aspyre.denoise.covar2d import RotCov2D

import logging

logger = logging.getLogger(__name__)


class Cov2DCTF(RotCov2D):
    """
    Define a derived class for denoising 2D images using CTF and Wiener Cov2D method
    """

    def get_mean_ctf(self, coeffs, ctf_fb, ctf_idx):
        """
        Calculate the mean vector from the expansion coefficient.
        param b_coeffs: A coefficient vector (or an array of coefficient vectors) to be evaluated.
        :return: The mean value vector for all images.
        """
        pass

    def get_covar_ctf(self, coeffs, ctf_fb, ctf_idx, mean_coeff, noise_var, covar_est_opt=None):
        """
        Calculate the covariance matrix from the expansion coefficients with CTF correction.
        param mean_coeff: The mean vector calculated from the `b_coeff`.
        param b_coeffs: A coefficient vector (or an array of coefficient vectors) to be evaluated.
        param do_refl: If true, enforce invariance to reflection (default false).
        :return: The covariance matrix of coefficients for all images.

        """
        pass

    def _shrink(self, covar_b_coeff, noise_variance, method=None):
        """
        Shrink covariance matrix
        :param covar_b_coeff: Outer products of the mean-subtracted images
        :param noise_variance: Noise variance
        :param method: One of None/'frobenius_norm'/'operator_norm'/'soft_threshold'
        :return: Shrunk covariance matrix
        """
        pass

    def conj_grad(self, b_coeff):
        pass

    def get_wiener_ctf(self, coeffs, filter_fb, filter_idx, mean_coeff, covar_coeff, noise_var):
        pass
