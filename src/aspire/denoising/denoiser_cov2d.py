import logging
import numpy as np

from aspire.denoising import Denoiser
from aspire.utils.filters import RadialCTFFilter
from aspire.utils.blk_diag_func import blk_diag_partition
from aspire.utils.blk_diag_func import blk_diag_eye
from aspire.basis.ffb_2d import FFBBasis2D
from aspire.estimation.covar2d import RotCov2D
from aspire.utils.optimize import fill_struct

logger = logging.getLogger(__name__)


class DenoiserCov2D(Denoiser):
    """
    Define a derived class for denoising 2D images using Cov2D method
    """
    def __init__(self, src, basis, var_noise, ctf_info):
        """
        Initialize an object for denoising 2D images using Cov2D method

        :param src: The source object of 2D images with metadata
        :param basis: The basis method to expand 2D images
        :param var_noise: The estimated variance of noise
        :param ctf_info: Whether the CTF filters are included
        """
        super().__init__(src)
        self.imgs_noise = self.src.images(start=0, num=np.inf).asnumpy()
        self.var_noise = var_noise
        if not isinstance(basis, FFBBasis2D):
            raise NotImplementedError('Currently only fast FB method is supported')
        self.coeffs_noise = None
        self.coeffs_estim = None
        self.ctf_idx = None
        self.ctf_fb = None
        self.basis = basis

        # Assign the CTF information and index for each image
        if ctf_info and not (src.filters is None):
            logger.info(f'Represent CTF filters in FB basis')
            uniq_ctf_filters = list(set(src.filters))
            # Create the indices of CTF filters from all images
            self.ctf_idx = np.array([uniq_ctf_filters.index(f) for f in src.filters])
            # Evaluate CTFs in the FFB basis
            self.ctf_fb = [f.fb_mat(basis) for f in uniq_ctf_filters]
        else:
            logger.info(f'CTF filters are not included in Cov2D denoising')
            # set all CTF filters to an identity filter
            self.ctf_idx = np.zeros(self.nimg, dtype=int)
            self.ctf_fb = [blk_diag_eye(blk_diag_partition(RadialCTFFilter().fb_mat(basis)))]

    def denoise(self, covar_opt=None):
        """
        Denoiser 2D images using Cov2D method

        :param basis: The basis method to expand 2D images
        :param covar_opt: The option list for building Cov2D matrix
        """

        self.coeffs_noise = self.basis.evaluate_t(self.imgs_noise)

        cov2d = RotCov2D(self.basis)

        default_opt = {'shrinker': 'frobenius_norm', 'verbose': 0, 'max_iter': 250,
            'iter_callback': [], 'store_iterates': False, 'rel_tolerance': 1e-12,
            'precision': 'float64'}
        covar_opt = fill_struct(default_opt, covar_opt)

        mean_coeffs_est = cov2d.get_mean(self.coeffs_noise, self.ctf_fb, self.ctf_idx)

        covar_coeffs_est = cov2d.get_covar(self.coeffs_noise, self.ctf_fb, self.ctf_idx, mean_coeffs_est,
                                               noise_var=self.var_noise, covar_est_opt=covar_opt)

        self.coeffs_estim = cov2d.get_cwf_coeffs(self.coeffs_noise, self.ctf_fb, self.ctf_idx,
                                                     mean_coeff=mean_coeffs_est,
                                                     covar_coeff=covar_coeffs_est, noise_var=self.var_noise)

        # Convert Fourier-Bessel coefficients back into 2D images
        self.imgs_estim = self.basis.evaluate(self.coeffs_estim)
