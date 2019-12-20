
import logging
import numpy as np

from aspire.denoise import Denoise
from aspire.utils.filters import RadialCTFFilter
from aspire.utils.blk_diag_func import radial_filter2fb_mat
from aspire.utils.blk_diag_func import blk_diag_partition
from aspire.utils.blk_diag_func import blk_diag_eye
from aspire.basis.ffb_2d import FFBBasis2D
from aspire.estimation.covar2d import RotCov2D
from aspire.utils.matrix import anorm

logger = logging.getLogger(__name__)


class DenoiseCov2D(Denoise):
    """
    Define a derived class for denoising 2D images using Cov2D method
    """
    def __init__(self, src, math_basis, var_noise, ctf_info):
        """
        Initialize an object for denoising 2D images using Cov2D method

        :param src: The source object of 2D images with metadata
        :param math_basis: The basis method to expand 2D images
        :param var_noise: The estimated variance of noise
        :param ctf_info: Whether the CTF filters are included
        """
        super().__init__(src)
        self.var_noise = var_noise
        if not isinstance(math_basis, FFBBasis2D):
            raise NotImplementedError('Currently only Fast FB method is supported')
        self.coeffs_noise = None
        self.coeffs_estim = None
        self.ctf_idx = None
        self.ctf_fb = None

        # Assign the CTF information and index for each image
        if ctf_info and not (src.filters is None):
            logger.info(f'Convert non radical CTF filters to radical ones')
            uniq_rad_filters = []
            uniq_ctf_filters = list(set(src.filters))
            for f in uniq_ctf_filters:
                # convert non radical CTF filters to radial ones
                defocus = np.sqrt(f.defocus_u*f.defocus_u + f.defocus_v*f.defocus_v)
                uniq_rad_filters.append(RadialCTFFilter(
                    f.pixel_size, f.voltage, defocus=defocus, Cs=f.Cs, alpha=f.alpha))
            self.ctf_idx = np.array([uniq_ctf_filters.index(f) for f in src.filters])
            # Evaluate CTF in the FFB basis
            self.ctf_fb = [radial_filter2fb_mat(f.evaluate, math_basis)
                           for f in uniq_rad_filters]
        else:
            logger.info(f'CTF filters are not included in Cov2D denoising.')
            f = RadialCTFFilter(1.0, 200, defocus=1.5e4, Cs=2.0, alpha=0.1)
            # set CTF filters to an identity filter
            self.ctf_idx = np.zeros(self.nimg, dtype=int)
            self.ctf_fb = [blk_diag_eye(blk_diag_partition(radial_filter2fb_mat(f.evaluate, math_basis)))]

    def denoise(self, math_basis, covar_opt=None):
        """
        Denoise 2D images using Cov2D method

        :param math_basis: The basis method to expand 2D images
        :param covar_opt: The option list for building Cov2D matrix
        """
        if isinstance(math_basis, FFBBasis2D):

            self.coeffs_noise = math_basis.evaluate_t(self.imgs_noise)

            cov2d = RotCov2D(math_basis)

            if covar_opt is None:
                covar_opt = {'shrinker': 'frobenius_norm', 'verbose': 0, 'max_iter': 250,
                'iter_callback': [], 'store_iterates': False, 'rel_tolerance': 1e-12,
                'precision': 'float64', 'preconditioner': 'identity'}

            mean_coeffs_est = cov2d.get_mean(self.coeffs_noise, self.ctf_fb, self.ctf_idx)

            covar_coeffs_est = cov2d.get_covar(self.coeffs_noise, self.ctf_fb, self.ctf_idx, mean_coeffs_est,
                                               noise_var=self.var_noise, covar_est_opt=covar_opt)

            self.coeffs_estim = cov2d.get_cwf_coeffs(self.coeffs_noise, self.ctf_fb, self.ctf_idx,
                                                     mean_coeff=mean_coeffs_est,
                                                     covar_coeff=covar_coeffs_est, noise_var=self.var_noise)

            # Convert Fourier-Bessel coefficients back into 2D images
            self.imgs_estim = math_basis.evaluate(self.coeffs_estim)

            # Calculate the normalized RMSE of the estimated images.
            nrmse_ims = anorm(self.imgs_estim-self.imgs_noise)/anorm(self.imgs_noise)
            logger.info(f'Estimated images normalized RMSE: {nrmse_ims}')
        else:
            raise NotImplementedError('Currently only Fast FB method is supported for Cov2D denoising.')




