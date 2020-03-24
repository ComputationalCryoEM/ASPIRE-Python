import logging
import numpy as np

from aspire.denoising import Denoiser
from aspire.basis.ffb_2d import FFBBasis2D
from aspire.estimation.covar2d import BatchedRotCov2D
from aspire.utils.optimize import fill_struct
from aspire.image import Image


logger = logging.getLogger(__name__)


class DenoiserCov2D(Denoiser):
    """
    Define a derived class for denoising 2D images using Cov2D method
    """
    def __init__(self, src, basis, var_noise):
        """
        Initialize an object for denoising 2D images using Cov2D method

        :param src: The source object of 2D images with metadata
        :param basis: The basis method to expand 2D images
        :param var_noise: The estimated variance of noise
        """
        super().__init__(src)
        self.var_noise = var_noise
        if not isinstance(basis, FFBBasis2D):
            raise NotImplementedError('Currently only fast FB method is supported')
        self.basis = basis
        self.cov2d = None
        self.mean_est = None
        self.covar_est = None
        self._im = None

    def denoise(self, covar_opt=None, batch_size=512):
        """
         Build covariance matrix of 2D images

        :param covar_opt: The option list for building Cov2D matrix
        :param batch_size: The batch size for processing images
        """

        # Initialize the rotationally invariant covariance matrix of 2D images
        # A fixed batch size is used to go through each image
        self.cov2d = BatchedRotCov2D(self.src, self.basis, batch_size=batch_size)

        default_opt = {'shrinker': 'frobenius_norm', 'verbose': 0, 'max_iter': 250,
            'iter_callback': [], 'store_iterates': False, 'rel_tolerance': 1e-12,
            'precision': 'float64'}
        covar_opt = fill_struct(covar_opt, default_opt)
        # Calculate the mean and covariance for the rotationally invariant covariance matrix of 2D images
        self.mean__est = self.cov2d.get_mean()

        self.covar_est = self.cov2d.get_covar(noise_var=self.var_noise, mean_coeff=self.mean_est,
                                           covar_est_opt=covar_opt)

    def images(self, istart=0, batch_size=512):
        """
        Denoise a batch size of 2D images using Cov2D matrix

        :param istart: the index of starting image
        :param batch_size: The batch size for processing images
        """
        src = self.src

        # Denoise one batch size of 2D images using the SPCAs from the rotationally invariant covariance matrix

        img_start = istart
        img_end = min(istart + batch_size, src.n)
        
        imgs_noise = src.images(img_start, batch_size)
        coeffs_noise = self.basis.evaluate_t(imgs_noise.data)
        logger.info(f'Estimating Cov2D coefficients for images from {img_start} to {img_end-1}')
        coeffs_estim = self.cov2d.get_cwf_coeffs(coeffs_noise, self.cov2d.ctf_fb,
                                                 self.cov2d.ctf_idx[img_start:img_start+batch_size],
                                                 mean_coeff=self.mean_est, covar_coeff=self.covar_est,
                                                 noise_var=self.var_noise)

        # Convert Fourier-Bessel coefficients back into 2D images
        logger.info(f'Converting Cov2D coefficients back to 2D images')
        imgs_estim = self.basis.evaluate(coeffs_estim)
        imgs_denoised = Image(imgs_estim)

        return imgs_denoised

    def cache(self, im=None):
        logger.info('Caching denoised images')

        if im is None:
            im = self.images(start=0, batch_size=self.src.n)
        self._im = im



    def save(self, batch_size=512, overwrite=False):
        """
        Save the denoised images to mrc files

        :param batch_size: Batch size of images to query.
        :param overwrite: Option to overwrite the output mrcs files.
        """
        logger.info("save denoise images")
        for istart in range(0, self.src.n, batch_size):

            if self._im is None:
                logger.info(f'save images {istart} to {istart+batch_size-1} after denoising')
                imgs_estim = self.images(istart=istart, batch_size=batch_size)
            else:
                logger.info(f'save images {istart} to {istart+batch_size-1} from cache')
                indices = np.arange(istart, min(istart + batch_size, self.src.n))
                imgs_estim = Image(self._im[:, :, indices])

            imgs_estim.save(self.src.mrcs_fileout[istart], overwrite=overwrite)
