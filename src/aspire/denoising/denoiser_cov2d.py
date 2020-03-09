import logging
import numpy as np
import os.path
import pandas as pd

from aspire.denoising import Denoiser
from aspire.utils.filters import RadialCTFFilter
from aspire.utils.blk_diag_func import blk_diag_partition
from aspire.utils.blk_diag_func import blk_diag_eye
from aspire.basis.ffb_2d import FFBBasis2D
from aspire.estimation.covar2d import BatchedRotCov2D
from aspire.utils.optimize import fill_struct
from aspire.image import Image
from aspire.io.starfile import StarFile, StarFileBlock

logger = logging.getLogger(__name__)


class DenoiserCov2D(Denoiser):
    """
    Define a derived class for denoising 2D images using Cov2D method
    """
    def __init__(self, src, basis, var_noise, ctf_info, star_file):
        """
        Initialize an object for denoising 2D images using Cov2D method

        :param src: The source object of 2D images with metadata
        :param basis: The basis method to expand 2D images
        :param var_noise: The estimated variance of noise
        :param ctf_info: Whether the CTF filters are included
        :param star_file: the star file path and name to output
        """
        super().__init__(src)
        self.var_noise = var_noise
        if not isinstance(basis, FFBBasis2D):
            raise NotImplementedError('Currently only fast FB method is supported')
        self.basis = basis
        self.ctf_info = ctf_info
        self.star_file = star_file
        self.starfile_path = os.path.dirname(star_file)
        self.starfile_name = os.path.basename(star_file)

        self.ctf_idx = None
        self.ctf_fb = None
        # Assign the CTF information and index for each image
        if ctf_info and not (src.filters is None):
            logger.info(f'Represent CTF filters in FB basis')
        else:
            logger.info(f'CTF filters are not included in Cov2D denoising')
            # set all CTF filters to an identity filter
            self.ctf_idx = np.zeros(self.nimg, dtype=int)
            self.ctf_fb = [blk_diag_eye(blk_diag_partition(RadialCTFFilter().fb_mat(basis)))]

        self.metadata = src._metadata.copy(deep=True)
        self._write_star_file()
        self.mrcs_out = self._get_mrcs_out()

    def denoise(self, covar_opt=None, overwrite=False):
        """
        Denoiser 2D images using Cov2D method

        :param covar_opt: The option list for building Cov2D matrix
        :param overwrite: The option to owerwrite mrcs files for denoised images or not
        """
        src = self.src

        # Initialize the rotationally invariant covariance matrix of 2D images
        # A fixed batch size is used to go through each image
        cov2d = BatchedRotCov2D(self.src, self.basis, batch_size=512)
        if not self.ctf_info:
            cov2d.ctf_idx = self.ctf_idx
            cov2d.ctf_fb = self.ctf_fb

        default_opt = {'shrinker': 'frobenius_norm', 'verbose': 0, 'max_iter': 250,
            'iter_callback': [], 'store_iterates': False, 'rel_tolerance': 1e-12,
            'precision': 'float64'}
        covar_opt = fill_struct(covar_opt, default_opt)
        # Calculate the mean and covariance for the rotationally invariant covariance matrix of 2D images
        mean_coeffs_est = cov2d.get_mean()

        covar_coeffs_est = cov2d.get_covar(noise_var=self.var_noise, mean_coeff=mean_coeffs_est,
                                           covar_est_opt=covar_opt)
        # Denoise the 2D images using the SPCAs from the rotationally invariant covariance matrix
        # The batch size varies to fit the number of noisy images in each mrcs file
        _, batch_size = src.group_unique_hist

        img_start = 0
        for ibatch in range(0, len(batch_size)):
            imgs_noise = src.images(img_start, batch_size[ibatch])
            coeffs_noise = self.basis.evaluate_t(imgs_noise.data)
            logger.info(f'Estimating Cov2D coefficients for image batch {ibatch} of size {batch_size[ibatch]}')
            coeffs_estim = cov2d.get_cwf_coeffs(coeffs_noise, cov2d.ctf_fb,
                                                cov2d.ctf_idx[img_start:img_start+batch_size[ibatch]],
                                                mean_coeff=mean_coeffs_est, covar_coeff=covar_coeffs_est,
                                                noise_var=self.var_noise)

            # Convert Fourier-Bessel coefficients back into 2D images
            logger.info(f'Converting Cov2D coefficients back to 2D images')
            imgs_estim = self.basis.evaluate(coeffs_estim)
            imgs_out = Image(imgs_estim)
            mrcs_file = os.path.join(self.src.proj_folder, self.starfile_path, self.mrcs_out[img_start])
            logger.info(f'Saving denoised images to {mrcs_file}')
            imgs_out.save(mrcs_file, overwrite=overwrite)
            img_start += batch_size[ibatch]

    def _write_star_file(self):
        """
        Save a new star file with field for denoised images
        """

        self.metadata['_rlnDenoisedImageName'] = pd.Series('', index=self.metadata.index)
        self.metadata = self.metadata.drop(['__mrc_filename', '__mrc_index', '__mrc_filepath',
                                            '__filter', '__filter_indices'], axis=1)
        df = self.metadata
        for idx in range(len(df.index)):
            img_prefix = df['_rlnImageName'][idx].split('@')[0]
            mrcs_filein = df['_rlnImageName'][idx].split('/')[-1]
            mrcs_fileout = mrcs_filein.split('.')[0] + f'_denoised.mrcs'
            mrcs_filepath = img_prefix + '@' + os.path.join(self.starfile_path, mrcs_fileout)
            df.loc[idx, '_rlnDenoisedImageName'] = mrcs_filepath

        with open(os.path.join(self.src.proj_folder, self.star_file), 'w') as f:
            starfile = StarFile(blocks=[StarFileBlock(loops=[df])])
            starfile.save(f)

    def _get_mrcs_out(self):
        """
        Generate a name list of output mrcs files for denoised images
        """
        df = self.metadata
        mrcs_fileout = []
        for idx in range(len(df.index)):
            mrcs_fileout.append(df['_rlnDenoisedImageName'][idx].split('/')[-1])
        return mrcs_fileout
