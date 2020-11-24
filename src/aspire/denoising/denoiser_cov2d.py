import logging

import numpy as np
from numpy.linalg import solve

from aspire.basis import FFBBasis2D
from aspire.covariance import BatchedRotCov2D
from aspire.denoising import Denoiser
from aspire.denoising.denoised_src import DenoisedImageSource
from aspire.optimization import fill_struct
from aspire.utils import mat_to_vec
from aspire.volume import Volume, qr_vols_forward

logger = logging.getLogger(__name__)


def src_wiener_coords(
    sim, mean_vol, eig_vols, lambdas=None, noise_var=0, batch_size=512
):
    """
    Calculate coordinates using Wiener filter
    :param sim: A simulation object containing the images whose coordinates we want.
    :param mean_vol: The mean volume of the source as Volume instance.
    :param eig_vols: The eigenvolumes of the source as Volume instance.
    :param lambdas: The eigenvalues in a K-by-K diagonal matrix (default `eye(K)`).
    :param noise_var: The variance of the noise in the images (default 0).
    :param batch_size: The size of the batches in which to compute the coordinates (default 512).
    :return: A K-by-`src.n` array of coordinates corresponding to the Wiener filter coordinates of each image in sim.

    The coordinates are obtained by the formula
        alpha_s = eig_vols^T H_s ( y_s - P_s mean_vol ) ,

    where P_s is the forward image mapping and y_s is the sth image,
        H_s = Sigma * P_s^T ( P_s Sigma P_s^T + noise_var I )^(-1) ,

    and Sigma is the covariance matrix eig_vols * lambdas * eig_vols^T.
    Note that when noise_var is zero, this reduces to the projecting y_s onto the span of P_s eig_vols.

    # TODO: Find a better place for this functionality other than in utils
    """

    if not isinstance(mean_vol, Volume):
        logger.debug(
            "src_wiener_coords mean_vol should be a Volume instance. Attempt correction."
        )
        if len(mean_vol.shape) == 4 and mean_vol.shape[3] != 1:
            msg = (
                f"Cannot naively convert {mean_vol.shape} to Volume instance."
                "Please change calling code."
            )
            logger.error(msg)
            raise RuntimeError(msg)

        mean_vol = Volume(mean_vol)

    if not isinstance(eig_vols, Volume):
        logger.debug(
            "src_wiener_coords eig_vols should be a Volume instance. Correcting for now."
        )
        eig_vols = Volume(eig_vols)

    if not sim.dtype == mean_vol.dtype == eig_vols.dtype:
        logger.warning(
            "Inconsistent types in src_wiener_coords"
            f" sim {sim.dtype},"
            f" mean_vol {mean_vol.dtype},"
            f" eig_vols {eig_vols.dtype}"
        )

    k = eig_vols.n_vols
    if lambdas is None:
        lambdas = np.eye(k)

    coords = np.zeros((k, sim.n))
    covar_noise = noise_var * np.eye(k)

    for i in range(0, sim.n, batch_size):
        ims = sim.images(i, batch_size)
        batch_n = ims.shape[0]
        ims -= sim.vol_forward(mean_vol, i, batch_n)

        Qs, Rs = qr_vols_forward(sim, i, batch_n, eig_vols, k)

        Q_vecs = mat_to_vec(Qs)

        # RCOPT
        ims = np.moveaxis(ims.data, 0, 2)
        im_vecs = mat_to_vec(ims)

        for j in range(batch_n):
            im_coords = Q_vecs[:, :, j].T @ im_vecs[:, j]
            covar_im = (Rs[:, :, j] @ lambdas @ Rs[:, :, j].T) + covar_noise
            xx = solve(covar_im, im_coords)
            im_coords = lambdas @ Rs[:, :, j].T @ xx
            coords[:, i + j] = im_coords

    return coords


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
            raise NotImplementedError("Currently only fast FB method is supported")
        self.basis = basis
        self.cov2d = None
        self.mean_est = None
        self.covar_est = None

    def denoise(self, covar_opt=None, batch_size=512):
        """
         Build covariance matrix of 2D images and return a new ImageSource object

        :param covar_opt: The option list for building Cov2D matrix
        :param batch_size: The batch size for processing images
        :return: A `DenoisedImageSource` object with the specified denoising object
        """

        # Initialize the rotationally invariant covariance matrix of 2D images
        # A fixed batch size is used to go through each image
        self.cov2d = BatchedRotCov2D(self.src, self.basis, batch_size=batch_size)

        default_opt = {
            "shrinker": "frobenius_norm",
            "verbose": 0,
            "max_iter": 250,
            "iter_callback": [],
            "store_iterates": False,
            "rel_tolerance": 1e-12,
            "precision": self.dtype,
        }

        covar_opt = fill_struct(covar_opt, default_opt)
        # Calculate the mean and covariance for the rotationally invariant covariance matrix of 2D images
        self.mean_est = self.cov2d.get_mean()

        self.covar_est = self.cov2d.get_covar(
            noise_var=self.var_noise, mean_coeff=self.mean_est, covar_est_opt=covar_opt
        )

        return DenoisedImageSource(self.src, self)

    def images(self, istart=0, batch_size=512):
        """
        Obtain a batch size of 2D images after denosing by Cov2D method

        :param istart: the index of starting image
        :param batch_size: The batch size for processing images
        :return: an `Image` object with denoised images
        """
        src = self.src

        # Denoise one batch size of 2D images using the SPCAs from the rotationally invariant covariance matrix
        img_start = istart
        img_end = min(istart + batch_size, src.n)
        imgs_noise = src.images(img_start, batch_size)
        coeffs_noise = self.basis.evaluate_t(imgs_noise.data)
        logger.info(
            f"Estimating Cov2D coefficients for images from {img_start} to {img_end-1}"
        )
        coeffs_estim = self.cov2d.get_cwf_coeffs(
            coeffs_noise,
            self.cov2d.ctf_fb,
            self.cov2d.ctf_idx[img_start:img_end],
            mean_coeff=self.mean_est,
            covar_coeff=self.covar_est,
            noise_var=self.var_noise,
        )

        # Convert Fourier-Bessel coefficients back into 2D images
        logger.info("Converting Cov2D coefficients back to 2D images")
        imgs_denoised = self.basis.evaluate(coeffs_estim)

        return imgs_denoised
