import logging
from copy import deepcopy

import numpy as np
from numpy.linalg import solve

from aspire.basis import FFBBasis2D
from aspire.covariance import BatchedRotCov2D
from aspire.denoising import Denoiser
from aspire.noise import WhiteNoiseEstimator
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
        ims = sim.images[i : i + batch_size]
        batch_n = ims.shape[0]
        ims -= sim.vol_forward(mean_vol, i, batch_n)

        Qs, Rs = qr_vols_forward(sim, i, batch_n, eig_vols, k)

        Q_vecs = mat_to_vec(Qs)

        # RCOPT
        ims = np.moveaxis(ims.asnumpy(), 0, 2)
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

    # Default options for cov2d configuration.
    default_opt = {
        "shrinker": "frobenius_norm",
        "verbose": 0,
        "max_iter": 250,
        "iter_callback": [],
        "store_iterates": False,
        "rel_tolerance": 1e-12,
    }

    def __init__(self, src, basis=None, var_noise=None, batch_size=512, covar_opt=None):
        """
        Initialize an object for denoising 2D images using Cov2D method

        :param src: The source object of 2D images with metadata
        :param basis: The basis method to expand 2D images
        :param var_noise: The estimated variance of noise
        :param batch_size: Integer batch size for processing images.
            Defaults to 512.
        :param covar_opt: Optional dictionary of option overides for Cov2D.
            Provided options will supersede defaults in `DenoiserCov2D.default_opt`.
        """

        super().__init__(src)
        self.batch_size = int(batch_size)

        # When var_noise is not specfically over-ridden,
        #   recompute it now. See #496.
        if var_noise is None:
            logger.info("Estimating noise of images using WhiteNoiseEstimator")
            noise_estimator = WhiteNoiseEstimator(src)
            var_noise = noise_estimator.estimate()
            logger.info(f"Estimated Noise Variance: {var_noise}")
        self.var_noise = var_noise

        if basis is None:
            basis = FFBBasis2D((self.src.L, self.src.L), dtype=src.dtype)

        self.basis = basis
        self.cov2d = None
        self.mean_est = None
        self.covar_est = None

        # Create a local copy of the default options.
        default_opt = deepcopy(self.default_opt)
        # Assign the dtype corresponding to this instance.
        default_opt["precision"] = self.dtype
        # Apply any overrides provided by the user.
        self.covar_opt = fill_struct(covar_opt, default_opt)

        # Initialize the rotationally invariant covariance matrix of 2D images
        # A fixed batch_size is used to loop through image stack.
        self.cov2d = BatchedRotCov2D(self.src, self.basis, batch_size=batch_size)

    def build_denoiser(self):
        """
        Build estimated mean and covariance matrix of 2D images.

        This method should be computed once, on first `images` access.
        """

        if self.covar_est is not None:
            return

        logger.info(f"Building mean estimate for {len(self.src)} images.")
        self.mean_est = self.cov2d.get_mean()

        logger.info(f"Building covariance estimates for {len(self.src)} images.")
        self.covar_est = self.cov2d.get_covar(
            noise_var=self.var_noise,
            mean_coef=self.mean_est,
            covar_est_opt=self.covar_opt,
        )

    def _denoise(self, indices):
        """
        Compute denoised 2D images corresponding to `indices`.

        :return: `Image` object containing denoised images.
        """

        # Lazy evaluate estimates on access.
        # `build_denoiser` internally guards to compute once.
        self.build_denoiser()

        # Denoise requested `indices` selection of 2D images.
        imgs_noise = self.src.images[indices]

        coefs_noise = self.basis.evaluate_t(imgs_noise)
        logger.debug(f"Estimating Cov2D coefficients for {imgs_noise.n_images} images.")
        coefs_estim = self.cov2d.get_cwf_coefs(
            coefs_noise,
            self.cov2d.ctf_basis,
            self.cov2d.ctf_idx[indices],
            mean_coef=self.mean_est,
            covar_coef=self.covar_est,
            noise_var=self.var_noise,
        )

        # Convert Fourier-Bessel coefficients back into 2D images
        logger.info("Converting Cov2D coefficients back to 2D images")
        imgs_denoised = self.basis.evaluate(coefs_estim)

        return imgs_denoised
