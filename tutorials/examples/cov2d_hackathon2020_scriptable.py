"""
This script illustrates the covariance Wiener filtering functionality of the
ASPIRE, implemented by estimating the covariance of the unfiltered
images in a Fourier-Bessel basis and applying the Wiener filter induced by
that covariance matrix. The results can be reproduced exactly to the Matlab
version if the same methods of generating random numbers are used.
"""

import argparse
import os
import logging
import numpy as np
import mrcfile

from aspire.source.simulation import Simulation
from aspire.basis.ffb_2d import FFBBasis2D
from aspire.utils.filters import RadialCTFFilter
from aspire.utils.preprocess import downsample
from aspire.utils.coor_trans import qrand_rots
from aspire.utils.preprocess import vol2img
from aspire.image import Image
from aspire.utils.matrix import anorm
from aspire.utils.matlab_compat import randn
from aspire.estimation.covar2d import RotCov2D
from aspire.utils.profiler_helper import prof_sandwich
from aspire.nfft import all_backends

backend = all_backends()[0]

logger = logging.getLogger('aspire')

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/')

def main(img_size=64, num_imgs=1024):
    """
    This script illustrates 2D covariance Wiener filtering functionality in ASPIRE package.

    :param img_size: Set the sizes of images to img_size x img_size, defaults 64 x 64.
    :param num_imgs: Set the total number of images generated from the 3D map, defaults 1024.

    :return: Estimated images normalized RMS.
    """

    logger.info('This script illustrates 2D covariance Wiener filtering functionality in ASPIRE package.')

    # Set the number of 3D maps
    num_maps = 1

    # Set the signal-noise ratio
    sn_ratio = 1

    # Specify the CTF parameters
    pixel_size = 5                   # Pixel size of the images (in angstroms).
    voltage = 200                    # Voltage (in KV)
    defocus_min = 1.5e4              # Minimum defocus value (in angstroms).
    defocus_max = 2.5e4              # Maximum defocus value (in angstroms).
    defocus_ct = 7                   # Number of defocus groups.
    Cs = 2.0                         # Spherical aberration
    alpha = 0.1                      # Amplitude contrast

    logger.info('Initialize simulation object and CTF filters.')
    # Create filters
    filters = [RadialCTFFilter(pixel_size, voltage, defocus=d, Cs=2.0, alpha=0.1)
               for d in np.linspace(defocus_min, defocus_max, defocus_ct)]

    # Load the map file of a 70S Ribosome and downsample the 3D map to desired resolution.
    # The downsampling should be done by the internal function of sim object in future.
    # Below we use alternative implementation to obtain the exact result with Matlab version.
    logger.info(f'Load 3D map and downsample 3D map to desired grids '
                f'of {img_size} x {img_size} x {img_size}.')
    infile = mrcfile.open(os.path.join(DATA_DIR, 'clean70SRibosome_vol_65p.mrc'))
    vols = infile.data
    vols = vols[..., np.newaxis]
    vols = downsample(vols, (img_size*np.ones(3, dtype=int)))

    # Create a simulation object with specified filters and the downsampled 3D map
    logger.info('Use downsampled map to creat simulation object.')
    sim = Simulation(
        L=img_size,
        n=num_imgs,
        vols=vols,
        C=num_maps,
        filters=filters
    )

    # Initialize a class object of the fast FB basis method for expending the 2D images
    ffbbasis = FFBBasis2D((img_size, img_size))

    # Generate 2D clean images from input 3D map. The following statement can be used from the sim object:
    # imgs_clean = sim.clean_images(start=0, num=num_imgs)
    # To be consistent with the Matlab version in the numbers, we need to use the statements as below:
    logger.info('Generate random distributed rotation angles and obtain corresponding 2D clean images.')
    rots = qrand_rots(num_imgs, seed=0)
    imgs_clean = vol2img(sim.vols[..., 0], rots)

    # Assign the CTF information and index for each image
    h_idx = np.array([filters.index(f) for f in sim.filters])

    # Evaluate CTF in the 8X8 FB basis
    h_ctf_fb = [filt.fb_mat(ffbbasis) for filt in filters]

    # Apply the CTF to the clean images.
    logger.info('Apply CTF filters to clean images.')
    imgs_ctf_clean = Image(sim.eval_filters(imgs_clean))
    sim.cache(imgs_ctf_clean)

    # imgs_ctf_clean is an Image object. Convert to numpy array for subsequent statements
    imgs_ctf_clean = imgs_ctf_clean.asnumpy()

    # Apply the noise at the desired singal-noise ratio to the filtered clean images
    logger.info('Apply noise filters to clean images.')
    power_clean = anorm(imgs_ctf_clean)**2/np.size(imgs_ctf_clean)
    noise_var = power_clean/sn_ratio
    imgs_noise = imgs_ctf_clean + np.sqrt(noise_var)*randn(img_size, img_size, num_imgs, seed=0)

    # Expand the noisy images in the Fourier-Bessel basis. This
    # can be done exactly (that is, up to numerical precision) using the
    # `basis.expand` function, but for our purposes, an approximation will do.
    # Since the basis is close to orthonormal, we may approximate the exact
    # expansion by applying the adjoint of the evaluation mapping using
    # `basis.evaluate_t`.
    logger.info('Get coefficients of noisy images in FFB basis.')

    #  This part can be improved using GPU
    with prof_sandwich(event_name='evaluate_t', verbose=False, suffix=f's{img_size}_n{num_imgs}_{backend}'):
        coeffs_noise = ffbbasis.evaluate_t(imgs_noise)

    # Estimate mean and covariance for noise images with CTF and shrink method.
    # We now estimate the mean and covariance from the Fourier-Bessel
    # coefficients of the noisy, filtered images. These functions take into
    # account the filters applied to each image to undo their effect on the
    # estimates. For the covariance estimation, the additional information of
    # the estimated mean and the variance of the noise are needed. Again, the
    # covariance matrix estimate is provided in block diagonal form.

    #  This part can be improved using GPU
    cov2d = RotCov2D(ffbbasis)
    covar_opt = {'shrinker': 'frobenius_norm', 'verbose': 0, 'max_iter': 250,
                 'iter_callback': [], 'store_iterates': False, 'rel_tolerance': 1e-12,
                 'precision': 'float64', 'preconditioner': 'identity'}
    #  This part can be improved using GPU
    with prof_sandwich(event_name='get_mean', verbose=False, suffix=f's{img_size}_n{num_imgs}_{backend}'):
        mean_coeffs_est = cov2d.get_mean(coeffs_noise, h_ctf_fb, h_idx)
    #  This part can be improved using GPU
    with prof_sandwich(event_name='get_covar', verbose=False, suffix=f's{img_size}_n{num_imgs}_{backend}'):
        covar_coeffs_est = cov2d.get_covar(coeffs_noise, h_ctf_fb, h_idx, mean_coeffs_est,
                                           noise_var=noise_var, covar_est_opt=covar_opt)

    # Estimate the Fourier-Bessel coefficients of the underlying images using a
    # Wiener filter. This Wiener filter is calculated from the estimated mean,
    # covariance, and the variance of the noise. The resulting estimator has
    # the lowest expected mean square error out of all linear estimators.
    logger.info('Get the CWF coefficients of noising images.')
    #  This part can be improved using GPU
    with prof_sandwich(event_name='get_cwf_coeffs', verbose=False, suffix=f's{img_size}_n{num_imgs}_{backend}'):
        coeffs_est = cov2d.get_cwf_coeffs(coeffs_noise, h_ctf_fb, h_idx,
                                         mean_coeff=mean_coeffs_est,
                                         covar_coeff=covar_coeffs_est, noise_var=noise_var)

    # Convert Fourier-Bessel coefficients back into 2D images
    #  This part can be improved using GPU
    with prof_sandwich(event_name='evaluate', verbose=False, suffix=f's{img_size}_n{num_imgs}_{backend}'):
        imgs_est = ffbbasis.evaluate(coeffs_est)

    # Calculate the normalized RMSE of the estimated images.
    nrmse_ims = anorm(imgs_est-imgs_clean)/anorm(imgs_clean)

    logger.info(f'Estimated images normalized RMSE: {nrmse_ims}')

    return nrmse_ims

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='This script illustrates 2D covariance Wiener'
        ' filtering functionality in ASPIRE package.')
    parser.add_argument('-s', '--img_size', type=int, default=64,
                        help='Sizes of images, img_size x img_size, defaults 64 x 64.')
    parser.add_argument('-n', '--num_imgs', type=int, default=1024,
                        help='Total number of images, defaults 1024.')
    args = parser.parse_args()
    main(args.img_size, args.num_imgs)
