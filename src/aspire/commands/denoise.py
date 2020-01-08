import logging
import click

from aspire.source.relion import RelionSource
from aspire.estimation.noise import WhiteNoiseEstimator
from aspire.estimation.noise import AnisotropicNoiseEstimator
from aspire.basis.ffb_2d import FFBBasis2D
from aspire.denoise.denoise_cov2d import DenoiserCov2D

logger = logging.getLogger('aspire')


@click.command()
@click.option('--starfile_in', required=True,
              help='Path to input starfile')
@click.option('--data_folder', default=None,
              help='Path to mrcs files referenced in starfile')
@click.option('--starfile_out', required=True,
              help='Path to output starfile')
@click.option('--pixel_size', default=1, type=float,
              help='Pixel size of images in starfile')
@click.option('--max_rows', default=None, type=int,
              help='Max. no. of image rows to read from starfile')
@click.option('--max_resolution', default=16, type=int,
              help='Resolution of downsampled images read from starfile')
@click.option('--noise_type', default='Isotropic', type=str,
              help='Noise type for estimation')
@click.option('--ctf_info', default=True, type=bool,
              help='Whether include CTF information')
@click.option('--denoise_method', default='CWF', type=str,
              help='Specified method for denoising 2D images')
def denoise(starfile_in, data_folder, starfile_out, pixel_size, max_rows, max_resolution,
            noise_type, ctf_info, denoise_method):
    """
    Denoise the images and output the clean images using the default CWF method.
    """
    # Create a source object for 2D images
    logger.info(f'Read in images from {starfile_in} and preprocess the images.')
    source = RelionSource(
        starfile_in,
        data_folder, 
        pixel_size=pixel_size,
        max_rows=max_rows
    )

    logger.info(f'Set the resolution to {max_resolution} X {max_resolution}')
    if max_resolution < source.L:
        # Downsample the images
        source.downsample(max_resolution)

    # Specify the fast FB basis method for expending the 2D images
    basis = FFBBasis2D((max_resolution, max_resolution))

    # Estimate the noise of images
    noise_estimator = None
    if noise_type == 'Isotropic':
        logger.info(f'Estimate the noise of images using isotropic method')
        noise_estimator = WhiteNoiseEstimator(source)
    else:
        logger.info(f'Estimate the noise of images using anisotropic method')
        noise_estimator = AnisotropicNoiseEstimator(source)

    # Whiten the noise of images
    logger.info(f'Whiten the noise of images from the noise estimator')
    source.whiten(noise_estimator.filter)
    var_noise = noise_estimator.estimate()
    # source.cache()
    # img_whitened = source.eval_filters(source.images())
    if denoise_method == 'CWF':
        denoise_cov2d = DenoiserCov2D(source, basis, var_noise, ctf_info)
        logger.info(f'Denoise the images using CWF cov2D method.')
        denoise_cov2d.denoise()
        logger.info(f'Output the denoised images.')
        denoise_cov2d.save(starfile_out, batch_size=max_rows)
    else:
        raise NotImplementedError('Currently only covariance Wiener filtering method is supported')


