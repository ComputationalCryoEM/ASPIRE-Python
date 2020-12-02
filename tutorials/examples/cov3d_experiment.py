#!/usr/bin/env  python
"""
This script illustrates Cov3D analysis using experimental dataset
"""

from aspire.basis import FBBasis3D
from aspire.covariance import CovarianceEstimator
from aspire.noise import AnisotropicNoiseEstimator
from aspire.reconstruction import MeanEstimator
from aspire.source.relion import RelionSource

# Set input path and files and initialize other parameters
# DATA_FOLDER = '/path/to/untarred/empiar/dataset/'
DATA_FOLDER = "/scratch/gpfs/junchaox/yan_wu"
# STARFILE = "/path/to/untarred/empiar/dataset/input.star"
STARFILE = "/scratch/gpfs/junchaox/yan_wu/All_class001_r1_ct1_data.star"
PIXEL_SIZE = 5.0
MAX_ROWS = 1024
MAX_RESOLUTION = 8
CG_TOL = 1e-5

# Create a source object for 2D images
print(f'Read in images from {STARFILE} and preprocess the images.')
source = RelionSource(
    STARFILE, data_folder=DATA_FOLDER, pixel_size=PIXEL_SIZE, max_rows=MAX_ROWS
)

# Downsample the images
print(f'Set the resolution to {MAX_RESOLUTION} X {MAX_RESOLUTION}')
if MAX_RESOLUTION < source.L:
    source.downsample(MAX_RESOLUTION)

# Estimate the noise of images
print(f'Estimate the noise of images using anisotropic method')
noise_estimator = AnisotropicNoiseEstimator(source, batchSize=512)

# Whiten the noise of images
print(f'Whiten the noise of images from the noise estimator')
source.whiten(noise_estimator.filter)
# Estimate the noise variance. This is needed for the covariance estimation step below.
noise_variance = noise_estimator.estimate()
print(f"Noise Variance = {noise_variance}")

# Specify the fast FB basis method for expending the 2D images
basis = FBBasis3D((MAX_RESOLUTION, MAX_RESOLUTION, MAX_RESOLUTION))

mean_estimator = MeanEstimator(source, basis, batch_size=512)
mean_est = mean_estimator.estimate()

# Passing in a mean_kernel argument to the following constructor speeds up some calculations
covar_estimator = CovarianceEstimator(
    source, basis, mean_kernel=mean_estimator.kernel
)
covar_estimator.estimate(mean_est, noise_variance, tol=CG_TOL)
