"""
This script illustrates several expansion methods for 2D images developed in ASPIRE package
based on the basis functions of  Fourier Bessel (FB) and Prolate Spheroidal Wave Function (PSWF).
"""

import os
import logging

import numpy as np

from aspire.basis.fb_2d import FBBasis2D
from aspire.basis.ffb_2d import FFBBasis2D
from aspire.basis.pswf_2d import PSWFBasis2D
from aspire.utils.matrix import anorm
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Load the images from numpy array
DATA_DIR = os.path.join(os.path.dirname(__file__), '../tests/saved_test_data')
org_images = np.load(os.path.join(DATA_DIR, 'example_data_np_array.npy'))

# Set the sizes of images 129X129
img_size = 129

#
# Specify the normal FB basis method for expending the 2D images
#
fb_basis = FBBasis2D((img_size, img_size))
# Get the expansion coefficients based on FB basis
fb_coeffs = fb_basis.evaluate_t(org_images)
# Reconstruct images from the expansion coefficients based on FB basis
fb_images = fb_basis.evaluate(fb_coeffs)
logger.info('Finish normal FB expansion and reconstruction.')
# Calculate the maximum difference between the FB estimated images to the original images
fb_maxdiff = np.max(abs(fb_images-org_images))
print(f'Maximum difference between the FB estimated images to the original images: {fb_maxdiff}')
# Calculate the normalized RMSE of the FB estimated images
nrmse_ims = anorm(fb_images-org_images)/anorm(org_images)
print(f'FB estimated images normalized RMSE: {nrmse_ims}')
# plot the first images
plt.subplot(3, 3, 1)
plt.imshow(np.real(org_images[..., 0]), cmap='gray')
plt.title('Original')
plt.subplot(3, 3, 2)
plt.imshow(np.real(fb_images[..., 0]), cmap='gray')
plt.title('FB Image')
plt.subplot(3, 3, 3)
plt.imshow(np.real(org_images[..., 0] - fb_images[..., 0]), cmap='gray')
plt.title('Differences')

#
# Specify the fast FB basis method for expending the 2D images
#
ffb_basis = FFBBasis2D((img_size, img_size))
# Get the expansion coefficients based on fast FB basis
ffb_coeffs = ffb_basis.evaluate_t(org_images)
# Reconstruct images from the expansion coefficients based on fast FB basis
ffb_images = ffb_basis.evaluate(ffb_coeffs)
logger.info('Finish Fast FB expansion and reconstruction.')
# Calculate the maximum difference between the fast FB estimated images to the original images
ffb_maxdiff = np.max(abs(ffb_images-org_images))
print(f'Maximum value of the differences from the FFB estimated images to the original images: {ffb_maxdiff}')
# Calculate the normalized RMSE of the estimated images
nrmse_ims = anorm(ffb_images-org_images)/anorm(org_images)
print(f'FFB Estimated images normalized RMSE: {nrmse_ims}')
# plot the first images
plt.subplot(3, 3, 4)
plt.imshow(np.real(org_images[..., 0]), cmap='gray')
plt.title('Original')
plt.subplot(3, 3, 5)
plt.imshow(np.real(ffb_images[..., 0]), cmap='gray')
plt.title('FFB Image')
plt.subplot(3, 3, 6)
plt.imshow(np.real(org_images[..., 0] - ffb_images[..., 0]), cmap='gray')
plt.title('Differences')
#
# Specify the direct PSWF basis method for expending the 2D images
#
img_size = 64
pswf_basis = PSWFBasis2D(img_size, 1.0, 1.0, 0)
# Get the expansion coefficients based on direct PSWF basis
pswf_coeffs = pswf_basis.evaluate_t(org_images)
# Reconstruct images from the expansion coefficients based on direct PSWF basis
pswf_images = pswf_basis.evaluate(pswf_coeffs)
logger.info('Finish direct PSWF expansion and reconstruction.')
# Calculate the maximum difference between the direct PSWF estimated images to the original images
pswf_maxdiff = np.max(abs(pswf_images-org_images))
print(f'Maximum value of the differences from the PSWF estimated images to the original images: {pswf_maxdiff}')
# Calculate the normalized RMSE of the estimated images
nrmse_ims = anorm(pswf_images-org_images)/anorm(org_images)
print(f'PSWF Estimated images normalized RMSE: {nrmse_ims}')
# plot the first images
plt.subplot(3, 3, 7)
plt.imshow(np.real(org_images[..., 0]), cmap='gray')
plt.title('Original')
plt.subplot(3, 3, 8)
plt.imshow(np.real(pswf_images[..., 0]), cmap='gray')
plt.title('PSWF Image')
plt.subplot(3, 3, 9)
plt.imshow(np.real(org_images[..., 0] - pswf_images[..., 0]), cmap='gray')
plt.title('Differences')
plt.show()

