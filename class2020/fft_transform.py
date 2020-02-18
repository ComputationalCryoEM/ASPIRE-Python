"""
This script illustrates the FFT/IFFT transformation of 3D map and 2d image
"""

import os
import logging
import numpy as np
import mrcfile
import matplotlib.pyplot as plt

from aspire.utils.fft import centered_fft2, centered_ifft2
from aspire.utils.fft import centered_fft3, centered_ifft3
from aspire.utils.matrix import anorm
from aspire.utils.preprocess import vol2img
from scipy.spatial.transform import Rotation as R
from aspire.basis.ffb_2d import FFBBasis2D

logger = logging.getLogger('aspire')

logger.info('This script illustrates FFT/IFFT transformation of 3D map and 2D image.')

# Load the map file of a 70S Ribosome
DATA_DIR = os.path.join(os.path.dirname(__file__), '../src/aspire/data/')
infile = mrcfile.open(os.path.join(DATA_DIR, 'clean70SRibosome_vol_65p.mrc'))
vol = infile.data
img_size = np.size(vol, 0)
logger.info(f'Load 3D map with the desired grids of {img_size} x {img_size} x {img_size}.')

# Show the examples of 3D FFT and IFFT
logger.info(f'Transform 3D map to Fourier space and transform back.')
# Transform 3D map into Fourier space by FFT
vol_ft = centered_fft3(vol)
# Transform back to 3D map from Fourier space by IFFT
vol_est = centered_ifft3(vol_ft)
# Calculate the normalized RMSE of the estimated density map.
nrmse_vol = np.abs(anorm(vol_est-vol)/anorm(vol))
logger.info(f'Estimated 3D map normalized RMSE using normal FFT is {nrmse_vol}')

# Project the map to image using the convention of ZYZ such as Rot = R(Z,α)R(Y,β)R(Z,γ).
# For more details, see https://www.ccpem.ac.uk/user_help/rotation_conventions.php
angleRot = 0
angleTilt = 90
anglePsi = 0
angles = [[angleRot, angleTilt, anglePsi]]
rots = R.from_euler('ZYZ', angles, degrees=True).as_dcm()
img = vol2img(vol, rots)
logger.info(f'Project the map using ZYZ convention with the angles of {angleRot}, {angleTilt}, {anglePsi}.')

logger.info(f'Transform 2D image to Fourier space and transform back.')
img_ft = centered_fft2(img)
img_est = np.real(centered_ifft2(img_ft))
nrmse_img = anorm(img_est-img)/anorm(img)
logger.info(f'Estimated image normalized RMSE using normal FFT is {nrmse_img}')

# Specify the fast FB basis method for expending the 2D images
logger.info(f'Transform 2D image to Fast FB basis and transform back.')
ffbbasis = FFBBasis2D((img_size, img_size))
coeff = ffbbasis.evaluate_t(img)
img_fb = ffbbasis.evaluate(coeff)

# plot the projected and FFT transformed images
logger.info(f'Output 2D images and related differences.')
idm = 0
plt.subplot(2, 3, 1)
plt.imshow(img[..., 0], cmap='gray')
plt.colorbar(orientation='horizontal')
plt.title('Projected Image')
plt.subplot(2, 3, 2)
plt.imshow(img_est[..., 0], cmap='gray')
plt.colorbar(orientation='horizontal')
plt.title('FFT Estimated Image')
plt.subplot(2, 3, 3)
plt.imshow(img_est[..., 0]-img[..., 0], cmap='gray')
plt.colorbar(orientation='horizontal')
plt.title('FFT Differences')

plt.subplot(2, 3, 4)
plt.imshow(img[..., 0], cmap='gray')
plt.colorbar(orientation='horizontal')
plt.title('Projected Image')
plt.subplot(2, 3, 5)
plt.imshow(img_fb[..., 0], cmap='gray')
plt.colorbar(orientation='horizontal')
plt.title('FFB Estimated Image')
plt.subplot(2, 3, 6)
plt.imshow(img_fb[..., 0]-img[..., 0], cmap='gray')
plt.colorbar(orientation='horizontal')
plt.title('FFB Differences')
plt.show()
