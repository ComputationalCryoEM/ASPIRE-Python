"""
This script illustrates the rotation and projection of 3D density map to 2D images
from a list of specified angles.
"""

import os
import logging
import numpy as np
import mrcfile
import matplotlib.pyplot as plt

from aspire.utils.preprocess import downsample
from scipy.spatial.transform import Rotation as R
from aspire.utils.preprocess import vol2img

logger = logging.getLogger('aspire')

logger.info('This script illustrates the rotation and projection of 3D density map to 2D image.')

# Load the map file of a 70S Ribosome
DATA_DIR = os.path.join(os.path.dirname(__file__), '../src/aspire/data/')
infile = mrcfile.open(os.path.join(DATA_DIR, 'clean70SRibosome_vol_65p.mrc'))
vol = infile.data
img_size = np.size(vol, 0)
logger.info(f'Load 3D map with the desired grids of {img_size} x {img_size} x {img_size}.')

# Rotate and project to specified orientation angles.
# A general rotation of a rigid body (e.g. a 3D reconstruction) can be described as
# a series of 3 rotations about the axes of the coordinate system. The Relion software
# is using the convention of ZYZ such as Rot = R(Z,α)R(Y,β)R(Z,γ). For more details,
# see https://www.ccpem.ac.uk/user_help/rotation_conventions.php

angleRot = 0
angleTilt = 90
anglePsi = 0
angles = [[angleRot, angleTilt, anglePsi]]
rots = R.from_euler('ZYZ', angles, degrees=True).as_dcm()
imgs_clean = vol2img(vol, rots)
logger.info(f'Project the map using ZYZ convention with the angles of {angleRot}, {angleTilt}, {anglePsi}.')

# plot the rotated and projected image
logger.info(f'Output 2D image at specified angles.')
plt.subplot(1, 1, 1)
plt.imshow(imgs_clean[..., 0], cmap='gray')
plt.colorbar()
plt.title('Clean Image')
plt.show()

