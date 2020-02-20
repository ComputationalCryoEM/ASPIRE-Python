"""
This script illustrates how to load CryoEM density map, downsample to a desired resolution,
and show it in a simple 3D view.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from aspire.utils.preprocess import downsample

import mrcfile

logger = logging.getLogger('aspire')

logger.info('This script illustrates how to load CryoEM density map and downsample it.')

# Set the size to downsample
down_size = 30

# Load the map file of a 70S Ribosome and downsample the 3D map to desired resolution.
DATA_DIR = os.path.join(os.path.dirname(__file__), '../src/aspire/data/')
file_path = os.path.join(DATA_DIR, 'clean70SRibosome_vol_65p.mrc')
logger.info(f'Load 3D EM map from {file_path}')
file_in = mrcfile.open(file_path)
vol = file_in.data
file_in.close()

# Downsample the 3D map to a desired resolution.
# The downsampling should be done by the internal function of sim object in future.
# Below we use an alternative implementation to obtain the exact result with Matlab version.
logger.info(f'Downsample to desired grids of {down_size} x {down_size} x {down_size}.')
vol = vol[..., np.newaxis]
vol = downsample(vol, down_size*np.ones(3, dtype=int))

# Save the downsampled 3D map to a file
file_path = 'downsampled_emd.mrc'
logger.info(f'Save the downsampled 3D map to {file_path}.')
file_out = mrcfile.new(file_path, overwrite=True)
file_out.set_data(vol[..., 0])
file_out.close()

# Generate x, y, z coordinates of each density points to visualize
x, y, z = np.meshgrid(np.arange(down_size), np.arange(down_size), np.arange(down_size))
# Calculate the average density from the cubic surfaces
surface = np.zeros((6, 2*down_size*down_size), dtype='float32')
surface[0, ...] = vol[0:2, :, :, 0].flatten()
surface[1, ...] = vol[-3:-1, :, :, 0].flatten()
surface[2, ...] = vol[:, 0:2, :, 0].flatten()
surface[3, ...] = vol[:, -3:-1, :, 0].flatten()
surface[4, ...] = vol[:, :, 0:2, 0].flatten()
surface[5, ...] = vol[:, :, -3:-1, 0].flatten()
avg_value = np.sum(surface)/surface.size
logger.info(f'The average density from 6 cubic surfaces are {avg_value} ')

# Remove the average density and reverse the contrast representation
vol_avg = -(vol - avg_value)

# Generate a mask so that only related density map is shown
mask = abs(vol_avg[..., 0]) > 0.0002

# Plot the density scaled as color
ax = plt.axes(projection='3d')
ax.scatter3D(x[mask], y[mask], z[mask], c=vol_avg[mask, 0].flatten(), cmap='gray')
plt.show()
