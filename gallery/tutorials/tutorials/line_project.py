"""
Radon Transform
=======================================

This tutorial will demonstrate how to set up and use ASPIRE's ``Sinogram`` and ``Image`` class methods to apply forward and backward projections using Radon and Inverse-Radon Transforms.
This document is designed to provide you with a step-by-step tutorial on how to use the corresponding classes and methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from aspire.image import Image
from aspire.utils import grid_2d
from aspire.sinogram import Sinogram  

# %%                                                                                                                                                                                                                  
# Overview                                                                                                                                                                                                            
# --------
# For our demonstration, we will first illustrate the forward ``project`` method within the ``Image`` class. Let's begin by processing an image from ``skimage`` and applying a mask.

# loading the image
image = data.camera().astype(np.float64)
img_size = image.shape[0]
image_without_mask = Image(image)

# masking the image for the line projections
grid = grid_2d(img_size, normalized=True, shifted=True)
mask = grid["r"] < 1
masked_image = image * mask
image_camera = Image(masked_image)

# plotting
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].imshow(image_without_mask.asnumpy()[0], cmap = 'gray', aspect = 'auto')
axes[0].set_title('Camera Man')
image = axes[1].imshow(image_camera.asnumpy()[0], cmap = 'gray', aspect = 'auto')
axes[1].set_title('Masked Camera Man')
plt.tight_layout()
plt.show()


# %%                                                                                                                                                                                                                 
# Forward Projection
# --------
# Next, let's apply the ``project`` method from ASPIRE's ``Image`` class.
# To use this method, we must first convert our data into an ``Image`` and pass in an input which will be a set of angles in radians. 
# We will begin by performing a couple line projection and visualizing the result.

# sinogram with five projections
angles_5 = np.linspace(0, 360, 5, endpoint=False)
rads_5 = angles_5 / 180.0 * np.pi
sinogram_five = image_camera.project(rads_5)
angles_to_plot = [0, 72, 216, 288]

# create figure and axis for the original image with line projections
fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(masked_image, cmap='gray')
ax.set_title('Camera Man with Line Projections')

# plotting the angle of the line projection 
for angle in angles_to_plot:
    rad_angle = np.pi * angle / 180
    ax.axline((img_size // 2, img_size // 2), slope=np.tan(rad_angle), color='r')

plt.show()    
    
n_cols = 2
n_rows = int(np.ceil(len(angles_to_plot) / n_cols)) 

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6))
axes = axes.flatten()

# grapphing the individual line projections
for i, angle in enumerate(angles_to_plot):
    axes[i].plot(sinogram_five.asnumpy()[0, :, int(angle * img_size / 360)], color='red')
    axes[i].set_title(f"Line Projection at {angle}°")
    axes[i].set_xlim([0, 4])
    axes[i].set_ylim([0, np.max(sinogram_five.asnumpy())])

plt.tight_layout()
plt.show()

# %%                                                                                                                                                                                                                 
# Results with Five Projections
# --------                                                                                                                                                                                                           
# As we can see below, the ``project`` method is able to take five different projections and output an ASPIRE ``Sinogram`` object.
# It's important to note that the more projections we take, the better we can construct our image when we apply the ``backproject`` method on the ``Sinogram`` Object.

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].imshow(masked_image, cmap = 'gray', aspect = 'auto')
axes[0].set_title('Original Camera Man Image')
image = axes[1].imshow(sinogram_five.asnumpy()[0].T, cmap = 'gray', aspect = 'auto')
axes[1].set_title('Sinogram of the Original Image')
axes[1].set_xlabel('Projection Angles')
axes[1].set_ylabel('Pixels')
plt.tight_layout()
plt.show()


# %%                                                                                                                                                                                                                 
# Updated Forward Projection
# --------                                                                                                                                                                                                           
# Now, let's take more line projections and create a new ``Sinogram``.
# We're going to use 90 different projections and output our Sinogram.
# Just like with ``sinogram_five``, we'll apply the same process as before and output the respective graphs.

# sinogram with five projections                                                                                                                                                                                     
angles_90 = np.linspace(0, 360, 90, endpoint=False)
rads_90 = angles_90 / 180.0 * np.pi
sinogram_90 = image_camera.project(rads_90)
angles_to_plot = [24, 154, 189, 227]

# create figure and axis for the original image with line projections                                                                                                                                                
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(masked_image, cmap='gray')
ax.set_title('Camera Man with Line Projections')

# plotting the angle of the line projection                                                                                                                                                                          
for angle in angles_to_plot:
    rad_angle = np.pi * angle / 180
    ax.axline((img_size // 2, img_size // 2), slope=np.tan(rad_angle), color='r')

plt.show()

n_cols = 2
n_rows = int(np.ceil(len(angles_to_plot) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6))
axes = axes.flatten()

# grapphing the individual line projections
for i, angle in enumerate(angles_to_plot):
    axes[i].plot(sinogram_90.asnumpy()[0, :, int(angle * img_size / 360)], color='red')
    axes[i].set_title(f"Line Projection at {angle}°")
    axes[i].set_xlim([0, 90])
    axes[i].set_ylim([0, np.max(sinogram_90.asnumpy())])

plt.tight_layout()
plt.show()

# %%                                                                                                                                                                                                                 
# Results with 90 Line Projections
# --------                                                                                                                                                                                                           
# As we can see below, our current ``Sinogram`` is less blurry than before.
# By taking more line projections on the ``Image``, we're able to output a more refined ``Sinogram``.
# This help us with reconstructing our image through the ``backproject`` method.

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].imshow(masked_image, cmap = 'gray', aspect = 'auto')
axes[0].set_title('Original Camera Man Image')
image = axes[1].imshow(sinogram_90.asnumpy()[0].T, cmap = 'gray', aspect = 'auto')
axes[1].set_title('Sinogram of the Original Image')
axes[1].set_xlabel('Projection Angles')
axes[1].set_ylabel('Pixels')
plt.tight_layout()
plt.show()


# %%                                                                                                                                                                                                                 
# Final Sinogram                                                                                                                                                                                           
# --------                                                                                                                                                                                                           
# As mentioned previously, we can transform our line projections back into the image through reconstruction techniques such as the inverse Radon transform.
# To wrap up the ``project`` method in the ASPIRE ``Image`` class,
# let's take 180 different line projections, visualize what the projection looks like from a specific angle on the image, and plot the respective graphs. 

# sinogram with 180 projections                                                                                                                                                                                     
angles_180 = np.linspace(0, 360, 180, endpoint=False)
rads_180 = angles_180 / 180.0 * np.pi
sinogram_180 = image_camera.project(rads_180)
angles_to_plot = [34, 88, 180, 343]

# create figure and axis for the original image with line projections                                                                                                                                                
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(masked_image, cmap='gray')
ax.set_title('Camera Man with Line Projections')
y = sinogram_180.asnumpy()[0, 0, :]
y = - (y - np.min(y)) / (np.max(y) - np.min(y)) / 2
x = np.linspace(-.5, .5, len(y), endpoint = True)
pts = np.vstack((x, y)).T

def rotate_points(pts, ang):
    r = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    return r @ pts.T

rot_points = rotate_points(pts, angles_180[0] * np.pi / 180).T
rot_points = (rot_points + 0.5) * len(y)
ax.plot(rot_points[:, 0], rot_points[:, 1], color='b')
ax.axline((len(y)//2, len(y)//2-1), slope=np.arctan(angles_180[0] * np.pi / 180), color='r')

ax.set_aspect('equal', 'box')
ax.set_xlim(0, masked_image.shape[1])
ax.set_ylim(masked_image.shape[0], 0)

plt.tight_layout(pad=0)
plt.show()

n_cols = 2
n_rows = int(np.ceil(len(angles_to_plot) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6))
axes = axes.flatten()

for i, angle in enumerate(angles_to_plot):
    axes[i].plot(sinogram_180.asnumpy()[0, :, int(angle * img_size / 360)], color='red')
    axes[i].set_title(f"Line Projection at {angle}°")
    axes[i].set_xlim([0, 180])
    axes[i].set_ylim([0, np.max(sinogram_180.asnumpy())])

plt.tight_layout()
plt.show()

# %%                                                                                                                                                                                                                 
# Results with Final Sinogram                                                                                                                                                                                        
# --------                                                                                                                                                                                                           
# After taking 180 different line projections, we're able to produce a clearer ``Sinogram`` compared to what we had before.
# From the image above, we were able plot the angle of the line projection and its corresponding line projection values.

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].imshow(masked_image, cmap = 'gray', aspect = 'auto')
axes[0].set_title('Original Camera Man Image')
image = axes[1].imshow(sinogram_180.asnumpy()[0].T, cmap = 'gray', aspect = 'auto')
axes[1].set_title('Sinogram of the Original Image')
axes[1].set_xlabel('Projection Angles')
axes[1].set_ylabel('Pixels')
plt.tight_layout()
plt.show()

# %%                                                                                                                                                                                                                 
# Backward Projections with 180 projections
# --------
# We can transform our line projections back into the original image through reconstruction techniques.
# The output of applying ``project`` on an ASPIRE ``Image`` object is a ``Sinogram`` object.
# Once we have our Sinogram object, the ``backproject`` method that takes in a set of radians and applies the inverse FFT to convert our line projections back to the original image.
# Let’s visualize what our most recent sinogram looks like after applying the ``backproject`` method.

back_project_180 = sinogram_180.backproject(rads_180)
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].imshow(back_project_180.asnumpy()[0], cmap = 'gray', aspect = 'auto')
axes[0].set_title('Back Projection with 180 line projections')
image = axes[1].imshow(image_camera.asnumpy()[0], cmap='gray', aspect='auto')
axes[1].set_title('Original Image')
plt.tight_layout()
plt.show()

# %%                                                                                                                                                                                                                 
# Backward Projections with 90 projections
# --------                                                                                                                                                                                                           
# This is what applying our ``backproject`` method looks like with 90 projections.
# As we can see, the reconstructed image is still blurry but we're able to tell what the reconstructed image is.

back_project_90 = sinogram_90.backproject(rads_90)
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].imshow(back_project_90.asnumpy()[0], cmap = 'gray', aspect = 'auto')
axes[0].set_title('Back Projection with 90 line projections')
image = axes[1].imshow(image_camera.asnumpy()[0], cmap='gray', aspect='auto')
axes[1].set_title('Original Image')
plt.tight_layout()
plt.show()

# %%                                                                                                                                                                                                                 
# Backward Projections with 5 projections                                                                                                                                                                         
# --------
# This is what applying our ``backproject`` method looks like with only five projections.
# As we can see, it’s difficult to recognize that the image we’re recreating is the camera man from earlier.

back_project_5 = sinogram_five.backproject(rads_5)
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].imshow(back_project_5.asnumpy()[0], cmap = 'gray', aspect = 'auto')
axes[0].set_title('Back Project with 5 line projections')
image = axes[1].imshow(image_camera.asnumpy()[0], cmap='gray', aspect='auto')
axes[1].set_title('Original Image')
plt.tight_layout()
plt.show()

# %%
# Final Remarks
# --------
# As we conclude our demonstration, it's crucial to have a sufficient number of line projections when applying the respective forward and backward project methods.
# The quality of the reconstructed image is highly dependent on the number of projections, as demonstrated through our examples.
