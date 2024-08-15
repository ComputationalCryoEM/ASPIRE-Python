"""
Radon Transform
=======================================

This example demonstrates how to compute the Radon transform and its
inverse using the Aspire library and compares it with the Skimage implementation.

We will:

- Load an example image.

- Apply the Radon transform at different angles.

- Visualize the results.

- Compare the performance with Skimage's implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import radon, iradon
from aspire.image import Image
from aspire.utils import grid_2d
from aspire.sinogram import Sinogram  

# %%                                                                                                                                                                                                                  
# Overview                                                                                                                                                                                                            
# -------- 
# The ``project`` and ``backproject`` are two methods that allow you to 
# manipulate images by simulating the process of creating and reconstructing projections. The ``project`` method
# transforms an image into a set of 1D projections, representing the image as it would be seen from different
# angles, while ``backproject`` method reconstructs the original image by combining these projections (add more).
# Let's begin by walking through these processes and referencing the ASPIRE library for our methods.

# 1. Start with loading the camera man image
image = data.camera().astype(np.float64)
img_size = image.shape[0]
image_without_mask = Image(image)

# 2. Masking the image to begin line projections (grid generation)
grid = grid_2d(img_size, normalized=True, shifted=True)
mask = grid["r"] < 1
masked_image = image * mask
image_camera = Image(masked_image)

# hide later
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
# Now that we have our image, let's take the forward projection using the ``project`` method in the ASPIRE ``image`` class.
# It's important to note that in order to take the forward projection of an image, it must be an ``Image`` object.
# We will start with an illustration of 5 projects and illustrate what this looks like

# sinogram with five projections
angles_5 = np.linspace(0, 360, 5, endpoint=False)
rads_5 = angles_5 / 180.0 * np.pi
sinogram_five = image_camera.project(rads_5)

# 3. Show camera man and a sinogram through camera angle the corresponding line projection
angles_to_plot = [0, 72, 216, 288]

# Create figure and axis for the original image with line projections
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(masked_image, cmap='gray')
ax.set_title('Camera Man with Line Projections')

# could try to make a tile or collection of these line projects, need to fix later

for angle in angles_to_plot:
    rad_angle = np.pi * angle / 180
    ax.axline((img_size // 2, img_size // 2), slope=np.tan(rad_angle), color='r')

plt.show()    
    
n_cols = 2
n_rows = int(np.ceil(len(angles_to_plot) / n_cols)) 

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6))
axes = axes.flatten()

for i, angle in enumerate(angles_to_plot):
    axes[i].plot(sinogram_five.asnumpy()[0, :, int(angle * img_size / 360)], color='red')
    axes[i].set_title(f"Line Projection at {angle}°")
    axes[i].set_xlim([0, 4])
    axes[i].set_ylim([0, np.max(sinogram_five.asnumpy())])

plt.tight_layout()
plt.show()

# %%                                                                                                                                                                                                                 
# Breaker
# --------                                                                                                                                                                                                           
# As we can see, our forward project method works yet looks quite blurry.
# add more commentary

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
# Breaker                                                                                                                                                                                                             
# --------                                                                                                                                                                                                            
# As we can see, our forward project method works yet looks quite blurry.                                                                                                                                             
# add more commentary     
# sinogram with five projections                                                                                                                                                                                     
angles_90 = np.linspace(0, 360, 90, endpoint=False)
rads_90 = angles_90 / 180.0 * np.pi
sinogram_90 = image_camera.project(rads_90)

# 3. Show camera man and a sinogram through camera angle the corresponding line projection                                                                                                                           
angles_to_plot = [24, 154, 189, 227]

# Create figure and axis for the original image with line projections                                                                                                                                                
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(masked_image, cmap='gray')
ax.set_title('Camera Man with Line Projections')

for angle in angles_to_plot:
    rad_angle = np.pi * angle / 180
    ax.axline((img_size // 2, img_size // 2), slope=np.tan(rad_angle), color='r')

plt.show()

n_cols = 2
n_rows = int(np.ceil(len(angles_to_plot) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6))
axes = axes.flatten()

for i, angle in enumerate(angles_to_plot):
    axes[i].plot(sinogram_90.asnumpy()[0, :, int(angle * img_size / 360)], color='red')
    axes[i].set_title(f"Line Projection at {angle}°")
    axes[i].set_xlim([0, 90])
    axes[i].set_ylim([0, np.max(sinogram_90.asnumpy())])

plt.tight_layout()
plt.show()

# %%                                                                                                                                                                                                                 
# Breaker                                                                                                                                                                                                            
# --------                                                                                                                                                                                                           
# As we can see, our forward project method works yet looks quite blurry.                                                                                                                                            
# add more commentary                                                                                                                                                                                                  

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
# Sinogram with 180 angles                                                                                                                                                                                           
# --------                                                                                                                                                                                                           
# As mentioned previously, we can transform our line projections (or sinograms) into the original reconstructed image.                                                                                               
# now bring back the line projections into the reconstructed image. Let's start by taking a few back projections and seeing what we can construct.
# recall that this is the sinogram projection with all 180 angles
angles_180 = np.linspace(0, 360, 180, endpoint=False)
rads_180 = angles_180 / 180.0 * np.pi
sinogram_180 = image_camera.project(rads_180)

# 3. Show camera man and a sinogram through camera angle the corresponding line projection                                                                                                                            
angles_to_plot = [42, 88, 180, 343]

# Create figure and axis for the original image with line projections                                                                                                                                                 
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(masked_image, cmap='gray')
ax.set_title('Camera Man with Line Projections')

def rotate_points(pts, ang):
    r = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    return r @ pts.T

for angle in angles_to_plot:
    index = int(angle * img_size / 360)
    y = sinogram_180.asnumpy()[0, :, index]
    y = - (y - np.min(y)) / (np.max(y) - np.min(y)) / 2
    x = np.linspace(-0.5, 0.5, len(y), endpoint = True)
    pts = np.vstack((x,y)).T
    rot_points = rotate_points(pts, angle * np.pi / 180).T
    rot_points = (rot_points + 0.5) * img_size
    x_rot, y_rot = rot_points[:, 0], rot_points[:, 1]
    ax.plot(x_rot, y_rot, color='b')
    rad_angle = np.pi * angle / 180
    ax.axline((img_size // 2, img_size // 2), slope=np.tan(rad_angle), color='r')

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
# Breaker                                                                                                                                                                                           
# --------                                                                                                                                                                                                            
# As mentioned previously, we can transform our line projections (or sinograms) into the original reconstructed  
# now bring back the line projections into the reconstructed image. Let's start by taking a few back projections and seeing what we can construct.                                                                    
# recall that this is the sinogram projection with all 180 angles  


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
# Backward Projections @ 5                                                                                                                                                                                          
# --------
# As mentioned previously, we can transform our line projections (or sinograms) into the original reconstructed image.
# now bring back the line projections into the reconstructed image. Let's start by taking a few back projections of the five line projections we took earlier.
back_project_five = sinogram_five.backproject(rads_5)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].imshow(back_project_five.asnumpy()[0], cmap = 'gray', aspect = 'auto')
axes[0].set_title('Back Projection from 5 angles')
image = axes[1].imshow(image_camera.asnumpy()[0], cmap='gray', aspect='auto')
axes[1].set_title('Original Image')
plt.tight_layout()
plt.show()

# %%                                                                                                                                                                                                                  
# Backward Projections @ 90
# --------                                                                                                                                                                                                            
# As mentioned previously, we can transform our line projections (or sinograms) into the original reconstructed image.                                                                                                
# now bring back the line projections into the reconstructed image. Let's start by taking a few back projections of the five line projections we took earlier.                                                         
back_project_90 = sinogram_90.backproject(rads_90)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].imshow(back_project_90.asnumpy()[0], cmap = 'gray', aspect = 'auto')
axes[0].set_title('Back Projection from 90 angles')
image = axes[1].imshow(image_camera.asnumpy()[0], cmap='gray', aspect='auto')
axes[1].set_title('Original Image')
plt.tight_layout()
plt.show()

# %%                                                                                                                                                                                                                  
# Backward Projections @ 180                                                                                                                                                                                                
# --------
# As we can see, this is not very effective, let's add more back projections images.
# Add more commentary later

back_project_180 = sinogram_180.backproject(rads_180)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].imshow(back_project_180.asnumpy()[0], cmap = 'gray', aspect = 'auto')
axes[0].set_title('Back Project with 180 angles')
image = axes[1].imshow(image_camera.asnumpy()[0], cmap='gray', aspect='auto')
axes[1].set_title('Original Image')
plt.tight_layout()
plt.show()

# %%
# Final Product
# --------
# After taking enough back projections, we can fully reconstruct our original image
