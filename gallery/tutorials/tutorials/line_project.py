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

import time
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import radon, iradon
from aspire.image import Image
from aspire.utils import grid_2d
from aspire.line import Line  

# %%                                                                                                                                                                                                                  
# Overview                                                                                                                                                                                                            
# -------- 
# ``project`` and ``backproject`` are two methods that allow you to 
# manipulate images by simulating the process of creating and reconstructing projections. The ``project`` method
# transforms an image into a set of 1D projections, representing the image as it would be seen from different
# angles, while ``backproject`` method reconstructs the original image by combining these projections (add more).
# Let's begin by walking through these processes and referencing the ASPIRE library for our methods. 

"""
Let's begin by creating our test image. We'll be referencing the Camera Man from Scikit Learn.
"""
# 1. Start with the camera man image
image = data.camera().astype(np.float64)
img_size = image.shape[0]

# 2. Make some line projections
angles = np.linspace(0, 360, 180, endpoint=False)
rads = angles / 180.0 * np.pi
grid = grid_2d(img_size, normalized=True, shifted=True)
mask = grid["r"] < 1
masked_image = image * mask
image_camera = Image(masked_image)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(masked_image)
ax.set_title("Original Image")
plt.show()

# %%                                                                                                                                                                                                                  
# Forward Projection
# --------
# Now that we have our image, let's take the forward projection using the ``project`` method in the ASPIRE ``image`` class.
# It's important to note that in order to take the forward projection of an image, it must be an ``Image`` object.
# for our demonstration, we will take the forward projection and plot angles ``42, 88, and 180`` to illusrate how these projections look.

# sinogram projections for later
sinogram_camera = image_camera.project(rads)

# 3. Show camera man and a line through camera angle the corresponding line projection
angles_to_plot = [42, 88, 180]

# could try to make a tile or collection of these line projects, need to fix later
for angle in angles_to_plot:
    rad_angle = np.pi * angle / 180

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(masked_image)

    # Draw a line at the given angle
    ax.axline((img_size // 2, img_size // 2), slope=np.tan(rad_angle), color='r')
    ax.set_title(f"Camera Man with Line @ {angle} degrees")
    plt.show()

    # Single line projection from camera man
    plt.figure(figsize=(8, 4))
    plt.plot(sinogram_camera[0, :, int(angle * img_size / 360)], 'r')
    plt.title(f"Line Projection @ {angle} degrees")
    plt.show()

    
# Project the image and show the line projections (potentially just solo)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(masked_image, aspect='auto')
axes[0].set_title('Original Image')
axes[1].imshow(sinogram_camera[0].T, aspect='auto')
axes[1].set_title('Our Projection')
plt.tight_layout()
plt.show()


# %%                                                                                                                                                                                                                  
# Backward Projections                                                                                                                                                                                                
# --------
# As mentioned previously, we can transform our line projections (or sinograms) into the original reconstructed image.

# now bring back the line projections into the reconstructed image
line_sinogram = sinogram_camera.asnumpy()
back_project = line_sinogram.backproject(rads)

