"""
Radon Transform with Aspire and Skimage
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

"""
Let's begin by creating our test image. We'll be referencing the Camera Man from Scikit Learn.
"""
# camera man
camera_image = data.camera()
plt.imshow(camera_image)
plt.title('Camera Man')
plt.show()

num_angles = 360
resolutions = [32, 64, 128, 256, 512]
num_images = 3
dtype = np.float64

# specific angle for our demonstration
angle = 42
rads_single = np.array(angle / 180.0 * np.pi)

radon_image = radon(camera_image, theta=angle, circle=True)
plt.imshow(radon_image, aspect='auto')
plt.title(f'Radon Transform at {angles} Degrees')
plt.xlabel('Projection position')
plt.ylabel('Projection angle')
plt.show()

image_aspire = Image(camera_image[np.newaxis, :, :].astype(dtype))
line_projection = image_aspire.project(rads_single)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(camera_image)
plt.title('Camera Man with Line at 42 Degrees')

plt.subplot(122)
plt.plot(line_projection.asnumpy()[0, 0])
plt.title('Line Projection at 42 Degrees')
plt.xlabel('Projection position')
plt.ylabel('Intensity')

plt.show()

print(f'Line Instance: {line_projection}')

back_projected_image = line_projection.backproject(rads_single).asnumpy()[0]

plt.imshow(back_projected_image)
plt.title('Back Projected Image')
plt.show()

aspire_forward_times = np.zeros((len(resolutions)))
skimage_forward_times = np.zeros((len(resolutions)))
aspire_backproject_times = np.zeros((len(resolutions)))
skimage_backproject_times = np.zeros((len(resolutions)))
postdoc_forward_times = np.zeros((len(resolutions)))

