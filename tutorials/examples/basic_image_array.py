#pip install colorednoise
#import colorednoise

import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Lets get some image data as a numpy array.

# Grab some image, here we have a Procyon (common north american trash panda).
img_data = misc.face(gray=True)

# Crop to a square
l = min(img_data.shape)
img_data = img_data[0:l,0:l]
plt.imshow(img_data, cmap=plt.cm.gray)
print(f'Shape {img_data.shape}')

# Initially, we will add some white noise to the image.
# Later we will try other types of noise in combination with whitening.

mu = 0.
sigma = 256.
noise = np.random.normal(mu, sigma, img_data.shape)

img_data_withWhiteNoise = img_data + noise
plt.imshow(img_data_withWhiteNoise, cmap=plt.cm.gray)


#------------------------------------------------------------------------------
# Now that we have an example array, we''ll begin using the ASPIRE toolkit.

from aspire.image import Image
from aspire.source import ArrayImageSource
from aspire.noise import WhiteNoiseEstimator

# First we'll make an ASPIRE Image instance out of our data.
# This is a light wrapper over the numpy array. Many ASPIRE internals
# are built around an Image classes.

# Construct the Image class by passing it the array data.
img = Image(img_data_withWhiteNoise)

# "Source" classes are what we use in processing pipelines.
# They provide a consistent interface to a variety of underlying data sources.
# In this case, we'll just use our Image to run a small experiment.
# If you were happy with the experiment design on an array of test data,
# the source is easily swapped out to something like RelionSource,
# which might point at a stack of images too large to fit in memory at once.
# This would be managed behind the scenes for you.
img_src = ArrayImageSource(img)

# ASPIRE's WhiteNoiseEstimator consumes from a Source
noise_estimator = WhiteNoiseEstimator(img_src)

# We can use that estimator to whiten all the images in the Source.
img_src.whiten(noise_estimator.filter)

# We can get a copy as numpy array instead of an ASPIRE source object.
img_data_whitened = img_src.asnumpy()
