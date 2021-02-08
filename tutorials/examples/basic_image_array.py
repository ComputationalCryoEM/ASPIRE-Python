import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import colorednoise

#------------------------------------------------------------------------------
# Lets get some image data as a numpy array.

# Grab some image, here we have a Procyon (common north american trash panda).
img_data = misc.face(gray=True)

# Crop to a square
l = min(img_data.shape)
img_data = img_data[0:l,0:l]
print(f'Shape {img_data.shape}')
plt.imshow(img_data, cmap=plt.cm.gray)
plt.title('Original Image')
plt.show()

# Initially, we will add some white noise to the image.
# Later we will try other types of noise in combination with whitening.

mu = 0.
sigma = 256.
white = np.random.normal(mu, sigma, img_data.shape)

img_data_withWhiteNoise = img_data + white
# plt.imshow(img_data_withWhiteNoise, cmap=plt.cm.gray)
# plt.title('With White Noise')
# plt.show()


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
#noise_estimator = WhiteNoiseEstimator(img_src)

# We can use that estimator to whiten all the images in the Source.
#img_src.whiten(noise_estimator.filter)

# We can get a copy as numpy array instead of an ASPIRE source object.
#img_data_whitened = img_src.images(0,img_src.n).asnumpy()
# plt.imshow(img_data_whitened[0], cmap=plt.cm.gray)
# plt.title('Whitened')
# plt.show()


#------------------------------------------------------------------------------
# Lets try adding a different sort of noise

pink = (colorednoise.powerlaw_psd_gaussian(1, img_data.shape) + colorednoise.powerlaw_psd_gaussian(1, img_data.shape).T) * sigma
pink2 = (colorednoise.powerlaw_psd_gaussian(1, img_data.shape) + colorednoise.powerlaw_psd_gaussian(1, img_data.shape).T) * sigma
pink3 = (colorednoise.powerlaw_psd_gaussian(1, img_data.shape) + colorednoise.powerlaw_psd_gaussian(1, img_data.shape).T) * sigma
brown = (colorednoise.powerlaw_psd_gaussian(2, img_data.shape) + colorednoise.powerlaw_psd_gaussian(2, img_data.shape).T) * sigma

#noises = {'White': white, 'Pink': pink, 'Brown': brown}
noises = {'Pink1': pink, 'Pink2': pink2, 'Pink3': pink3}

# We'll go through the same steps as before,
#   but adding different noise to our original image data,
#   and keeping some fourier space arrays to plot later.
myarray = np.zeros((3,l,l))
myarray_f = np.zeros_like(myarray)
myarray_f_whitened = np.zeros_like(myarray)
for i, name in enumerate(sorted(noises)):

    myarray[i] = img_data + noises[name]
    # Lets get the fourier space
    myarray_f[i] = np.abs(
        np.fft.fftshift(
            np.fft.fft2(myarray[i])))

# Construct our Image and Source
images = Image(myarray)
imgs_src = ArrayImageSource(images)

# Use ASPIRE pipeline to Whiten 
noise_estimator = WhiteNoiseEstimator(imgs_src)
imgs_src.whiten(noise_estimator.filter)
img_data_whitened = imgs_src.images(0,imgs_src.n).asnumpy()


# Plot before and after whitening.
fig, axs = plt.subplots(3,4)
for i, name in enumerate(sorted(noises)):

    myarray[i] = img_data + noises[name]
    # Lets get a picture of the whitened fourier space
    myarray_f_whitened[i] = np.abs(
        np.fft.fftshift(
            np.fft.fft2(img_data_whitened[i])))

    # and we can make the plots now
    axs[i,0].imshow(myarray[i], cmap=plt.cm.gray)    
    axs[i,0].set_title(f"Image with {name} Noise")

    axs[i,1].imshow(np.log(1+myarray_f[i]), cmap=plt.cm.gray)
    axs[i,1].set_title(f"{name} Noise Spectrum")
    
    axs[i,2].imshow(np.log(1+myarray_f_whitened[i]), cmap=plt.cm.gray)
    axs[i,2].set_title(f"Whitened {name} Noise Spection")
    
    axs[i,3].imshow(img_data_whitened[i], cmap=plt.cm.gray)
    axs[i,3].set_title(f"Image with {name} Noise Whitened")

plt.show()
    
