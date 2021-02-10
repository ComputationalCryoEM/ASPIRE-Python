import colorednoise
import matplotlib._color_data as mcd
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from skimage.measure import block_reduce

from aspire.image import Image
from aspire.noise import WhiteNoiseEstimator
from aspire.source import ArrayImageSource

# ------------------------------------------------------------------------------
# Lets get some basic image data as a numpy array.

# Scipy ships with a portrait of a Procyon (common north american trash panda).
img_data = misc.face(gray=True)

# Crop to a square
n_pixels = min(img_data.shape)
img_data = img_data[0:n_pixels, 0:n_pixels]
# Block (down)sample the image.
img_data = block_reduce(img_data, (2, 2))

plt.imshow(img_data, cmap=plt.cm.gray)
plt.title("Starting Image")
plt.show()

# Initially, we will add some white noise to the image.
# Later we will try other types of noise in combination with whitening.

mu = 0.0
sigma = 256.0
white = np.random.normal(mu, sigma, img_data.shape)

snr = np.var(img_data) / np.var(white)
print(f"Rough SNR {snr}")

# We'll also compute the spectrum of the original image and white noise sample for later.
img_data_f = np.abs(np.fft.fftshift(np.fft.fft2(img_data)))
white_f = np.abs(np.fft.fftshift(np.fft.fft2(white)))


# ------------------------------------------------------------------------------
# Now that we have an example array, we'll begin using the ASPIRE toolkit.

# First we'll make an ASPIRE Image instance out of our data.
# This is a light wrapper over the numpy array. Many ASPIRE internals
# are built around an Image classes.

# Construct the Image class by passing it an array of data.
img = Image(img_data + white)

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
img_data_whitened = img_src.images(0, img_src.n).asnumpy()


# ------------------------------------------------------------------------------
# Lets try a small experiment, this time on a stack of thee images in an array.
# We'll go through the same steps as before.

# The following will generate additional distributions of noise.
pink = (
    colorednoise.powerlaw_psd_gaussian(1, img_data.shape)
    + colorednoise.powerlaw_psd_gaussian(1, img_data.shape).T
) * sigma
pink_f = np.abs(np.fft.fftshift(np.fft.fft2(pink)))

brown = (
    colorednoise.powerlaw_psd_gaussian(2, img_data.shape)
    + colorednoise.powerlaw_psd_gaussian(2, img_data.shape).T
) * sigma
brown_f = np.abs(np.fft.fftshift(np.fft.fft2(brown)))

# Storing noises in a dictionary for reference later.
noises = {"White": white, "Pink": pink, "Brown": brown}
noises_f = {"White": white_f, "Pink": pink_f, "Brown": brown_f}


# Setup some arrays to hold our data.
#   In real use, you would probably bring your own stack of images,
#   but we'll create some here as before.
stack = np.zeros((3, img_data.shape[-2], img_data.shape[-1]))
stack_f = np.zeros_like(stack)
stack_whitened_f = np.zeros_like(stack)
whitened_noises_f = dict()

for i, name in enumerate(sorted(noises)):
    #  Adding the different noises to our original image data.
    stack[i] = img_data + noises[name]
    # Compute and keep some fourier space arrays to plot later.
    stack_f[i] = np.abs(np.fft.fftshift(np.fft.fft2(stack[i])))


# Construct our Image and Source.
images = Image(stack)
img_src = ArrayImageSource(images)


# Use ASPIRE pipeline to Whiten
noise_estimator = WhiteNoiseEstimator(img_src)
img_src.whiten(noise_estimator.filter)
img_data_whitened = img_src.images(0, img_src.n).asnumpy()


# Plot before and after whitening.
fig, axs = plt.subplots(3, 4)
for i, name in enumerate(sorted(noises)):

    stack[i] = img_data + noises[name]
    # Lets save the whitened fourier space
    stack_whitened_f[i] = np.abs(np.fft.fftshift(np.fft.fft2(img_data_whitened[i])))

    # and retrieve the whitened noise profile by subracting from the original signal
    whitened_noises_f[name] = img_data_f - stack_whitened_f[i]

    # and we can make some plots now
    axs[i, 0].imshow(stack[i], cmap=plt.cm.gray)
    axs[i, 0].set_title(f"Image with {name} Noise")

    axs[i, 1].imshow(np.log(1 + stack_f[i]), cmap=plt.cm.gray)
    axs[i, 1].set_title(f"{name} Noise Spectrum")

    axs[i, 2].imshow(np.log(1 + stack_whitened_f[i]), cmap=plt.cm.gray)
    axs[i, 2].set_title(f"Whitened {name} Noise Spection")

    axs[i, 3].imshow(img_data_whitened[i], cmap=plt.cm.gray)
    axs[i, 3].set_title(f"Image with Whitened {name} Noise")

plt.show()


# We'll also want to take a look at the spectrum power distribution.
#  Since we just want to see the character of what is happening,
#  I'll assume each pixel's contribution is placed at their lower left corner,
#  and compute a crude radial profile.
#  Code from a discussion at https://stackoverflow.com/questions/21242011/most-efficient-way-to-calculate-radial-profile.
def radial_profile(data):
    y, x = np.indices((data.shape))
    # Distance from origin to lower left corner
    r = np.sqrt(x ** 2 + y ** 2).astype(np.int)
    binsum = np.bincount(r.ravel(), np.log(1 + data.ravel()))
    bincount = np.bincount(r.ravel())
    # Return the mean per bin
    return binsum / bincount


# Setup some plot colors
legend_colors = {
    "White": mcd.XKCD_COLORS["xkcd:black"],
    "Pink": mcd.XKCD_COLORS["xkcd:pink"],
    "Brown": mcd.XKCD_COLORS["xkcd:sienna"],
}

# Loop through the sprectral profiles and plot.
for i, name in enumerate(sorted(noises)):
    plt.plot(radial_profile(noises_f[name]), legend_colors[name], label=name)
    plt.plot(
        radial_profile(whitened_noises_f[name]),
        color=f"{legend_colors[name]}",
        linestyle="--",
        label=f"Whitened {name}",
    )

plt.title(f"Spectrum Profiles")
plt.legend()
plt.show()

# At this point we should see that ASPIRE's whitening procedure has
#   effected the distribution of noise.
# In other tutorials a Simulation source is generally used
#   in place of constructing your own image stacks.
