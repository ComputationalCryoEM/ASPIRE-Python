import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from skimage.measure import block_reduce

from aspire.image import Image
from aspire.image.xform import NoiseAdder
from aspire.noise import WhiteNoiseEstimator
from aspire.operators import ScalarFilter
from aspire.source import ArrayImageSource

# ------------------------------------------------------------------------------
# Lets get some basic image data as a numpy array.

# Scipy ships with a portrait.
#  We'll take the grayscale representation as floating point data.
stock_img_data = misc.face(gray=True).astype(np.float32)

# Crop to a square
n_pixels = min(stock_img_data.shape)
stock_img_data = stock_img_data[0:n_pixels, 0:n_pixels]
# downsample (speeds things up)
stock_img_data = block_reduce(stock_img_data, (4, 4))

plt.imshow(stock_img_data, cmap=plt.cm.gray)
plt.title("Starting Image")
plt.show()

# We'll also compute the spectrum of the original image sample for later.
stock_img_data_f = np.abs(np.fft.fftshift(np.fft.fft2(stock_img_data)))


# ------------------------------------------------------------------------------
# Now that we have an example array, we'll begin using the ASPIRE toolkit.

# First we'll make an ASPIRE Image instance out of our data.
# This is a light wrapper over the numpy array. Many ASPIRE internals
# are built around an Image classes.

# Construct the Image class by passing it an array of data.
img = Image(stock_img_data)

# We'll begin processing by adding some noise.
#   We'd like to create uniform noise for a 2d image with prescibed variance,
#   say yielding an SNR around 10.
noise_var = np.var(stock_img_data) * 10.0
noise_filter = ScalarFilter(dim=2, value=noise_var)

#   Then create a NoiseAdder,
noise = NoiseAdder(seed=123, noise_filter=noise_filter)

#   which we can apply to our image data.
img_with_noise = noise.forward(img)

# We'll plot the first image (we only have one in our stack here).
plt.imshow(img_with_noise[0], cmap=plt.cm.gray)
plt.title("Noisy Image")
plt.show()


# ------------------------------------------------------------------------------
# Lets try an experiment, this time on a stack of images in an array.
#   In real use, you would probably bring your own stack of images,
#   but we'll create some here as before.
n_imgs = 128
imgs_data = np.empty(
    (n_imgs, stock_img_data.shape[-2], stock_img_data.shape[-1]), dtype=np.float64
)
for i in range(n_imgs):
    imgs_data[i] = stock_img_data
imgs = Image(imgs_data)

# Similar to before, we'll construct noise with a constant variance,
#   but now for the whole stack.
noise_var = np.var(stock_img_data) * 10.0
noise_adder = NoiseAdder(seed=123, noise_filter=ScalarFilter(dim=2, value=noise_var))
imgs_with_noise = noise_adder.forward(imgs)

# We can check some images, which should be the same up to noise.
n_check = 2
fig, axs = plt.subplots(1, n_check)
for i in range(n_check):
    axs[i].imshow(imgs_with_noise[i], cmap=plt.cm.gray)
    axs[i].set_title(f"Noisy Image {i}")
plt.show()

# Lets say we want to add additonal noise to half the images, every other image.
indices = range(1, n_imgs, 2)
noise_adder = NoiseAdder(
    seed=123, noise_filter=ScalarFilter(dim=2, value=4 * noise_var)
)
# We can use the "indices" to selectively apply our xform.
#   Note, that the xform will return a dense Image, so we need to match dimensions
#   There is an IndexdXform that provides an alternative to this.
imgs_with_noise[indices] = noise_adder.forward(imgs_with_noise, indices=indices)[
    : len(indices)
]

# We can check now that the second image has a different noise profile.
n_check = 2
fig, axs = plt.subplots(1, n_check)
for i in range(n_check):
    axs[i].imshow(imgs_with_noise[i], cmap=plt.cm.gray)
    axs[i].set_title(f"Noisy Image {i}")
plt.show()


# ------------------------------------------------------------------------------
# Use ASPIRE pipeline to Whiten

# "Source" classes are what we use in processing pipelines.
# They provide a consistent interface to a variety of underlying data sources.
# In this case, we'll just use our Image to run a small experiment.
# If you were happy with the experiment design on an array of test data,
# the source is easily swapped out to something like RelionSource,
# which might point at a stack of images too large to fit in memory at once.
# This would be managed behind the scenes for you.
imgs_src = ArrayImageSource(imgs_with_noise)


# One of the tools we can use is a NoiseEstimator,
#   which consumes from a Source.
noise_estimator = WhiteNoiseEstimator(imgs_src)

# Once we have the estimator instance,
#   we can use it in a transform applied to our Source.
imgs_src.whiten(noise_estimator.filter)

# We can get numpy arrays from our source if we want them.
whitened_imgs_data = imgs_src.images(0, imgs_src.n).asnumpy()

fig, axs = plt.subplots(2, 2)
for i in range(axs.shape[1]):
    axs[0, i].imshow(imgs_with_noise[i], cmap=plt.cm.gray)
    axs[0, i].set_title(f"Noisy Image {i}")
    axs[1, i].imshow(whitened_imgs_data[i], cmap=plt.cm.gray)
    axs[1, i].set_title(f"Whitened Noisy Image {i}")
plt.show()


# Okay, so lets look at the spectrum
fig, axs = plt.subplots(2, 2)
for i in range(axs.shape[1]):
    imgs_with_noise_f = np.abs(np.fft.fftshift(np.fft.fft2(imgs_with_noise[i])))
    axs[0, i].imshow(np.log(1 + imgs_with_noise_f), cmap=plt.cm.gray)
    axs[0, i].set_title(f"Spectrum of Noisy Image {i}")
    whitened_imgs_f = np.abs(np.fft.fftshift(np.fft.fft2(whitened_imgs_data[i])))
    axs[1, i].imshow(np.log(1 + whitened_imgs_f), cmap=plt.cm.gray)
    axs[1, i].set_title(f"Whitened Noisy Image {i}")
plt.show()


# Stil hard to tell.
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


# Still not happy with this part
colors = ["r", "g"]
for i in range(axs.shape[1]):
    imgs_with_noise_f = np.abs(np.fft.fftshift(np.fft.fft2(imgs_with_noise[i])))
    noise_f = imgs_with_noise_f - stock_img_data_f
    plt.plot(radial_profile(noise_f), color=colors[i], label=f"PSD of Noisy Image {i}")

    whitened_imgs_f = np.abs(np.fft.fftshift(np.fft.fft2(whitened_imgs_data[i])))
    whitened_noise_f = stock_img_data_f - whitened_imgs_f
    plt.plot(
        radial_profile(whitened_noise_f),
        color=colors[i],
        linestyle="--",
        label=f"PSD Whitened Noisy Image {i}",
    )
plt.title("Spectrum Profiles")
plt.legend()
plt.show()
