# Functions for processing images and image stacks

import mrcfile
import matplotlib.pyplot as plt
import numpy


def center(dims):
    # Find the center of a vector/image.
        return (dims+1.0)/2.0


def image_grid(n):
    # Return the coordinates of Cartesian points in an NxN grid centered around the origin.
    # The origin of the grid is always in the center, for both odd and even N.
    p = (n-1.0)/2.0
    x, y = numpy.meshgrid(numpy.linspace(-p, p, n), numpy.linspace(-p, p, n))
    return x, y


def cart2rad(n):
    # Compute the radii corresponding of the points of a cartesian grid of size NxN points
    # XXX This is a name for this function.
    n = numpy.floor(n)
    x, y = image_grid(n)
    r = numpy.sqrt(numpy.square(x)+numpy.square(y))
    return r


def estimate_snr(projs, do_prewhiten=0):
    # Prewhiten the image if needed.
    if do_prewhiten != 0:
        raise NotImplementedError

    # Compute radius outside which the pixels are noise
    p = projs.data.shape[1]
    radius_of_mask = numpy.floor(p/2.0)-1.0

    # Compute indices of noise samples
    r = cart2rad(p)
    mask = numpy.zeros((p, p), dtype=numpy.float64)
    mask[r < radius_of_mask] = -1
    noise_idx = numpy.nonzero(mask != -1)

    # Compute noise variance from all projections
    if len(projs.data.shape) == 2:
        n = 1
    else:
        n = projs.data.shape[0]

    # Total number of noise samples
    total_noise_samples = n * len(noise_idx[0])

    # Calculate the mean of the noise and the signal + noise.
    noise_means = numpy.zeros(n, projs.data.dtype)
    projs_means = numpy.zeros(n, projs.data.dtype)

    for k in xrange(n):
        if n == 1:
            proj = projs.data
        else:
            proj = projs.data[k]

        noise_vals = proj[noise_idx]
        noise_means[k] = numpy.mean(noise_vals)
        projs_means[k] = numpy.mean(proj)

    # Calculate the sample variance of the noise and the projections. Due to the normalization of the projections,
    # there is no need to estimate the variance of the noise, as it is equal to 1. However, I estimate it to verify
    # that nothing went wrong.
    noise_sumsq = numpy.zeros(n, projs.data.dtype)
    projs_sumsq = numpy.zeros(n, projs.data.dtype)

    for k in xrange(n):
        if n == 1:
            proj = projs.data
        else:
            proj = projs.data[k]

        noise_vals = proj[noise_idx]
        noise_sumsq[k] = numpy.sum(numpy.square(numpy.abs(noise_vals-noise_means[k])))

        # Note that it is incorrect to subtract the mean image, since then the code won't work for a stack consisting of
        # multiple copies of the same image plus noise.
        projs_sumsq[k] = numpy.sum(numpy.square(numpy.abs(proj-projs_means[k])))

    var_n = numpy.sum(noise_sumsq) / (total_noise_samples - 1)
    var_splusn = sum(projs_sumsq) / (projs.data.size - 1)
    var_s = var_splusn - var_n
    snr = var_s / var_n

    return snr, var_s, var_n

if __name__ == "__main__":
    projs=mrcfile.open('/home/yoel/data/work/ASPIRE.py/tests/single_noisy_proj.mrcs')
    #print projs.data[1].shape
    #plt.imshow(projs.data[1], cmap='gray')
    #plt.show()
    #im=numpy.random.rand(4,4)
    #print center(numpy.asarray(im.shape))
    #proj=numpy.random.rand(5,5)
    print estimate_snr(projs)