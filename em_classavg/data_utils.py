from scipy.io import loadmat
import numpy as np


def mat_to_npy(file_name):
    return loadmat(file_name + '.mat')[file_name]


def mat_to_npy_vec(file_name):
    a = mat_to_npy(file_name)
    return a.reshape(a.shape[0] * a.shape[1])


def estimate_snr(images, prewhiten=False):
    # Prewhiten the image if needed.
    if prewhiten:
        raise NotImplementedError

    if len(images.shape) == 2:
        images = images[np.newaxis, :, :]

    images = np.transpose(images, axes=(1, 2, 0))

    n = images.shape[2]

    # Compute radius outside which the pixels are noise
    p = images.shape[1]
    radius_of_mask = np.floor(p / 2.0) - 1.0

    # Compute indices of noise samples
    r = cart2rad(p)
    points_inside_circle = r < radius_of_mask
    num_noise_points = p * p - np.count_nonzero(points_inside_circle)

    var_n = np.sum(np.var(images[~points_inside_circle], axis=0)) * num_noise_points / (num_noise_points * n - 1)
    var_s = np.sum(np.var(images, axis=(0, 1))) * p * p / (images.size - 1) - var_n
    snr = var_s / var_n

    return float(snr), float(var_s), float(var_n)


def cart2rad(n):
    # Compute the radii corresponding of the points of a cartesian grid of size NxN points
    # XXX This is a name for this function.
    n = np.floor(n)
    x, y = image_grid(n)
    r = np.sqrt(np.square(x) + np.square(y))
    return r


def image_grid(n):
    # Return the coordinates of Cartesian points in an NxN grid centered around the origin.
    # The origin of the grid is always in the center, for both odd and even N.
    p = (n - 1.0) / 2.0
    x, y = np.meshgrid(np.linspace(-p, p, n), np.linspace(-p, p, n))
    return x, y


def normalize_background(stack):
    # Normalizes background to mean 0 and std 1.
    #
    # stack = normalize_background(stack)
    #   Estimate the mean and std of each image in the stack using pixels
    #   outside radius r (=half the image size in pixels), and normalize the image such that the
    #   background has mean 0 and std 1. Each image in the stack is corrected
    #   separately.
    #
    # Example:
    # stack2 = normalize_background(stack)
    n_images = len(stack)
    m = np.shape(stack)[1]
    n = np.shape(stack)[2]

    if m != n:
        ValueError('Images in the stack must be square.')

    r = np.floor(n / 2)

    # Find indices of backgruond pixels in the images
    ctr = (n + 1) / 2

    xv, yv = np.meshgrid(np.arange(1, n + 1), np.arange(1, n + 1))

    radii_sq = (xv - ctr) ** 2 + (yv - ctr) ** 2
    background_pixels_mask = (radii_sq > r * r)

    sd_bg = np.zeros(n_images)
    mean_bg = np.zeros(n_images)
    for kk in np.arange(n_images):
        proj = stack[kk]
        background_pixels = proj[background_pixels_mask]

        # Compute mean and standard deviation of background pixels
        mm = np.mean(background_pixels)
        sd = np.std(background_pixels, ddof=1)

        proj = (proj - mm) / sd
        stack[kk] = proj

        sd_bg[kk] = sd
        mean_bg[kk] = mm

    return stack, mean_bg, sd_bg