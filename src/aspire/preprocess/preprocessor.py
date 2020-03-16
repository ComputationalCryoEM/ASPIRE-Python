import numpy as np
from aspire.preprocess.phaseflip import phaseflip_star_file
# from aspire.preprocess.prewhiten import prewhiten
# from aspire.preprocess.global_phaseflip import global_phaseflip
# import aspire.utils.common as common
from pyfftw.interfaces.numpy_fft import fft2, ifft2
import time


def preprocess(star_file, pixel_size=None, crop_size=-1, downsample_size=89):
    use_crop = crop_size > 0
    use_downsample = downsample_size > 0
    # flag to indicate not to transform back in phaseflip and to to transform in downsample
    flag = use_downsample and not use_crop
    print('Starting phaseflip')
    tic = time.time()
    stack = phaseflip_star_file(star_file, pixel_size, flag)
    toc = time.time()
    s = stack.shape
    print('Finished phaseflip in {} seconds, found {} images with resolution {}'.format(toc - tic, s[0], s[1]))
    if use_crop:
        print('Start cropping')
        tic = time.time()
        # TODO - out of core version
        stack = crop(stack, (-1, crop_size, crop_size))
        toc = time.time()
        print('Finished cropping in {} seconds, from {} to {}'.format(toc - tic, s[1], crop_size))
    else:
        print('Skip cropping')
        crop_size = s[1]
    if use_downsample > 0:
        print('Start downsampling')
        tic = time.time()
        stack = downsample(stack, downsample_size, stack_in_fourier=flag)
        toc = time.time()
        print('Finished downsampling in {} seconds, from {} to {}'.format(toc - tic, crop_size, downsample_size))
    else:
        print('Skip downsampling')

    # Up to this point, the stacks are C aligned, now aligning to matlab (in the future it will stay C aligned)
    print('Changing the stack to matlab align (temporary)')
    stack = np.ascontiguousarray(stack.T)

    print('Start normalizing background')
    # stack, _, _ = normalize_background(stack, stack.shape[1] * 45 // 100)
    print('Start prewhitening')
    # stack = prewhiten(stack)
    print('Start global phaseflip')
    # stack = global_phaseflip(stack)
    return stack


def downsample_preprocess(stack, n, mask=None, stack_in_fourier=False):
    """
    Use Fourier methods to change the sample interval and/or aspect ratio
    of any dimensions of the input image 'img'. If the optional argument
    stack is set to True, then the *first* dimension of 'img' is interpreted as the index of
    each image in the stack. The size argument side is an integer, the size of the
    output images.  Let the size of a stack
    of 2D images 'img' be n1 x n1 x k.  The size of the output will be side x side x k.

    If the optional mask argument is given, this is used as the
    zero-centered Fourier mask for the re-sampling. The size of mask should
    be the same as the output image size. For example for downsampling an
    n0 x n0 image with a 0.9 x nyquist filter, do the following:
    msk = fuzzymask(n,2,.45*n,.05*n)
    out = downsample(img, n, 0, msk)
    The size of the mask must be the size of output. The optional fx output
    argument is the padded or cropped, masked, FT of in, with zero
    frequency at the origin.
    """

    size_in = np.square(stack.shape[1])
    size_out = np.square(n)
    mask = 1 if mask is None else mask
    num_images = stack.shape[0]
    output = np.zeros((num_images, n, n), dtype='float32')
    images_batches = np.array_split(np.arange(num_images), 500)
    for batch in images_batches:
        curr_batch = np.array(stack[batch])
        curr_batch = curr_batch if stack_in_fourier else fft2(curr_batch)
        fx = crop(np.fft.fftshift(curr_batch, axes=(-2, -1)), (-1, n, n)) * mask
        output[batch] = ifft2(np.fft.ifftshift(fx, axes=(-2, -1))) * (size_out / size_in)
        print('finished {}/{}'.format(batch[-1] + 1, num_images))
    return output


def crop(x, out_shape):
    """
    :param x: ndarray of size (N_1,...N_k)
    :param out_shape: iterable of integers of length k. The value in position i (n_i) is the size we want to cut from
        the center of x in dimension i. If the value of n_i <= 0 or >= N_i then the dimension is left as is.
    :return: out: The center of x with size outshape.
    """
    in_shape = np.array(x.shape)
    out_shape = np.array([s if 0 < s < in_shape[i] else in_shape[i] for i, s in enumerate(out_shape)])
    start_indices = in_shape // 2 - out_shape // 2
    end_indices = start_indices + out_shape
    indexer = tuple([slice(i, j) for (i, j) in zip(start_indices, end_indices)])
    out = x[indexer]
    return out


def downsample(x, out_shape):
    """
    :param x: ndarray of size (N_1,...N_k)
    :param out_shape: iterable of integers of length k. The value in position i (n_i) is the size we want to cut from
        the center of x in dimension i. If the value of n_i <= 0 or >= N_i then the dimension is left as is.
    :return: out: downsampled x
    """
    dtype_in = x.dtype
    in_shape = np.array(x.shape)
    out_shape = np.array([s if 0 < s < in_shape[i] else in_shape[i] for i, s in enumerate(out_shape)])
    fourier_dims = np.array([i for i, s in enumerate(out_shape) if 0 < s < in_shape[i]])
    size_in = np.prod(in_shape[fourier_dims])
    size_out = np.prod(out_shape[fourier_dims])

    fx = crop(np.fft.fftshift(np.fft.fft2(x, axes=fourier_dims), axes=fourier_dims), out_shape)
    out = ifft2(np.fft.ifftshift(fx, axes=fourier_dims), axes=fourier_dims) * (size_out / size_in)
    return out.astype(dtype_in)


def normalize_background(stack, radius=None):
    n = stack.shape[1]
    radius = n // 2 if radius is None else radius
    circle = ~disc(n, radius)
    background_pixels = stack[circle]
    mean = np.mean(background_pixels, 0)
    std = np.std(background_pixels, 0, ddof=1)
    stack -= mean
    stack /= std
    return stack, mean, std


def disc(n, r=None, inner=False):
    """
    Return the points inside the circle of radius=r in a square with side n. if inner is True don't return only the
    strictly inside points.
    :param n: integer, the side of the square
    :param r: The radius of the circle (default: n // 2)
    :param inner:
    :return: nd array with 0 outside of the circle and 1 inside
    """
    r = n // 2 if r is None else r
    ctr = (n + 1) / 2
    y_axis, x_axis = np.meshgrid(np.arange(1, n + 1), np.arange(1, n + 1))
    radiisq = np.square(x_axis - ctr) + np.square(y_axis - ctr)
    if inner is True:
        return radiisq < r ** 2
    return radiisq <= r ** 2

