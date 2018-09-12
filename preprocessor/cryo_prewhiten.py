""" Prewhiten projections.
    converted (and adjusted) from MATLAB module/function "cryo_prewhiten.m".
"""
import math
import numpy
from numpy.core.defchararray import find
from numpy.fft import fft, fftshift, ifft, ifftshift, fft2, ifft2, fftn, ifftn
from numpy.linalg import norm

from helpers.helpers import f_flatten
from preprocessor.exceptions import DimensionsIncompatible


def cryo_prewhiten(stack, noise_response, rel_threshold=None):
    """
    Pre-whiten a stack of projections using the power spectrum of the noise.


    :param stack: stack of images/projections
    :param noise_response: 2d image with the power spectrum of the noise. If all
                           images are to be whitened with respect to the same power spectrum,
                           this is a single image. If each image is to be whitened with respect
                           to a different power spectrum, this is a three-dimensional array with
                           the same number of 2d slices as the stack of images.

    :param rel_threshold: The relative threshold used to determine which frequencies
                          to whiten and which to set to zero. If empty (the default)
                          all filter values less than 100*eps(class(proj)) are
                          zeroed out, while otherwise, all filter values less than
                          threshold times the maximum filter value for each filter
                          is set to zero.

    :return: Pre-whitened stack of images.
    """

    delta = numpy.finfo(float).eps
    num_images = stack.shape[0]
    img_side = stack.shape[1]
    l = math.floor(img_side / 2)
    K = noise_response.shape[1]
    k = math.ceil(K / 2)

    if noise_response.shape[0] not in [1, num_images]:
        raise DimensionsIncompatible('The number of filters must be either 1 or same as number of images!')

    # The whitening filter is the sqrt of of the power spectrum of the noise.
    # Also, normalized the enetgy of the filter to one.
    filter = numpy.sqrt(noise_response)
    filter = filter / norm(f_flatten(filter))

    # The power spectrum of the noise must be positive, and then, the values
    # in filter are all real. If they are not, this means that noise_response
    # had negative values so abort.
    assert (norm(numpy.imag(f_flatten(filter))) < 10 * delta * filter.shape[0])  # Allow loosing one digit
    filter = numpy.real(filter)  # Get rid of tiny imaginary components, if any.

    # The filter should be cicularly symmetric. In particular, it is up-down
    # and left-right symmetric.
    assert (norm(filter - numpy.flipud(filter)) < 10 * delta)
    assert (norm(filter - numpy.fliplr(filter)) < 10 * delta)

    # Get rid of any tiny asymmetries in the filter.
    filter = (filter + numpy.flipud(filter)) / 2
    filter = (filter + numpy.fliplr(filter)) / 2

    # The filter may have very small values or even zeros. We don't want to
    # process these so make a list of all large entries
    if rel_threshold:
        # from MATLAB:
        # nzidx = find(bsxfun( @ gt, filter, rel_threshold * max(max(filter, [], 1), [], 2)));
        raise NotImplementedError('You can use default threshold by omitting re_threshold from kw/args.')
    else:
        noise_idx = find(filter > 100 * delta)

    noise_idx = f_flatten(noise_idx)
    fnz = [x for x in noise_idx if x != 0]

    # Pad the input projections
    pp = numpy.zeros(K)
    p2 = numpy.zeros(img_side, img_side, num_images)

    for i in range(num_images):

        # Zero pad the image to double the size
        if numpy.mod(img_side, 2) == 1:  # Odd-sized image
            pp[k - l: k + l, k - l: k + l] = stack[i, :, :]
        else:
            pp[k - l: k + l - 1, k - l: k + l - 1] = stack[i:, :, :]


        #fp = cfft2(pp)
        p = numpy.zeros(fp.shape)

        # Divide the image by the whitening filter, but onlyin places where the filter is
        # large. In frequnecies where the filter is tiny  we cannot pre-whiten so we just put zero.
        p(nzidx) = bsxfun( @ times, fp(nzidx), 1. / fnz)
        pp2 = icfft2(p)  # pp2 for padded p2
        assert (norm(imag(pp2(:))) / norm(pp2(:)) < 1.0e-13)  # The resulting image should be real.

        if numpy.mod(img_side, 2) == 1:
            p2[i, :, :] = pp2[k - l: k + l, k - l: k + l]
        else:
            p2[i, :, :] = pp2[k - l: k + l - 1, k - l: k + l - 1]

    return numpy.real(p2)
