""" Down/up sample projections.
    converted (and adjusted) from MATLAB module/function "cryo_compare_stacks.m".
"""

import numpy
from numpy.fft import fft, fftshift, ifft, ifftshift, fft2, ifft2, fftn, ifftn

from preprocessor.cryo_crop import cryo_crop
from preprocessor.exceptions import DimensionsIncompatible
from helpers.helpers import f_flatten, TupleCompare


def cryo_downsample(img, side, compute_fx=False, stack=False, mask=None):
    """ Use Fourier methods to change the sample interval and/or aspect ratio
        of any dimensions of the input image 'img'. If the optional argument
        stack is set to True, then the last dimension of 'img' is interpreted as the index of each
        image in the stack. The size argument szout is either a scalar or a
        vector of the dimension of the output images.  Let the size of a stack
        of 2D images 'img' be n1 x n1 x ni.  The size of the output (szout=n or
        szout=[n n]) will be n x n x ni. The size argument szout can be chosen
        to change the aspect ratio of the output; however the routine will not
        allow one dimension to be scaled down and another scaled up.

        If the optional mask argument is given, this is used as the
        zero-centered Fourier mask for the re-sampling.  The size of mask should
        be the same as the output image size. For example for downsampling an
        n0 x n0 image with a 0.9 x nyquist filter, do the following:
            msk = fuzzymask(n,2,.45*n,.05*n)
            out = cryo_downsample(img, n, 0, msk)
            The size of the mask must be the size of out. The optional fx output
            argument is the padded or cropped, masked, FT of in, with zero
            frequency at the origin.

    """

    try:
        side = int(side)
    except ValueError:
        raise ValueError("side should be an integer!")

    if not isinstance(stack, bool):
        raise TypeError("stack should be a bool! set it to either True/False.")

    if mask is not None and mask.shape != img.shape:
        raise DimensionsIncompatible(f'Dimensions incompatible! mask shape={mask.shape}, img shape={img.shape}.')

    ndim = sum([True for i in img.shape if i > 1])  # number of non-singleton dimensions
    if ndim not in [1, 2, 3]:
        raise DimensionsIncompatible(f"Can't downsample image with {ndim} dimensions!")

    if ndim == 1:
        szout = (1, side)  # this is the shape of the final vector
    elif ndim == 2 or ndim == 3 and stack:
        szout = (side, side)  # this is the shape of the final mat
    else:  # ndim == 3 and not stack
        szout = numpy.array([side, side, side])  # this is the shape of the final cube

    if ndim == 1:
        # force input img into row vector with the shape (1, img.size)
        img = numpy.asmatrix(f_flatten(img))

    # check sizes of input and output
    szin = img[0, :, :].shape if stack else img.shape

    if TupleCompare.eq(szout, szin):  # no change in shape
        if not compute_fx:
            return img

    # todo should we remove this? On MATLAB this is mandatory for scaling up/down. Here it seems to not be needed.
    # if TupleCompare.lt(szout, szin, eq=True):
    #     down = True  # scale down
    #
    # elif TupleCompare.gt(szout, szin, eq=True):
    #     down = False  # scale up
    #
    # else:  # make sure we don't scale down and up at the same time
    #     raise DimensionsIncompatible("Can't scale up and down at the same time!")

    # adjust mask to be the size of desired output
    mask = cryo_crop(mask, side) if mask else 1

    if ndim == 1:
        # return a vector scaled from the original vector
        x = fftshift(fft(img))
        fx = cryo_crop(x, side) * mask
        out = ifft(ifftshift(fx), axis=0) * (numpy.prod(szout) / numpy.prod(szin))

    elif ndim == 2:
        # return a 2D image scaled from the original image
        fx = cryo_crop(fftshift(fft2(img)), side) * mask
        out = ifft2(ifftshift(fx)) * (numpy.prod(szout) / numpy.prod(szin))

    elif ndim == 3 and stack:
        # return a stack of 2D images where each one of them is downsampled
        num_images = img.shape[0]
        out = numpy.zeros([num_images, side, side], dtype=complex)
        for i in range(num_images):
            fx = cryo_crop(fftshift(fft2(img[i, :, :])), side) * mask
            out[i, :, :] = ifft2(ifftshift(fx)) * (numpy.prod(szout) / numpy.prod(szin))

    else:  # ndim == 3 and not stack
        # return a 3D object scaled from the input 3D cube
        fx = cryo_crop(fftshift(fftn(img)), side) * mask
        out = ifftn(ifftshift(fx)) * (numpy.prod(szout) / numpy.prod(szin))

    if numpy.all(numpy.isreal(img)):
        out = numpy.real(out)

    if compute_fx:
        fx = numpy.fft.ifftshift(fx)
        return out, fx

    return out
