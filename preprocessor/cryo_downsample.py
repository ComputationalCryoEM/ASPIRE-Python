""" Down/up sample projections.
    converted from MATLAB module/function "cryo_compare_stacks.m".

TODO open questions:
1) Do we use compute_fx
2) Do we use scale up?
3) Do we need a progress bar?
"""
import numpy
from numpy.fft import fft, fftshift, ifft, ifftshift, fft2, ifft2

from preprocessor.cryo_crop import cryo_crop
from preprocessor.exceptions import DimensionsError
from helpers.helpers import f_flatten


def cryo_downsample(img, szout, compute_fx=False, stack=False, mask=None):
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
    if not isinstance(stack, bool):
        raise TypeError("stack should be a bool! set it to either True/False.")

    if mask is not None and mask.shape != img.shape:
        raise DimensionsError('Dimensions incompatible! mask '
                              f'shape={mask.shape}, img shape={img.shape}.')

    ndim = sum([True for i in img.shape if i > 1])  # number of non-singleton dimensions
    if ndim not in [1, 2, 3]:
        raise DimensionsError(f"Can't downsample image with {ndim} dimensions!")

    # force into array
    if isinstance(szout, int):
        szout = numpy.array([[szout]])  # shape now is (szout, szout)

    szout = f_flatten(szout).conjugate()  # force a *row* vector of size ndim

    # todo is this needed? so far for 1, 2 and a stack of 2 dimensions this isn't needed
    # if szout.size < ndim:
    #     szout = szout[0] * numpy.ones(ndim)

    if ndim == 1:
        # force input to be a column vector in the 1d case
        img = f_flatten(img)

    copy = False
    if numpy.all(szout == img.shape):  # no change in shape
        out = img
        if not compute_fx:
            return img

        copy = True

    szin = img[0, :, :].shape if stack else img.shape

    if numpy.all(szout <= szin):  # scale down
        down = True

    elif numpy.all(szin < szout):  # scale up
        down = False

    else:  # make sure we don't scale down and up at the same time
        raise DimensionsError("Can't scale up and down at the same time!")

    # scaling down: crop mask to be the size of output
    mask = cryo_crop(mask, szout) if mask else 1

    # TODO progress bar for long operations?

    if ndim == 3:
        if down:  # scale down
            if stack:
                num_images = img.shape[0]
                out = numpy.zeros([num_images, szout[0], szout[0]])
                for i in range(num_images):
                    # IPython.embed()
                    out[i, :, :] = cryo_downsample(img[i, :, :], szout.item())

            else:  # real 3D
                # x = numpy.fft.fftshift(numpy.fft.fftn(img[:, :, i]))
                # fx = cryo_crop(x, szout) * mask
                # if copy:
                #     out[:, :, :, i] = numpy.fft.ifftn(numpy.fft.ifftshift(fx))\
                #                       * (numpy.prod(szout) / numpy.prod(img.shape))
                raise NotImplementedError("scaling up currently isn't supported!")

        else:  # up-sample (scale up)
            raise NotImplementedError("scaling up currently isn't supported!")

    elif ndim == 2:
        if down:
            x = fftshift(fft2(img))
            fx = cryo_crop(x, int(szout[0])) * mask
            out = ifft2(ifftshift(fx)) * (numpy.prod(szout)**2 / numpy.prod(img.shape))

        else:  # up-sample
            raise NotImplementedError("scaling up currently isn't supported!")

    elif ndim == 1:
        if down:
            fx = cryo_crop(fftshift(fft(img)), szout[0]) * mask
            if not copy:
                out = ifft(ifftshift(fx), axis=0) * (numpy.prod(szout) / numpy.prod(img.shape[:ndim]))

        else:  # up-sample
            raise NotImplementedError("scaling up currently isn't supported!")

    else:
        raise DimensionsError(f"Unknown data structure! number of dimensions: {ndim}.")

    if numpy.all(numpy.isreal(img)):
        out = numpy.real(out)

    if compute_fx:
        fx = numpy.fft.ifftshift(fx)
        return out, fx

    return out
