""" Down/up sample projections.
    converted from MATLAB module/function "cryo_compare_stacks.m".
"""
import numpy
from numpy.fft import fftshift, fftn, ifftn, ifftshift

from preprocessor.cryo_crop import cryo_crop
from preprocessor.exceptions import DimensionsError


def cryo_downsample(img, szout, comopute_fx=False, stack=False, mask=None):
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

    # TODO ask Yoel if this is necessary in Numpy
    # if isinstance(img, int):
    #     img = numpy.single(img)
    #     img = img - numpy.mean(img[:])

    ndim = sum([True for i in img.shape if i > 1])  # number of non-singleton dimensions

    if ndim not in [1, 2, 3]:
        raise DimensionsError(f"Can't downsample image with {ndim} dimensions!")

    nim = 1
    if stack:
        nim = img.shape[0]  # Z axis is the stack size
        ndim -= 1  # 3D turns into a stack of 2D images

    size_img = img.shape[: ndim + 1]

    # force into array
    if isinstance(szout, int):
        szout = numpy.array([[szout]])  # shape now is (1, 1)

    szout = numpy.resize(szout.T, szout.size).conjugate()  # force a *row* vector of size ndim

    if szout.size < ndim:
        szout = szout[0] * numpy.ones(ndim)

    if ndim == 1:
        # force input to be a column vector in the 1d case
        # we transpose to align with Matlab ( "img=img(:)" )
            img = numpy.resize(img.T, img.size)  # TODO should we conjugate like before?

    copy = False
    if numpy.all(szout == img.shape):  # no change in shape
        if not comopute_fx:
            return img

        copy = True

    if numpy.all(szout <= img.shape):  # scale down
        down = True

    elif numpy.all(img.shape < szout):  # scale up
        down = False

    else:  # make sure we don't scale down and up at the same time
        raise DimensionsError("Can't scale up and down at the same time!")

    # scaling down: crop mask to be the size of output
    mask = cryo_crop(mask, szout) if mask else 1

    # todo remove print after review
    print(f'mask.shape:{mask.shape if mask!=1 else 1}')

    if not copy:
        # translated from Matlab "if ~isa(img, 'double')"
        # TODO ask Yoel about the type condition (double/single)
        out = numpy.zeros([szout, nim])

    # TODO progress bar for long operations?

    if ndim == 3:
        if down:  # scale down
            for i in range(nim):
                # TODO verify with Yoel why originally it was img(:, :, :, i)?
                # TODO is this necessarily a stack? (3D is always a stack?)
                x = numpy.fft.fftshift(numpy.fft.fftn(img[:, :, i]))
                fx = cryo_crop(x, szout) * mask
                if copy:
                    out[:, :, :, i] = numpy.fft.ifftn(numpy.fft.ifftshift(fx))\
                                      * (numpy.prod(szout) / numpy.prod(img.shape))

        else:  # scale up
            ...
            raise NotImplemented()

    elif ndim == 2:
        if down:
            for i in range(nim):
                fx = cryo_crop(fftshift(fftn(img[i, :, :])), szout) * mask
                if not copy:
                    out[i, :, :] = (ifftn(ifftshift(fx))
                                    * (numpy.prod(szout) / numpy.prod(img.shape)))
        else:
            ...

    elif ndim == 1:
        if down:
            ...
        else:
            ...

    else:
        raise DimensionsError(f"not sure how to handle ndim of size {ndim}!")

    if comopute_fx:
        numpy.fft.ifftshift(fx)

    if numpy.all(numpy.isreal(img)):
        return numpy.real(out)
