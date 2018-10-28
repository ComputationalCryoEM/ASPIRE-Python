import math
import os

import mrcfile
import numpy
from numpy import meshgrid, mean
from numpy.core.multiarray import zeros
from numpy.fft import fftshift, fft, ifft, ifftshift, fft2, ifft2, fftn, ifftn
from numpy.ma import sqrt

from aspire.common.config import PreProcessorConfig
from aspire.common.exceptions import DimensionsIncompatible
from aspire.common.logger import logger
from aspire.utils.data_utils import load_stack_from_file
from aspire.utils.helpers import TupleCompare, set_output_name, yellow
from aspire.utils.array_utils import flatten


class PreProcessor:

    @staticmethod
    def crop(mat, n, stack=False, fill_value=None):
        """
            Reduce the size of a vector, square or cube 'mat' by cropping (or
            increase the size by padding with fill_value, by default zero) to a final
            size of n, (n x n), or (n x n x n) respectively. This is the analogue of down-sample, but
            it doesn't change magnification.

            If mat is 2-dimensional and n is a vector, m is cropped to n=[mat_x mat_y].

            The function handles odd and even-sized arrays correctly. The center of
            an odd array is taken to be at (n+1)/2, and an even array is n/2+1.

            If flag is_stack is set to True, then a 3D array 'mat' is treated as a stack of 2D
            images, and each image is cropped to (n x n).

            For 2D images, the input image doesn't have to be square.

            * The original MATLAB function supported cropping to non-square matrices.
              As real-world uses will always crop to square (n, n), we don't support it with Python.


            Args:
                mat (numpy.array): Vector, 2D array, stack of 2D arrays or a 3D array
                n (int): Size of desired cropped vector, side of 2D array or side of 3D array
                stack (bool): Set to True in order to handle a 3D mat as a stack of 2D
                fill_value (:obj:`int`, optional): Padding value. Defaults to 0.

            Returns:
                numpy.array: Cropped or padded mat to size of n, (n x n) or (n x n x n)

            TODO: change name to 'resize'? crop/pad is confusing b/c either you save different
            TODO:   output names or you stick to a misleading name like cropped.mrc for padded stack
        """

        num_dimensions = len(mat.shape)

        if num_dimensions not in [1, 2, 3]:
            raise DimensionsIncompatible("cropping/padding failed! number of dimensions is too big!"
                                         f" ({num_dimensions} while max is 3).")

        if num_dimensions == 2 and 1 in mat.shape:
            num_dimensions = 1

        if fill_value is None:
            fill_value = PreProcessorConfig.crop_stack_fill_value

        if num_dimensions == 1:  # mat is a vector
            mat = numpy.reshape(mat, [mat.size, 1])  # force a column vector
            ns = math.floor(mat.size / 2) - math.floor(n / 2)  # shift term for scaling down
            if ns >= 0:  # cropping
                return mat[ns: ns + n].astype('float32')

            else:  # padding
                result_mat = fill_value * numpy.ones([n, 1], dtype=complex)
                result_mat[-ns: mat.size - ns] = mat
                return result_mat.astype('float32')

        elif num_dimensions == 2:  # mat is 2D image
            mat_x, mat_y = mat.shape
            # start_x = math.floor(mat_x / 2) - math.floor(n / 2)  # shift term for scaling down
            start_x = mat_x / 2 - n / 2  # shift term for scaling down
            # start_y = math.floor(mat_y / 2) - math.floor(n / 2)
            start_y = mat_y / 2 - n / 2

            if start_x >= 0 and start_y >= 0:  # cropping
                start_x, start_y = math.floor(start_x), math.floor(start_y)
                logger.debug(f'crop:cropping from {mat.shape} to {n}..')
                return mat[start_x: start_x + int(n), start_y: start_y + int(n)].astype('float32')

            elif start_x < 0 and start_y < 0:  # padding
                logger.debug('crop:padding..')
                start_x, start_y = math.floor(start_x), math.floor(start_y)
                result_mat = fill_value * numpy.ones([n, n], dtype=complex)
                result_mat[-start_x: mat_x - start_x, -start_y: mat_y - start_y] = mat
                return result_mat.astype('float32')

            else:
                raise DimensionsIncompatible("Can't crop and pad simultaneously!")

        else:  # mat is 3D or a stack of 2D images

            if stack:
                # break down the stack and treat each image as an individual image
                # then return the cropped stack
                result_mat = numpy.zeros([mat.shape[0], n, n], dtype='float32')
                for img in range(mat.shape[0]):
                    # TODO iterate instead of using recursion. this is too memoery-expensive
                    result_mat[img, :, :] = PreProcessor.crop(mat[img, :, :], n)

                return result_mat.astype('float32')

            else:  # this is a 3D structure
                # crop/pad mat into a new smaller/bigger cell - 'destination cell'
                from_shape = numpy.array(mat.shape)
                to_shape = numpy.array((n, n, n))

                ns = numpy.floor(from_shape / 2) - numpy.floor(to_shape / 2)
                ns, to_shape = ns.astype(int), to_shape.astype(int)  # can't slice later with float

                if numpy.all(ns >= 0):  # crop
                    return mat[ns[0]: ns[0]+to_shape[0],
                               ns[1]: ns[1]+to_shape[1],
                               ns[2]: ns[2]+to_shape[2]]

                elif numpy.all(ns <= 0):  # pad
                    result_mat = fill_value * numpy.ones([n, n, n], dtype=complex)
                    result_mat[-ns[0]: from_shape[0] - ns[0],
                               -ns[1]: from_shape[2] - ns[1],
                               -ns[2]: from_shape[2] - ns[2]] = mat

                    return result_mat.astype('float32')

                else:
                    raise DimensionsIncompatible("Can't crop and pad simultaneously!")

    @classmethod
    def crop_stack(cls, array, size, fill_value=None):
        return cls.crop(array, size, stack=True, fill_value=fill_value)

    @classmethod
    def crop_stack_file(cls, stack_file, size, output_stack_file=None, fill_value=None):

        if output_stack_file is None:
            output_stack_file = set_output_name(stack_file, 'cropped')

        if os.path.exists(output_stack_file):
            raise FileExistsError(f"output file '{yellow(output_stack_file)}' already exists!")

        stack = load_stack_from_file(stack_file)
        fill_value = fill_value or PreProcessorConfig.crop_stack_fill_value
        cropped_stack = cls.crop_stack(stack, size, fill_value=fill_value)

        action = 'cropped' if size < stack.shape[1] else 'padded'
        logger.info(f"{action} stack from size {stack.shape} to size {cropped_stack.shape}."
                    f" saving to {yellow(output_stack_file)}..")

        with mrcfile.new(output_stack_file) as mrc:
            mrc.set_data(cropped_stack)
        logger.debug(f"saved to {output_stack_file}")

    @classmethod
    def downsample(cls, img, side, compute_fx=False, stack=False, mask=None):
        """ Use Fourier methods to change the sample interval and/or aspect ratio
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
            img = numpy.asmatrix(flatten(img))

        # check sizes of input and output
        szin = img[0, :, :].shape if stack else img.shape

        if TupleCompare.eq(szout, szin):  # no change in shape
            if not compute_fx:
                return img

        # adjust mask to be the size of desired output
        mask = cls.crop(mask, side) if mask else 1

        if ndim == 1:
            # return a vector scaled from the original vector
            x = fftshift(fft(img))
            fx = cls.crop(x, side) * mask
            out = ifft(ifftshift(fx), axis=0) * (numpy.prod(szout) / numpy.prod(szin))

        elif ndim == 2:
            # return a 2D image scaled from the original image
            fx = cls.crop(fftshift(fft2(img)), side) * mask
            out = ifft2(ifftshift(fx)) * (numpy.prod(szout) / numpy.prod(szin))

        elif ndim == 3 and stack:
            # return a stack of 2D images where each one of them is downsampled
            num_images = img.shape[0]
            out = numpy.zeros([num_images, side, side], dtype=complex)
            for i in range(num_images):
                fx = cls.crop(fftshift(fft2(img[i, :, :])), side) * mask
                out[i, :, :] = ifft2(ifftshift(fx)) * (numpy.prod(szout) / numpy.prod(szin))

        else:  # ndim == 3 and not stack
            # return a 3D object scaled from the input 3D cube
            fx = cls.crop(fftshift(fftn(img)), side) * mask
            out = ifftn(ifftshift(fx)) * (numpy.prod(szout) / numpy.prod(szin))

        if numpy.all(numpy.isreal(img)):
            out = numpy.real(out)

        if compute_fx:
            fx = numpy.fft.ifftshift(fx)
            return out, fx

        return out.astype('float32')

    @classmethod
    def downsample_stack_file(cls, stack_file, side, output_stack_file=None, mask_file=None):

        if output_stack_file is None:
            output_stack_file = set_output_name(stack_file, 'downsampled')

        if os.path.exists(output_stack_file):
            raise FileExistsError(f"output file '{yellow(output_stack_file)}' already exists!")

        if mask_file:
            if not os.path.exists(mask_file):
                logger.error(f"mask file {yellow(mask_file)} doesn't exist!")
            mask = load_stack_from_file(mask_file)
        else:
            mask = None

        stack = load_stack_from_file(stack_file)
        downsampled_stack = cls.downsample(stack, side, compute_fx=False, stack=True, mask=mask)
        logger.info(f"downsampled stack from size {stack.shape} to {downsampled_stack.shape}."
                    f" saving to {yellow(output_stack_file)}..")

        with mrcfile.new(output_stack_file) as mrc_fh:
            mrc_fh.set_data(downsampled_stack)
        logger.debug(f"saved to {output_stack_file}")

    @staticmethod
    def phaseflip_stack(stack):
        """ Apply global phase flip to an image stack if needed.

        Check if all images in a stack should be globally phase flipped so that
        the molecule corresponds to brighter pixels and the background corresponds
        to darker pixels. This is done by comparing the mean in a small circle
        around the origin (supposed to correspond to the molecule) with the mean
        of the noise, and making sure that the mean of the molecule is larger.

        Examples:
            >> import mrcfile
            >> stack = mrcfile.open('stack.mrcs')
            >> stack = phaseflip_stack(stack)

        :param stack: stack of images to phaseflip if needed
        :return stack: stack which might be phaseflipped when needed
        """

        if not len(stack.shape) in [2, 3]:
            raise Exception('illegal stack size/shape! stack should be either 2 or 3 dimensional. '
                            '(stack shape:{})'.format(stack.shape))

        num_of_images = stack.shape[2] if len(stack.shape) == 3 else 1

        # make sure images are square
        if stack.shape[1] != stack.shape[2]:
            raise Exception(f'images must be square! ({stack.shape[0]}, {stack.shape[1]})')

        image_side_length = stack.shape[0]
        image_center = (image_side_length + 1) / 2
        coor_mat_m, coor_mat_n = meshgrid(range(image_side_length), range(image_side_length))
        distance_from_center = sqrt((coor_mat_m - image_center)**2 + (coor_mat_n - image_center)**2)

        # calculate indices of signal and noise samples assuming molecule is around the center
        signal_indices = distance_from_center < round(image_side_length / 4)
        signal_indices = signal_indices.astype(int)  # fill_value by default is True/False
        noise_indices = distance_from_center > round(image_side_length / 2 * 0.8)
        noise_indices = noise_indices.astype(int)

        signal_mean = zeros([num_of_images, 1])
        noise_mean = zeros([num_of_images, 1])

        for image_idx in range(num_of_images):
            proj = stack[:, :, image_idx]
            signal_mean[image_idx] = mean(proj[signal_indices])
            noise_mean[image_idx] = mean(proj[noise_indices])

        signal_mean = mean(signal_mean)
        noise_mean = mean(noise_mean)

        if signal_mean < noise_mean:
                logger.info('phase-flipping stack..')
                return -stack

        logger.info('no need to phase-flip stack.')
        return stack

    @classmethod
    def phaseflip_stack_file(cls, stack_file, output_stack_file=None):

        if output_stack_file is None:
            output_stack_file = set_output_name(stack_file, 'phaseflipped')

        if os.path.exists(output_stack_file):
            raise FileExistsError(f"output file '{yellow(output_stack_file)}' already exists!")

        in_stack = load_stack_from_file(stack_file)
        out_stack = cls.phaseflip_stack(in_stack)

        # check if stack was flipped
        if (out_stack[0] == in_stack[0]).all():
            logger.info('not saving new mrc file.')

        else:
            with mrcfile.new(output_stack_file) as mrc:
                mrc.set_data(out_stack)
            logger.info(f"stack is flipped and saved as {yellow(output_stack_file)}")

    # def prewhiten(stack, noise_response, rel_threshold=None):
    #   from numpy.core.defchararray import find
    #   """
    #     Pre-whiten a stack of projections using the power spectrum of the noise.
    #
    #
    #     :param stack: stack of images/projections
    #     :param noise_response: 2d image with the power spectrum of the noise. If all
    #                            images are to be whitened with respect to the same power spectrum,
    #                            this is a single image. If each image is to be whitened with respect
    #                            to a different power spectrum, this is a three-dimensional array with
    #                            the same number of 2d slices as the stack of images.
    #
    #     :param rel_threshold: The relative threshold used to determine which frequencies
    #                           to whiten and which to set to zero. If empty (the default)
    #                           all filter values less than 100*eps(class(proj)) are
    #                           zeroed out, while otherwise, all filter values less than
    #                           threshold times the maximum filter value for each filter
    #                           is set to zero.
    #
    #     :return: Pre-whitened stack of images.
    #     """
    #
    #     delta = numpy.finfo(float).eps
    #     num_images = stack.shape[0]
    #     img_side = stack.shape[1]
    #     l = math.floor(img_side / 2)
    #     K = noise_response.shape[1]
    #     k = math.ceil(K / 2)
    #
    #     if noise_response.shape[0] not in [1, num_images]:
    #         raise DimensionsIncompatible('The number of filters must be either 1 or same as number of images!')
    #
    #     # The whitening filter is the sqrt of of the power spectrum of the noise.
    #     # Also, normalized the enetgy of the filter to one.
    #     filter = numpy.sqrt(noise_response)
    #     filter = filter / norm(flatten(filter))
    #
    #     # The power spectrum of the noise must be positive, and then, the values
    #     # in filter are all real. If they are not, this means that noise_response
    #     # had negative values so abort.
    #     assert (norm(numpy.imag(flatten(filter))) < 10 * delta * filter.shape[0])  # Allow loosing one digit
    #     filter = numpy.real(filter)  # Get rid of tiny imaginary components, if any.
    #
    #     # The filter should be cicularly symmetric. In particular, it is up-down
    #     # and left-right symmetric.
    #     assert (norm(filter - numpy.flipud(filter)) < 10 * delta)
    #     assert (norm(filter - numpy.fliplr(filter)) < 10 * delta)
    #
    #     # Get rid of any tiny asymmetries in the filter.
    #     filter = (filter + numpy.flipud(filter)) / 2
    #     filter = (filter + numpy.fliplr(filter)) / 2
    #
    #     # The filter may have very small values or even zeros. We don't want to
    #     # process these so make a list of all large entries
    #     if rel_threshold:
    #         # from MATLAB:
    #         # nzidx = find(bsxfun( @ gt, filter, rel_threshold * max(max(filter, [], 1), [], 2)));
    #         raise NotImplementedError('You can use default threshold by omitting re_threshold from kw/args.')
    #     else:
    #         noise_idx = find(filter > 100 * delta)
    #
    #     noise_idx = flatten(noise_idx)
    #     fnz = [x for x in noise_idx if x != 0]
    #
    #     # Pad the input projections
    #     pp = numpy.zeros(K)
    #     p2 = numpy.zeros(img_side, img_side, num_images)
    #
    #     for i in range(num_images):
    #
    #         # Zero pad the image to double the size
    #         if numpy.mod(img_side, 2) == 1:  # Odd-sized image
    #             pp[k - l: k + l, k - l: k + l] = stack[i, :, :]
    #         else:
    #             pp[k - l: k + l - 1, k - l: k + l - 1] = stack[i:, :, :]
    #
    #
    #         #fp = cfft2(pp)
    #         p = numpy.zeros(fp.shape)
    #
    #         # Divide the image by the whitening filter, but onlyin places where the filter is
    #         # large. In frequnecies where the filter is tiny  we cannot pre-whiten so we just put zero.
    #         p(nzidx) = bsxfun( @ times, fp(nzidx), 1. / fnz)
    #         pp2 = icfft2(p)  # pp2 for padded p2
    #         assert (norm(imag(pp2(:))) / norm(pp2(:)) < 1.0e-13)  # The resulting image should be real.
    #
    #         if numpy.mod(img_side, 2) == 1:
    #             p2[i, :, :] = pp2[k - l: k + l, k - l: k + l]
    #         else:
    #             p2[i, :, :] = pp2[k - l: k + l - 1, k - l: k + l - 1]
    #
    #     return numpy.real(p2)