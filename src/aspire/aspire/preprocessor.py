import math
import os

import mrcfile
import numpy as np

from box import Box
from console_progressbar import ProgressBar
from numpy import meshgrid, mean
from numpy.core.multiarray import zeros
from numpy.fft import fftshift, fft, ifft, ifftshift, fft2, ifft2, fftn, ifftn
from numpy.ma import sqrt

from aspire.aspire.common.config import PreProcessorConfig, AspireConfig
from aspire.aspire.common.exceptions import DimensionsIncompatible, WrongInput
from aspire.aspire.common.logger import logger
from aspire.aspire.utils.data_utils import load_stack_from_file, validate_square_projections, fctr, \
    c_to_fortran, fortran_to_c
from aspire.aspire.utils.helpers import TupleCompare, set_output_name, yellow
from aspire.aspire.utils.parse_star import read_star

from aspire.aspire.utils.array_utils import (flatten,
                                      radius_norm,
                                      fast_cfft2,
                                      fast_icfft2,
                                      cart2rad,
                                      cryo_epsds)


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
            mat = np.reshape(mat, [mat.size, 1])  # force a column vector
            ns = math.floor(mat.size / 2) - math.floor(n / 2)  # shift term for scaling down
            if ns >= 0:  # cropping
                return mat[ns: ns + n].astype('float32')

            else:  # padding
                result_mat = fill_value * np.ones([n, 1], dtype=complex)
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
                result_mat = fill_value * np.ones([n, n], dtype=complex)
                result_mat[-start_x: mat_x - start_x, -start_y: mat_y - start_y] = mat
                return result_mat.astype('float32')

            else:
                raise DimensionsIncompatible("Can't crop and pad simultaneously!")

        else:  # mat is 3D or a stack of 2D images

            if stack:
                # break down the stack and treat each image as an individual image
                # then return the cropped stack
                result_mat = np.zeros([mat.shape[0], n, n], dtype='float32')
                for img in range(mat.shape[0]):
                    # TODO iterate instead of using recursion. this is too memoery-expensive
                    result_mat[img, :, :] = PreProcessor.crop(mat[img, :, :], n)

                return result_mat.astype('float32')

            else:  # this is a 3D structure
                # crop/pad mat into a new smaller/bigger cell - 'destination cell'
                from_shape = np.array(mat.shape)
                to_shape = np.array((n, n, n))

                ns = np.floor(from_shape / 2) - np.floor(to_shape / 2)
                ns, to_shape = ns.astype(int), to_shape.astype(int)  # can't slice later with float

                if np.all(ns >= 0):  # crop
                    return mat[ns[0]: ns[0]+to_shape[0],
                               ns[1]: ns[1]+to_shape[1],
                               ns[2]: ns[2]+to_shape[2]]

                elif np.all(ns <= 0):  # pad
                    result_mat = fill_value * np.ones([n, n, n], dtype=complex)
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
            szout = np.array([side, side, side])  # this is the shape of the final cube

        if ndim == 1:
            # force input img into row vector with the shape (1, img.size)
            img = np.asmatrix(flatten(img))

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
            out = ifft(ifftshift(fx), axis=0) * (np.prod(szout) / np.prod(szin))

        elif ndim == 2:
            # return a 2D image scaled from the original image
            fx = cls.crop(fftshift(fft2(img)), side) * mask
            out = ifft2(ifftshift(fx)) * (np.prod(szout) / np.prod(szin))

        elif ndim == 3 and stack:
            # return a stack of 2D images where each one of them is downsampled
            num_images = img.shape[0]
            out = np.zeros([num_images, side, side], dtype=complex)
            for i in range(num_images):
                fx = cls.crop(fftshift(fft2(img[i, :, :])), side) * mask
                out[i, :, :] = ifft2(ifftshift(fx)) * (np.prod(szout) / np.prod(szin))

        else:  # ndim == 3 and not stack
            # return a 3D object scaled from the input 3D cube
            fx = cls.crop(fftshift(fftn(img)), side) * mask
            out = ifftn(ifftshift(fx)) * (np.prod(szout) / np.prod(szin))

        if np.all(np.isreal(img)):
            out = np.real(out)

        if compute_fx:
            fx = np.fft.ifftshift(fx)
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
    def global_phaseflip_stack(stack):
        """ Apply global phase flip to an image stack if needed.

        Check if all images in a stack should be globally phase flipped so that
        the molecule corresponds to brighter pixels and the background corresponds
        to darker pixels. This is done by comparing the mean in a small circle
        around the origin (supposed to correspond to the molecule) with the mean
        of the noise, and making sure that the mean of the molecule is larger.

        Examples:
            >> import mrcfile
            >> stack = mrcfile.open('stack.mrcs')
            >> stack = global_phaseflip_stack(stack)

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
    def global_phaseflip_stack_file(cls, stack_file, output_stack_file=None):

        if output_stack_file is None:
            output_stack_file = set_output_name(stack_file, 'g-pf')

        if os.path.exists(output_stack_file):
            raise FileExistsError(f"output file '{yellow(output_stack_file)}' already exists!")

        in_stack = load_stack_from_file(stack_file)
        out_stack = cls.global_phaseflip_stack(in_stack)

        # check if stack was flipped
        if (out_stack[0] == in_stack[0]).all():
            logger.info('not saving new mrc file.')

        else:
            with mrcfile.new(output_stack_file) as mrc:
                mrc.set_data(out_stack)
            logger.info(f"stack is flipped and saved as {yellow(output_stack_file)}")

    @classmethod
    def prewhiten_stack_file(cls, stack_file, output=None):
        if output is None:
            output = "prewhitened.mrc"

        if os.path.exists(output):
            raise FileExistsError(f"output file '{yellow(output)}' already exists!")

        stack = load_stack_from_file(stack_file)

        # TODO adjust to same unified F/C contiguous
        prewhitten_stack = cls.prewhiten_stack(c_to_fortran(stack))
        with mrcfile.new(output) as fh:
            fh.set_data(np.transpose(prewhitten_stack, (2,1,0)).astype('float32'))

    @staticmethod
    def cryo_noise_estimation(projections, radius_of_mask=None):
        p = projections.shape[0]

        if radius_of_mask is None:
            radius_of_mask = p // 2 - 1

        center_polar_samples = cart2rad(p)
        noise_idx = np.where(center_polar_samples >= radius_of_mask)

        power_spectrum, r, r2, x = cryo_epsds(projections, noise_idx, p // 3)
        power_spectrum = np.real(power_spectrum)

        return power_spectrum, r, r2

    @classmethod
    def prewhiten_stack(cls, stack):
        noise_response, _, _ = cls.cryo_noise_estimation(stack)
        output_images, _, _ = cls.cryo_prewhiten(stack, noise_response)
        return output_images

    @classmethod
    def cryo_prewhiten(cls, proj, noise_response, rel_threshold=None):
        """
        Pre-whiten a stack of projections using the power spectrum of the noise.


        :param proj: stack of images/projections
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

        delta = np.finfo(proj.dtype).eps

        resolution, _, num_images = proj.shape
        l = resolution // 2
        k = int(np.ceil(noise_response.shape[0] / 2))

        filter_var = np.sqrt(noise_response)
        filter_var /= np.linalg.norm(filter_var)

        filter_var = (filter_var + np.flipud(filter_var)) / 2
        filter_var = (filter_var + np.fliplr(filter_var)) / 2

        if rel_threshold is None:
            nzidx = np.where(filter_var > 100 * delta)
        else:
            raise NotImplementedError('not implemented for rel_threshold != None')

        start_idx = k - l - 1
        end_idx = k + l
        if resolution % 2 == 0:
            end_idx -= 1

        fnz = filter_var[nzidx]
        one_over_fnz = 1 / fnz

        # matrix with 1/fnz in nzidx, 0 elsewhere
        one_over_fnz_as_mat = np.zeros((noise_response.shape[0], noise_response.shape[0]))
        one_over_fnz_as_mat[nzidx] += one_over_fnz
        pp = np.zeros((noise_response.shape[0], noise_response.shape[0]))
        p2 = np.zeros((num_images, resolution, resolution), dtype='complex128')
        proj = proj.transpose((2, 0, 1)).copy()

        pb = ProgressBar(total=100, prefix='whitening', suffix='completed',
                         decimals=0, length=100, fill='%')

        for i in range(num_images):
            pp[start_idx:end_idx, start_idx:end_idx] = proj[i]

            fp = fast_cfft2(pp)
            fp *= one_over_fnz_as_mat
            pp2 = fast_icfft2(fp)

            p2[i] = pp2[start_idx:end_idx, start_idx:end_idx]
            if AspireConfig.verbosity == 1:
                pb.print_progress_bar((i + 1) / num_images * 100)

        # change back to x,y,z convention
        proj = p2.real.transpose((1, 2, 0)).copy()
        return proj, filter_var, nzidx

    @classmethod
    def phaseflip_star_file(cls, star_file, pixel_size=None):
        """
            todo add verbosity
        """
        # star is a list of star lines describing projections
        star_records = read_star(star_file)['__root__']

        num_projections = len(star_records)
        projs_init = False  # has the stack been initialized already

        last_processed_stack = None
        for idx in range(num_projections):
            # Get the identification string of the next image to process.
            # This is composed from the index of the image within an image stack,
            #  followed by '@' and followed by the filename of the MRC stack.
            image_id = star_records[idx].rlnImageName
            image_parts = image_id.split('@')
            image_idx = int(image_parts[0]) - 1
            stack_name = image_parts[1]

            # Read the image stack from the disk, if different from the current one.
            # TODO can we revert this condition to positive? what they're equal?
            if stack_name != last_processed_stack:
                mrc_path = os.path.join(os.path.dirname(star_file), stack_name)
                stack = load_stack_from_file(mrc_path)
                logger.info(f"flipping stack in {yellow(os.path.basename(mrc_path))}"
                            f" - {stack.shape}")
                last_processed_stack = stack_name

            if image_idx > stack.shape[2]:
                raise DimensionsIncompatible(f'projection {image_idx} in '
                                             f'stack {stack_name} does not exist')

            proj = stack[image_idx]
            validate_square_projections(proj)
            side = proj.shape[1]

            if not projs_init:  # TODO why not initialize before loop (maybe b/c of huge stacks?)
                # projections was "PFprojs" originally
                projections = np.zeros((num_projections, side, side), dtype='float32')
                projs_init = True

            star_record_data = Box(cryo_parse_Relion_CTF_struct(star_records[idx]))

            if pixel_size is None:
                if star_record_data.tmppixA != -1:
                    pixel_size = star_record_data.tmppixA

                else:
                    raise WrongInput("Pixel size not provided and does not appear in STAR file")

            h = cryo_CTF_Relion(side, star_record_data)
            imhat = fftshift(fft2(proj))
            pfim = ifft2(ifftshift(imhat * np.sign(h)))

            if side % 2 == 1:
                # This test is only vali for odd n
                # images are single precision
                imaginery_comp = np.norm(np.imag(pfim[:])) / np.norm(pfim[:])
                if imaginery_comp > 5.0e-7:
                    logger.warning(f"Large imaginary components in image {image_idx}"
                                   f" in stack {stack_name} = {imaginery_comp}")

            pfim = np.real(pfim)
            projections[idx, :, :] = pfim.astype('float32')

        return projections

    @classmethod
    def normalize_background(cls, stack, radius=None):
        """
            Normalize background to mean 0 and std 1.
            Estimate the mean and std of each image in the stack using pixels
            outside radius r (pixels), and normalize the image such that the
            background has mean 0 and std 1. Each image in the stack is corrected
            separately.

            :param radius: radius for normalization (default is half the side of image)

            Example:
            normalized_stack = cryo_normalize_background(stack,55);
        """
        validate_square_projections(stack)
        num_images = stack.shape[0]  # assuming C-contiguous array
        side = stack.shape[1]

        if radius is None:
            radius = np.floor(side/2)

        # find indices of backgruond pixels in the images
        ctr = (side + 1) / 2
        x_axis, y_axis = np.meshgrid(range(1, side+1), range(1, side+1))
        radiisq = (x_axis.flatten() - ctr) ** 2 + (y_axis.flatten() - ctr) ** 2
        background_pixels_idx = radiisq > radius * radius

        if AspireConfig.verbosity == 1:
            pb = ProgressBar(total=100, prefix='normalizing background', suffix='completed',
                             decimals=0, length=100, fill='%')

        else:
            pb = None

        normalized_stack = np.ones(stack.shape)
        for i in range(num_images):
            if pb:
                pb.print_progress_bar((i + 1) / num_images * 100)

            proj = stack[i, :, :]
            background_pixels = proj.flatten() * background_pixels_idx
            background_pixels = background_pixels[background_pixels != 0]

            # compute mean and standard deviation of background pixels
            proj_mean = np.mean(background_pixels)
            std = np.std(background_pixels, ddof=1)

            # normalize the projections
            if std < 1.0e-5:
                logger.warning(f'Variance of background of image {i} is too small (std={std}). '
                               'Cannot normalize!')

            normalized_stack[i, :, :] = (proj - proj_mean) / std

        return normalized_stack


def cryo_parse_Relion_CTF_struct(star_record):
    voltage = star_record.rlnVoltage
    DefocusU = star_record.rlnDefocusU/10  # Relion uses Angstrom. Convert to nm

    if hasattr(star_record, 'rlnDefocusV'):
        DefocusV = star_record.rlnDefocusV/10  # Relion uses Angstrom. Convert to nm
    else:
        DefocusV = DefocusU

    if hasattr(star_record, 'rlnDefocusAngle'):
        DefocusAngle = star_record.rlnDefocusAngle * math.pi/180  # convert to radians
    else:
        DefocusAngle = 0

    spherical_aberration = star_record.rlnSphericalAberration  # in mm, no conversion is needed
    pixel_size = None

    if hasattr(star_record, 'rlnDetectorPixelSize'):
        pixel_size = star_record.rlnDetectorPixelSize  # in Microns
        mag = star_record.rlnMagnification
        pixel_size = pixel_size * 10**4 / mag  # in Angstrem
    elif hasattr(star_record, 'pixA'):
        pixel_size = star_record.pixA

    return {
        'amplitude_contrast': star_record.rlnAmplitudeContrast,
        'voltage': voltage,
        'DefocusU': DefocusU,
        'DefocusV': DefocusV,
        'DefocusAngle': DefocusAngle,
        'spherical_aberration': spherical_aberration,
        'tmppixA': pixel_size,
        'pixel_size': pixel_size,
        }


def cryo_CTF_Relion(square_side, star_record):
    """
        Compute the contrast transfer function corresponding an n x n image with
        the sampling interval DetectorPixelSize.

    """
    #  wavelength in nm
    wave_length = 1.22639 / math.sqrt(star_record.voltage * 1000 + 0.97845 * star_record.voltage**2)

    # Divide by 10 to make pixel size in nm. BW is the bandwidth of
    #  the signal corresponding to the given pixel size
    bw = 1 / (star_record.pixel_size / 10)

    s, theta = radius_norm(square_side, origin=fctr(square_side))

    # RadiusNorm returns radii such that when multiplied by the
    #  bandwidth of the signal, we get the correct radial frequnecies
    #  corresponding to each pixel in our nxn grid.
    s = s * bw

    DFavg = (star_record.DefocusU + star_record.DefocusV) / 2
    DFdiff = (star_record.DefocusU - star_record.DefocusV)
    df = DFavg + DFdiff * np.cos(2 * (theta - star_record.DefocusAngle)) / 2
    k2 = math.pi * wave_length * df
    # 10**6 converts spherical_aberration from mm to nm
    k4 = math.pi / 2*10**6 * star_record.spherical_aberration * wave_length**3
    chi = k4 * s**4 - k2 * s**2

    return (sqrt(1 - star_record.amplitude_contrast ** 2) * np.sin(chi)
            - star_record.amplitude_contrast * np.cos(chi))
