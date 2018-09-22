# converted from MATLAB func "cryo_globalphaseflip.m"
import argparse
import logging
import sys

import mrcfile
from numpy import meshgrid, zeros, mean
from numpy.ma import sqrt

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)


def cryo_global_phase_flip(stack):
    """ Apply global phase flip to an image stack if needed.

    Check if all images in a stack should be globally phase flipped so that
    the molecule corresponds to brighter pixels and the background corresponds
    to darker pixels. This is done by comparing the mean in a small circle
    around the origin (supposed to correspond to the molecule) with the mean
    of the noise, and making sure that the mean of the molecule is larger.

    Examples:
        >> import mrcfile
        >> stack = mrcfile.open('stack.mrcs')
        >> stack = cryo_global_phase_flip(stack)

    :param stack: stack of images to phaseflip if needed
    :return: stack which might be phaseflipped when needed
    """

    if not len(stack.shape) in [2, 3]:
        raise Exception('illegal stack size/shape! stack should be either 2 or 3 dimensional. '
                        f'(stack shape:{stack.shape})')

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
            logger.info('phaseflipping stack..')
            return -stack

    return stack


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply global phase flip to an image stack in mrc',
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("mrcfile", help="mrc file containing the stack to flip")
    args = parser.parse_args()

    mrc_file_handle = mrcfile.open(args.mrcfile)
    stack = cryo_global_phase_flip(mrc_file_handle.data)

    if stack[0].all() == mrc_file_handle.data[0].all():
        print('stack was not phaseflipped.')
        sys.exit()

    input_file_prefix, input_file_suffix = args.mrcfile.rsplit(".", maxsplit=1)
    output_file_name = f'{input_file_prefix}_phaseflipped.{input_file_suffix}'

    mrcfile.new(output_file_name, stack)

    logger.info(f"stack is flipped and saved as ({output_file_name}).")
