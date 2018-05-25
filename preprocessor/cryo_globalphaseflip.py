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
    """ Apply global phase flip to an image stack.

    Check if all images in a stack should be globally phase flipped so that
    the molecule corresponds to brighter pixels and the background corresponds
    to darker pixels. This is done by comparing the mean in a small circle
    around the origin (supposed to correspond to the molecule) with the mean
    of the noise, and making sure that the mean of the molecule is larger.

    Examples:
        > import mrcfile
        > stack = mrcfile.open('stack.mrcs')
        > res_stack, is_flipped, signal_mean, noise_mean = cryo_global_phase_flip(stack)

    :param stack: stack of images
    :return: res_stack, is_flipped, signal_mean, noise_mean
    """

    if not len(stack.shape) in [2, 3]:
        Exception('illegal stack size/shape! stack should be either 2 or 3 dimensional. '
                  f'(stack shape:{stack.shape})')

    k = stack.shape[2] if len(stack.shape) == 3 else 1

    if stack.shape[0] != stack.shape[1]:
        raise Exception('images must be square!')

    n = stack.shape[0]
    center = (n + 1) / 2
    i, j = meshgrid(n, n)
    r = sqrt((i - center)**2 + (j - center)**2)
    sigind = r < round(n / 4)  # indices of signal samples
    sigind = sigind.astype(int)  # fill_value by default is True/False
    noiseind = r > round(n / 2 * 0.8)
    noiseind = noiseind.astype(int)

    signal_mean = zeros([k, 1])
    noise_mean = zeros([k, 1])
    is_flipped = False

    # import IPython
    # IPython.embed()
    for idx in range(k):
        proj = stack[:, :, idx]
        signal_mean[idx] = mean(proj[sigind])
        noise_mean[idx] = mean(proj[noiseind])

    signal_mean = mean(signal_mean)
    noise_mean = mean(noise_mean)

    if signal_mean < noise_mean:
            is_flipped = True
            stack = -stack

    return stack, is_flipped, signal_mean, noise_mean


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply global phase flip to an image stack in mrc',
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("mrcfile", help="mrc file containing the stack to flip")
    args = parser.parse_args()

    mrc_file_handle = mrcfile.open(args.mrcfile)
    stack, is_flipped, signal_mean, noise_mean = cryo_global_phase_flip(mrc_file_handle.data)

    logger.info("stack:", stack, "is_flipped:", is_flipped, "signal_mean:", signal_mean,
                "noise_mean:", noise_mean)
