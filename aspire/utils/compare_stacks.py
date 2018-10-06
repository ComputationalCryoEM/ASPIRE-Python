import mrcfile
import numpy as np

from aspire.common.logger import logger
from console_progressbar import ProgressBar
from os.path import basename


def cryo_compare_mrc_files(mrcname1, mrcname2, verbose=0, max_err=None):
    """ Compare two MRC stacks stored in the files mrcname1 and mrcname2 image by image.
        Return the relative error between the stacks.

        :param mrcname1: first mrc file to compare
        :param mrcname2: second mrc file to compare
        :param verbose:  level of verbosity
               verbose=0   silent
               verbose=1   show progress bar
               verbose=2   print progress every 1000 images
               verbose=3   print message for each processed image
        :param max_err:  when given, raise an exception if difference between stacks is too big
        :return: returns the accumulative error between the two stacks
    """

    if max_err is not None:
        try:
            max_err = np.longdouble(max_err)
        except (TypeError, ValueError):
            raise Exception("max_err must be either a float or an integer!")

    stack1 = mrcfile.open(mrcname1)
    stack2 = mrcfile.open(mrcname2)

    # check the dimensions of the stack are compatible
    dimension_map = {0: 'x', 1: 'y', 2: 'z'}
    for dimension in range(3):
        if stack1.data.shape[dimension] != stack2.data.shape[dimension]:
            raise Exception(f'{dimension_map[dimension]} dimension in both stacks is not '
                            f'compatible: {mrcname1} has {stack1.data.shape[dimension]} pixels, '
                            f'{mrcname2} has {stack2.data.shape[dimension]} pixels')

    num_of_images = stack1.data.shape[2]
    if num_of_images == 0:
        logger.warning('stacks are empty!')

    if verbose == 1:
        pb = ProgressBar(total=100, prefix='comparing:', suffix='completed', decimals=0, length=100,
                         fill='%')

    relative_err = 0
    accumulated_err = 0
    for i in range(num_of_images):

        err = np.linalg.norm(stack1.data[i] - stack2.data[i])
        accumulated_err += err
        relative_err = accumulated_err / (i+1)

        # if we already reached a relatively big error, we can stop here
        # we can't ask "if max_err" as max_err is so small and treated as 0 (False)
        if max_err is not None and relative_err > max_err:
            raise Exception(f'Stacks comparison failed! error is too big. {relative_err}')

        if verbose == 0:
            continue

        elif verbose == 1:
            pb.print_progress_bar((i + 1) / num_of_images * 100)

        elif verbose == 2 and (i+1) % 100 == 0:
            logger.info(f'Finished comparing {i+1}/{num_of_images} projections. '
                        f'Relative error so far: {relative_err}')

        elif verbose == 3:
            logger.info(f'Difference between projections {basename(mrcname1)}({i+1})< '
                        f'>{basename(mrcname2)}({i+1}): {err}')

    if verbose == 2:
        logger.info(f'Finished comparing {num_of_images}/{num_of_images} projections. '
                    f'Relative error: {relative_err}')

    return relative_err
