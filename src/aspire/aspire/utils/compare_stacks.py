import numpy as np

from console_progressbar import ProgressBar

from aspire.aspire.common.config import AspireConfig
from aspire.aspire.common.exceptions import ErrorTooBig, WrongInput, DimensionsIncompatible
from aspire.aspire.common.logger import logger
from aspire.aspire.utils.data_utils import load_stack_from_file
from aspire.aspire.utils.helpers import accepts


@accepts(np.ndarray, np.ndarray, int, float)
def compare_stacks(stack1, stack2, verbose=None, max_error=None):
    """ Calculate the difference between two projection-stacks.
        Return the relative error between them.

        :param stack1: first stack to compare
        :param stack2: second stack to compare
        :param verbose:  level of verbosity
               verbose=0   silent
               verbose=1   show progress bar
               verbose=2   print progress every 1000 images
               verbose=3   print message for each processed image
        :param max_error:  when given, raise an exception if difference between stacks is too big
        :return: returns the accumulative error between the two stacks
    """

    if max_error is not None:
        try:
            max_error = np.longdouble(max_error)
        except (TypeError, ValueError):
            raise WrongInput("max_error must be either a float or an integer!")

    if verbose is None:
        verbose = AspireConfig.verbosity

    # check the dimensions of the stack are compatible
    if stack1.shape != stack2.shape:
        raise DimensionsIncompatible("Can't compare stacks of different sizes!"
                                     f" {stack1.shape} != {stack2.shape}")

    num_of_images = stack1.shape[0]
    if num_of_images == 0:
        logger.warning('stacks are empty!')

    if verbose == 1:
        pb = ProgressBar(total=100, prefix='comparing:', suffix='completed', decimals=0, length=100,
                         fill='%')

    relative_err = 0
    accumulated_err = 0
    for i in range(num_of_images):

        err = np.linalg.norm(stack1[i] - stack2[i])/np.linalg.norm(stack1[i])
        accumulated_err += err
        relative_err = accumulated_err / (i+1)

        # if we already reached a relatively big error, we can stop here
        # we can't ask "if max_error" as max_error is so small and treated as 0 (False)
        if max_error is not None and relative_err > max_error:
            raise ErrorTooBig('Stacks comparison failed! error is too big: {}'.format(relative_err))

        if verbose == 0:
            continue

        elif verbose == 1:
            pb.print_progress_bar((i + 1) / num_of_images * 100)

        elif verbose == 2 and (i+1) % 100 == 0:
            logger.info(f'Finished comparing {i+1}/{num_of_images} projections. '
                        f'Relative error so far: {relative_err}')

        elif verbose == 3:
            logger.info(f'Difference between projections ({i+1}) <> ({i+1}): {err}')

    if verbose == 2:
        logger.info(f'Finished comparing {num_of_images}/{num_of_images} projections. '
                    f'Relative error: {relative_err}')

    return relative_err


def compare_stack_files(file1, file2, verbose=None, max_error=None):
    """ Wrapper for func compare_stacks. """

    stack1 = load_stack_from_file(file1)
    stack2 = load_stack_from_file(file2)
    return compare_stacks(stack1, stack2, verbose=verbose, max_error=max_error)
