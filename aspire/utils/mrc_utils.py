import os
import mrcfile

from aspire.common.config import CropStackConfig
from aspire.common.logger import logger
from aspire.preprocessor import cryo_global_phase_flip_mrc_stack
from aspire.preprocessor.cryo_crop import cryo_crop


# TODO impolement decorator
def mrc_validator():
    pass


# TODO impolement decorator
def validate_input_output():
    # return stack
    pass


def global_phase_flip_mrc_file(input_mrc_file, output_mrc_file=None):

    if output_mrc_file is None:
        split = input_mrc_file.rsplit(".", maxsplit=1)
        input_file_prefix = split[0]
        # take care of stack files without suffix such as .mrc
        input_file_suffix = '.' + split[1] if len(split) == 2 else ''
        output_mrc_file = '{}_phaseflipped{}'.format(input_file_prefix,
                                                     input_file_suffix)
    if os.path.exists(output_mrc_file):
        logger.error("output file '{}' already exists!".format(output_mrc_file))
        return
    
    in_stack = mrcfile.open(input_mrc_file).data
    out_stack = cryo_global_phase_flip_mrc_stack(in_stack)
    if out_stack[0].all() == ~in_stack[0].all():
        with mrcfile.new(output_mrc_file) as mrc:
            mrc.set_data(out_stack)
        logger.info("stack is flipped and saved as {}".format(output_mrc_file))

    else:
        logger.debug('not saving new mrc file.')


def crop_mrc_file(input_mrc_file, size, output_mrc_file=None, fill_value=None):
    # TODO move check of output file to a decorator
    if output_mrc_file is None:
        split = input_mrc_file.rsplit(".", maxsplit=1)
        input_file_prefix = split[0]
        # take care of stack files without suffix such as .mrc
        input_file_suffix = '.' + split[1] if len(split) == 2 else ''
        output_mrc_file = '{}_phaseflipped{}'.format(input_file_prefix,
                                                     input_file_suffix)
    if os.path.exists(output_mrc_file):
        logger.error("output file '{}' already exists!".format(output_mrc_file))
        return

    in_stack = mrcfile.open(input_mrc_file).data
    fill_value = fill_value or CropStackConfig.fill_value
    out_stack = cryo_crop(in_stack, size, stack=True, fill_value=fill_value)

    action = 'cropped' if in_stack.shape[2] > size else 'padded'
    logger.info("{} stack from size {} to size {}. saving to {}..".format(action,
                                                                          in_stack.shape,
                                                                          out_stack.shape,
                                                                          output_mrc_file))

    with mrcfile.new(output_mrc_file) as mrc:
        mrc.set_data(out_stack)
    logger.debug("saved to {}".format(output_mrc_file))
