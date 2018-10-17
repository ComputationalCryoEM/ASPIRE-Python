import os
import mrcfile

from aspire.common.config import CropStackConfig
from aspire.common.logger import logger
from aspire.preprocessor import cryo_global_phase_flip_mrc_stack
from aspire.preprocessor.crop import crop


# TODO impolement decorator
from aspire.preprocessor.downsample import downsample


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
        output_mrc_file = f'{input_file_prefix}_cropped{input_file_suffix}'

    if os.path.exists(output_mrc_file):
        logger.error(f"output file '{output_mrc_file}' already exists!")
        return

    in_stack = mrcfile.open(input_mrc_file).data
    fill_value = fill_value or CropStackConfig.fill_value
    out_stack = crop(in_stack, size, stack=True, fill_value=fill_value)

    action = 'cropped' if in_stack.shape[2] > size else 'padded'
    logger.info(f"{action} stack from size {in_stack.shape} to size {out_stack.shape}."
                f" saving to {output_mrc_file}..")

    with mrcfile.new(output_mrc_file) as mrc:
        mrc.set_data(out_stack)
    logger.debug(f"saved to {output_mrc_file}")


def downsample_mrc_file(input_mrc_file, side, output_mrc_file=None, mask=None):
    if output_mrc_file is None:
        split = input_mrc_file.rsplit(".", maxsplit=1)
        input_file_prefix = split[0]
        # take care of stack files without suffix such as .mrc
        input_file_suffix = '.' + split[1] if len(split) == 2 else ''
        output_mrc_file = f'{input_file_prefix}_downsampled{input_file_suffix}'

    if os.path.exists(output_mrc_file):
        logger.error(f"output file '{output_mrc_file}' already exists!")
        return

    if mask:
        if os.path.exists(mask):
            mask = mrcfile.open(mask).data
        else:
            logger.error(f"mask file {mask} doesn't exist!")
            return

    in_stack = mrcfile.open(input_mrc_file).data
    # TODO route input arg compute_fx from CLI to downsample
    out_stack = downsample(in_stack, side, compute_fx=False, stack=True, mask=mask)
    logger.info(f"downsampled stack to size {side}x{side}. saving to {output_mrc_file}..")

    with mrcfile.new(output_mrc_file) as mrc:
        mrc.set_data(out_stack)
    logger.debug(f"saved to {output_mrc_file}")
