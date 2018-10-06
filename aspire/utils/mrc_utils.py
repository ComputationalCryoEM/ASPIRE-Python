import os
import mrcfile

from aspire.common.logger import logger
from aspire.preprocessor import cryo_global_phase_flip_mrc_stack


def cryo_global_phase_flip_mrc_file(input_mrc_file, output_mrc_file=None):

    if output_mrc_file is None:
        split = input_mrc_file.rsplit(".", maxsplit=1)
        input_file_prefix = split[0]
        # take care of stack files without suffix such as .mrc
        input_file_suffix = '.' + split[1] if len(split) == 2 else ''
        output_mrc_file = '{}_phaseflipped{}'.format(input_file_prefix,
                                                     input_file_suffix)
    print(output_mrc_file)
    if os.path.exists(output_mrc_file):
        logger.error("output file '{}' already exists!".format(output_mrc_file))
        return
    
    in_stack = mrcfile.open(input_mrc_file).data
    out_stack = cryo_global_phase_flip_mrc_stack(in_stack)

    if out_stack[0].all() == ~in_stack[0].all():
        mrcfile.new(output_mrc_file, out_stack)
        logger.info("stack is flipped and saved as {}".format(output_mrc_file))

    logger.info('stack was not phaseflipped.')
