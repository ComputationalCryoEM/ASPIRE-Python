#!/usr/bin/env python3
import logging
import os
import sys
import click
import mrcfile
import numpy as np

try:
    import finufftpy
except ImportError:
    print(f"Can't import finufftpy! Please run 'make finufftpy' and try again.")
    sys.exit(2)

from aspire.abinitio import Abinitio
from aspire.common.logger import logger
from aspire.common.config import AspireConfig, PreProcessorConfig
from aspire.class_averaging import ClassAverages
from aspire.preprocessor import PreProcessor
from aspire.utils.compare_stacks import compare_stack_files
from aspire.utils.data_utils import load_stack_from_file
from aspire.utils.helpers import yellow, requires_binaries, set_output_name
from aspire.utils.viewstack import view_stack


np.random.seed(1137)
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(chain=False, context_settings=CONTEXT_SETTINGS)
@click.option('--debug/--no-debug', default=False, help="Default is --no-debug.")
@click.option('-v', '--verbosity', default=0, help='Verbosity level (0-3).')
def simple_cli(debug, verbosity):
    """\b
        \033[1;33m ASPIRE-Python \033[0m
        \b
        ASPIRE tool accepts one command at a time, executes it and terminates.
        \b
        To view the help message of a command, simply type:
        $ python aspire.py <cmd> -h

        \b
        To view the full docs, please visit
        https://aspire-python.readthedocs.io
    """
    AspireConfig.verbosity = verbosity
    if debug:
        logger.setLevel(logging.DEBUG)


@simple_cli.command('abinitio', short_help='Abinitio algorithm')
@click.argument('stack_file', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(exists=False), help='output file name')
def abinitio_cmd(stack_file, output):
    """\b
        ############################
                 Abinitio
        ############################

        Abinitio accepts a stack file, calculates Abinitio algorithm on it and saves
        the results into output file (default adds '_abinitio' to stack name)
    """
    logger.info(f'running abinitio on stack file {stack_file}..')

    if os.path.exists(output):
        logger.error(f"file {yellow(output)} already exsits! remove first "
                     "or use another name with '-o NAME'")
        return

    stack = load_stack_from_file(stack_file)
    output_stack = Abinitio.cryo_abinitio_c1_worker(stack)

    with mrcfile.new(output) as mrc_fh:
        mrc_fh.set_data(output_stack.astype('float32'))

    logger.info(f"saved to {yellow(output)}.")


@simple_cli.command('classify', short_help='Classification-Averaging algorithm')
@click.argument('stack_file', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(exists=False),
              help='output file name')
@click.option("--avg_nn", default=50,
              help="Number of images to average into each class. (default=50)")
@click.option("--classification_nn", default=100,
              help=("Number of nearest neighbors to find for each "
                    "image during initial classification. (default=100)"))
@requires_binaries('bessel.npy')
def classify_cmd(stack_file, output, avg_nn, classification_nn):
    """ \b
        ############################
          Classification-Averaging
        ############################

        This command accepts a stack file and calculates the
        classification averaging algorithm.

        \b
        When it's done, it saves 2 files:
            1) The full classified stack
            2) A subset of the classified stack (for faster calculations)

        \b
        Example:
            input - stack.mrc
            output1 - stack_classified.mrc (or use flag -o to override)
            output2 - stack_classified_subset.mrc
    """
    if output is None:
        output = set_output_name(stack_file, 'classified')

    if os.path.exists(output):
        logger.error(f"output file {yellow(output)} already exsits! "
                     f"remove first or use another name with '-o NAME' flag")
        return

    subset_output_name = set_output_name(output, 'subset')
    if os.path.exists(subset_output_name):
        logger.error(f"subset file {yellow(subset_output_name)} already exsits! "
                     f"remove first or use another name with '-o NAME' flag")
        return

    logger.info(f'class-averaging {stack_file}..')
    ClassAverages.run(stack_file, output, n_nbor=classification_nn, nn_avg=avg_nn)
    logger.info(f"saved to {yellow(output)}.")
    logger.info(f"saved to {yellow(subset_output_name)}.")


@simple_cli.command('compare', short_help='Compare 2 stack files')
@click.argument('stack_file_1', type=click.Path(exists=True))
@click.argument('stack_file_2', type=click.Path(exists=True))
@click.option('--max-error', default=None, type=float,
              help='if given, raise an error once the err is bigger than given value')
def compare_cmd(stack_file_1, stack_file_2, max_error):
    """ \b
        ############################
              Compare stacks
        ############################

        Calculate the relative error between 2 stack files.
        Stack files can be in MRC/MRCS, NPY or MAT formats.
    """
    logger.info(f"calculating relative err between '{stack_file_1}' and '{stack_file_2}'..")
    relative_err = compare_stack_files(stack_file_1, stack_file_2,
                                       verbose=AspireConfig.verbosity, max_error=max_error)
    logger.info(f"relative err: {relative_err}")


@simple_cli.command('crop', short_help='Crop/Pad projections in stack')
@click.argument('stack_file', type=click.Path(exists=True))
@click.argument('size', type=int)
@click.option('--fill-value', type=float, default=PreProcessorConfig.crop_stack_fill_value)
@click.option('-o', '--output', help="output file name (default adds '_cropped' to input name)")
def crop_cmd(stack_file, size, output, fill_value):
    """ \b
        ############################
                  Crop Stack
        ############################

        Crop projections in stack to squares of 'size x size' px.
        Then save the cropped stack into a new MRC file.
        In case size is bigger than original stack, padding will apply.
        When padding, `--fill-value=VAL` will be used for the padded values. """
    logger.info(f"resizing projections in {stack_file} to {size}x{size}..")
    PreProcessor.crop_stack_file(stack_file, size, output_stack_file=output, fill_value=fill_value)


@simple_cli.command('inspect', short_help='Show stack size/type')
@click.argument('stack_file', type=click.Path(exists=True))
def inspect_cmd(stack_file):
    """ \b
        ############################
                Inspect Stack
        ############################

        Print info about projections in stack file.
    """
    stack, stack_type = load_stack_from_file(stack_file, return_format=True)
    logger.info(f"stack shape: {yellow(stack.shape)}, stack format: {yellow(stack_type)}")


@simple_cli.command('global_phaseflip', short_help='Global-phaseflip stack file')
@click.option('-o', '--output',
              help="output mrc file name (default adds '_g-pf' to input name)")
@click.argument('stack_file', type=click.Path(exists=True))
def global_phaseflip_cmd(stack_file, output):
    """ \b
        ############################
              Global Phaseflip
        ############################

        Apply global phase-flip to a stack file """
    logger.info("calculating global-phaseflip..")
    PreProcessor.global_phaseflip_stack_file(stack_file, output_stack_file=output)


@simple_cli.command('phaseflip', short_help='Read STAR and created unified phaseflipped stack')
@click.option('-o', '--output', help=("output mrc file name (default "
                                      "adds '_phaseflipped' to input name)"))
@click.argument('star_file', type=click.Path(exists=True))
def star_phaseflip_cmd(star_file, output=None):
    """ \b
        ############################
            Phaseflip (STAR file)
        ############################

        \b
        Apply phase-flip to projections in multiple mrc files having a
        STAR file pointing at them.
        After phaseflipping them, they will all be saved in 1 MRC file.
        Default output will add '_phaseflipped.mrc' to star file basename

        \b
        Example:
        ./aspire.py phaseflip ../my_projections/set.star
        will produce file set_phaseflipped.mrc

    """

    if not star_file.endswith('.star'):
        logger.error("input file name doesn't end with '.star'!")

    if output is None:
        # convert 'path/to/foo.star' -> 'foo_phaseflipped.mrc'
        output = '_phaseflipped.mrc'.join(star_file.rsplit('.star', 1))
        output = os.path.basename(output)

    if os.path.exists(output):
        raise FileExistsError(f"output file {yellow(output)} already exists! "
                              "Use flag '-o my_output' or remove file.")
    logger.info("phaseflipping projections..")
    stack = PreProcessor.phaseflip_star_file(star_file)
    with mrcfile.new(output) as fh:
        fh.set_data(stack.astype('float32'))

    logger.info(f"saved {yellow(output)}.")


@simple_cli.command('prewhitten', short_help='Prewhitten projections in stack')
@click.option('-o', '--output', help=("output mrc file name (default "
                                      "adds '_prewhitten' to input name)"))
@click.argument('stack_file', type=click.Path(exists=True))
def star_phaseflip_cmd(stack_file, output=None):
    """ \b
        ############################
            Prewhitten Stack
        ############################

        Prewhitten projections in stack file.

        \b
        Example:
        $ python aspire.py prewhitten projections.mrc
        will produce file projections_prewhitten.mrc

    """

    if output is None:
        output = set_output_name(stack_file, 'prewhitten')

    if os.path.exists(output):
        logger.error(f"output file {yellow(output)} already exsits! "
                     f"remove first or use another name with '-o NAME' flag")
        return

    logger.info("phaseflipping projections..")
    PreProcessor.prewhiten_stack_file(stack_file, output=output)
    logger.info(f"saved {yellow(output)}.")


@simple_cli.command('downsample', short_help='Downsample projections in stack')
@click.argument('stack_file', type=click.Path(exists=True))
@click.argument('side', type=int)
@click.option('--mask', default=None)
@click.option('-o', '--output', help="output file name (default adds '_downsampled' to input name)")
def downsample_cmd(stack_file, side, output, mask):
    """ \b
        ############################
            Downsample Stack
        ############################

        Use Fourier methods to change the sample interval and/or aspect ratio
        of any dimensions of the input projections-stack to the output of SIZE x SIZE.
        If the optional mask argument is given, this is used as the
        zero-centered Fourier mask for the re-sampling. The size of mask should
        be the same as the output image size.
    """
    logger.info(f"downsampling stack {stack_file} to size {side}x{side} px..")
    PreProcessor.downsample_stack_file(stack_file, side, output_stack_file=output, mask_file=mask)


@simple_cli.command('viewstack', short_help='Plot projections in stack file')
@click.argument('stack_file', type=click.Path(exists=True))
@click.option('--numslices', type=int, default=16)
@click.option('--startslice', type=int, default=1)
@click.option('--nrows', type=int, default=4, help="number of rows")
@click.option('--ncols', type=int, default=4, help="number of columns")
def viewstack_cmd(stack_file, numslices, startslice, nrows, ncols):
    """ \b
        ############################
                View Stack
        ############################

        Plot projections using GUI.
    """
    logger.info(f"viewing stack {stack_file}..")
    stack = load_stack_from_file(stack_file)
    view_stack(stack, numslices=numslices, startslice=startslice, nrows=nrows, ncols=ncols)


###################################################################
# The following is the foundation for creating a piped aspire cli
###################################################################
class PipedObj:
    """ This object will be passed between piped commands and be
        used for saving intermediate results and settings.
    """

    def __init__(self, mrc_file, debug, verbosity):
        self.stack = mrcfile.open(mrc_file).data
        self.debug = debug
        AspireConfig.verbosity = verbosity


pass_obj = click.make_pass_decorator(PipedObj, ensure=True)


@click.group(chain=True)
@click.option('--debug/--no-debug', default=False, help="Default is --no-debug.")
@click.option('-v', '--verbosity', default=0, help='Verbosity level (0-3).')
@click.argument('input_mrc')
@click.pass_context
def piped_cli(ctx, input_mrc, debug, verbosity):
    """ Piped cli accepts multiple commands, executes one by one and passes on
        the intermediate results on, between the commands. """
    logger.setLevel(debug)
    ctx.obj = PipedObj(input_mrc, debug, verbosity)  # control log/verbosity per command


@piped_cli.command("global_phaseflip")
@pass_obj
def global_phaseflip_stack(ctx_obj):
    """ Apply global phase-flip to an MRC stack """
    logger.debug("calculating global phaseflip..")
    ctx_obj.stack = PreProcessor.global_phaseflip_stack(ctx_obj.stack)


@piped_cli.command("save")
@click.option('-o', type=click.Path(exists=False), default='output.mrc', help='output file name')
@pass_obj
def chained_save_stack(ctx_obj, o):
    """ Save MRC stack to output file """
    if os.path.exists(o):  # TODO move this check before anything starts running
        logger.error("output file {} already exists! "
                     "please rename/delete or use flag -o with different output name")
        sys.exit(1)

    logger.info("saving stack {}..".format(o))
    mrcfile.new(o, ctx_obj.stack)


if __name__ == "__main__":
    simple_cli()

    # todo this should allow users to funnel a stack into a pipeline of commands
    # piped_cli
