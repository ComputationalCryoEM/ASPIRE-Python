#!/usr/bin/env python3
import logging
import os
import sys
import click
import mrcfile

from aspire.common.logger import logger
from aspire.common.config import AspireConfig, CropStackConfig
from aspire.preprocessor import cryo_global_phase_flip_mrc_stack
from aspire.utils.compare_stacks import cryo_compare_mrc_files
from aspire.utils.mrc_utils import global_phase_flip_mrc_file, crop_mrc_file, downsample_mrc_file

try:
    from aspire.class_averaging.averaging import ClassAverages
    finufftpy_imported = True
except ImportError:
    finufftpy_imported = False
    logger.warning("Couldn't import finufftwpy! Some commands will fail!")


@click.group(chain=False)
@click.option('--debug/--no-debug', default=False, help="Default is --no-debug.")
@click.option('-v', '--verbosity', default=0, help='Verbosity level (0-3).')
def simple_cli(debug, verbosity):
    """ Aspire tool accepts one command at a time, executes it and terminates\t
        \n To see usage of a command, simply type 'command --help'\t
        \n e.g. python3 aspire.py classify --help
    """
    AspireConfig.verbosity = verbosity
    if debug:
        logger.setLevel(logging.DEBUG)


@simple_cli.command('classify')
@click.argument('mrc_file', type=click.Path(exists=True))
@click.option('-o', default='classified.mrc', type=click.Path(exists=False),
              help='output file name')
@click.option("--avg_nn", default=50,
              help="Number of images to average into each class. (default=50)")
@click.option("--classification_nn", default=100,
              help=("Number of nearest neighbors to find for each "
                    "image during initial classification. (default=100)"))
@click.option("--k_vdm_in", default=20,
              help="Number of nearest neighbors for building VDM graph. (default=20")
@click.option("--k_vdm_out", default=200,
              help="Number of nearest neighbors to return for each image. (default=200)")
def classify(mrc_file, o, avg_nn, classification_nn, k_vdm_in, k_vdm_out):
    """ Classification-Averaging command
    """
    # TODO route optional args to the algoritm
    if not finufftpy_imported:
        logger.error("Couldn't import finufftpy! terminating.. (try to run ./install.sh)")
        return

    logger.info('class-averaging..')
    ClassAverages.run(mrc_file, o, n_nbor=classification_nn, nn_avg=avg_nn)


@simple_cli.command('compare')
@click.argument('mrcfile1', type=click.Path(exists=True))
@click.argument('mrcfile2', type=click.Path(exists=True))
@click.option('--max-error', default=None, type=float,
              help='if given, raise an error once the err is bigger than given value')
def compare_mrc_files(mrcfile1, mrcfile2, max_error):
    """ Calculate the relative error between 2 mrc stacks """
    logger.info("calculating relative err between '{}' and '{}'..".format(mrcfile1, mrcfile2))
    relative_err = cryo_compare_mrc_files(mrcfile1, mrcfile2,
                                          verbose=AspireConfig.verbosity, max_err=max_error)
    logger.info("relative err: {}".format(relative_err))


@simple_cli.command('phaseflip')
@click.argument('mrc_file', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(exists=False), default='phaseflipped.mrc',
              help="output file name (default 'phaseflipped.mrc')")
def phaseflip_mrc(mrc_file, output):
    """ Apply global phase-flip to an MRC file """
    logger.info("calculating global phaseflip..")
    global_phase_flip_mrc_file(mrc_file, output)


@simple_cli.command('crop')
@click.argument('mrc_file', type=click.Path(exists=True))
@click.argument('size', type=int)
@click.option('--fill-value', type=float, default=CropStackConfig.fill_value)
@click.option('-o', '--output', type=click.Path(exists=False), default='cropped.mrc',
              help="output file name (default 'cropped.mrc')")
def phaseflip_mrc(mrc_file, size, output, fill_value):
    """ Crop projections in stack to squares of 'size x size' px.
        Then save the cropped stack into a new MRC file.
        In case size is bigger than original stack, padding will apply.
        When padding, `--fill-value=VAL` will be used for the padded values. """
    logger.info("cropping stack {} to squre of size {}..".format(mrc_file, size))
    crop_mrc_file(mrc_file, size, output_mrc_file=output, fill_value=fill_value)


@simple_cli.command('downsample')
@click.argument('mrc_file', type=click.Path(exists=True))
@click.argument('side', type=int)
@click.option('--mask', default=None)
@click.option('-o', '--output', type=click.Path(exists=False), default='downsampled.mrc',
              help="output file name (default 'downsampled.mrc')")
def downsample_mrc(mrc_file, side, output, mask):
    """ Use Fourier methods to change the sample interval and/or aspect ratio
        of any dimensions of the input projections-stack to the output of SIZE x SIZE.
        If the optional mask argument is given, this is used as the
        zero-centered Fourier mask for the re-sampling. The size of mask should
        be the same as the output image size.
    """
    logger.info(f"downsampling stack {mrc_file} to size {side}x{side} px..")
    downsample_mrc_file(mrc_file, side, output_mrc_file=output, mask=mask)


simple_cli.add_command(classify)
simple_cli.add_command(compare_mrc_files)
simple_cli.add_command(phaseflip_mrc)
simple_cli.add_command(downsample_mrc)


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


@piped_cli.command("phaseflip")
@pass_obj
def phaseflip_stack(ctx_obj):
    """ Apply global phase-flip to an MRC stack """
    logger.debug("calculating global phaseflip..")
    ctx_obj.stack = cryo_global_phase_flip_mrc_stack(ctx_obj.stack)


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


piped_cli.add_command(phaseflip_stack)
piped_cli.add_command(chained_save_stack)


if __name__ == "__main__":
    simple_cli()
    # piped_cli
