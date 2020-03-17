#!/usr/bin/env python3
import os
import click
import numpy as np
from aspire.preprocess.preprocessor import preprocess
# from aspire.class_averaging.class_averaging import class_averaging
# from aspire.abinitio.cryo_abinitio_c1_worker import cryo_abinitio_c1_worker
# from aspire.utils.read_write import write_mrc, read_mrc

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
    return


@simple_cli.command('abinitio', short_help='Abinitio algorithm')
@click.argument('stack_file', type=click.Path(exists=True))
@click.option('-o', '--output', default=None, type=click.Path(exists=False))
@click.option('--num_images', type=int, default=None)
@click.option('--max_shift', type=float, default=0.15)
def abinitio_cmd(stack_file, output, num_images, max_shift):
    """\b
        ############################
                 Abinitio
        ############################

        Abinitio accepts a stack file, calculates Abinitio algorithm on it and saves
        the results into output file (default adds '_abinitio' to stack name)
    """

    output = 'volume.mrc' if output is None else output

    print('Loading images from {}'.format(stack_file))
    stack = np.ascontiguousarray(read_mrc(stack_file))
    num_images = min(num_images, stack.shape[2])
    print('There are {} images in the file, selecting the top {}'.format(stack.shape[2], num_images))
    stack = stack[:, :, :num_images]

    volume = cryo_abinitio_c1_worker(stack, 2, max_shift=max_shift)
    write_mrc(output, volume)


@simple_cli.command('classify', short_help='Classification-Averaging algorithm')
@click.argument('stack_file', type=click.Path(exists=True))
@click.option('-o', '--output', default=None,
              type=click.Path(exists=False), help='output file name. (default averages.mrcs)')
@click.option("--num_nbor", default=100,
              help=("Number of nearest neighbors to find for each "
                    "image during initial classification. (default=100)"))
@click.option("--nn_avg", default=50, help="Number of images to average into each class. (default=50)")
@click.option("--max_shift", default=15, help="Max shift of projections from the center. (default=15)")
@click.option("--size_subset", default=5000, help="Number of images to pick for abinitio. (default=5000)")
@click.option("--ordered_output", default=None, help="Name of file for top images. (default averages_subset.mrcs)")
def classify_cmd(stack_file, output, num_nbor, nn_avg, max_shift, size_subset, ordered_output):
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
    output = 'averages.mrcs' if output is None else output
    ordered_output = 'ordered_averages.mrcs' if ordered_output is None else ordered_output

    stack = read_mrc(stack_file)
    averages, ordered_averages = class_averaging(stack, num_nbor, nn_avg, max_shift, size_subset)
    write_mrc(output, averages)
    write_mrc(ordered_output, ordered_averages)


@simple_cli.command('preprocess', short_help='Gather projections from star file, downsample normalize and whiten them')
@click.argument('star_file', type=click.Path(exists=True))
@click.option('-ps', '--pixel_size', default=None)
@click.option('-cs', '--crop_size', type=int, default=-1)
@click.option('-ds', '--downsample_size', type=int, default=89)
@click.option('-o', '--output', type=str, default='preprocessed_stack.mrcs')
def preprocess_cmd(star_file, pixel_size=None, crop_size=-1, downsample_size=89, output='preprocessed_stack.mrcs'):
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
    stack = preprocess(star_file, pixel_size, crop_size, downsample_size)
    write_mrc(output, stack)


if __name__ == "__main__":
    simple_cli()
