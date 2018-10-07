#!/usr/bin/env python3

import click
import mrcfile

from functools import update_wrapper


from aspire.class_averaging.averaging import ClassAverages
from aspire.common.logger import logger
from aspire.preprocessor import cryo_global_phase_flip_mrc_stack
from aspire.utils.compare_stacks import cryo_compare_mrc_files
from aspire.utils.mrc_utils import cryo_global_phase_flip_mrc_file


@click.group(chain=True)
def cli():
    pass


@cli.resultcallback()
def process_commands(processors):
    """ This result callback is invoked with an iterable of all the chained
        subcommands.  As in this example each subcommand returns a function
        we can chain them together to feed one into the other, similar to how
        a pipe on unix works.
    """
    # Start with an empty iterable.
    stream = ()

    # Pipe it through all stream processors.
    for click_processor in processors:
        stream = click_processor(stream)

    # Evaluate the stream and throw away the items.
    for _ in stream:
        pass


def processor(f):
    """Helper decorator to rewrite a function so that it returns another
    function from it.
    """
    def new_func(*args, **kwargs):
        def wrapped_processor(stream):
            return f(stream, *args, **kwargs)
        return wrapped_processor
    return update_wrapper(new_func, f)


def generator(f):
    """Similar to the :func:`processor` but passes through old values
    unchanged and does not pass through the values as parameter.
    """
    @processor
    def new_func(stream, *args, **kwargs):
        for item in stream:
            yield item
        for item in f(*args, **kwargs):
            yield item
    return update_wrapper(new_func, f)


@click.command()
@click.argument('mrcfile')
def preprocess(mrcfile):
    # TODO add preprocessor flow
    logger.info('preprocessing {}..'.format(mrcfile))
    raise NotImplementedError("Preprocessor isn't support yet. Stay tuned!")


@click.command()
@click.argument('filename', type=click.Path(exists=True))
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
def classify(filename, o, avg_nn, classification_nn, k_vdm_in, k_vdm_out):
    # TODO route optional args to the algoritm
    logger.info('classifying..')
    ClassAverages.run(filename, o, n_nbor=classification_nn, nn_avg=avg_nn)


@click.command()
@click.argument('mrcfile1', type=click.Path(exists=True))
@click.argument('mrcfile2', type=click.Path(exists=True))
@click.option('-v', default=0, help='verbosity (0-None 1-progress bar 2-each 100 3-each projection')
@click.option('--max-error', default=None,
              help='if given, raise an error once the err is bigger than given value')
def compare_stacks(mrcfile1, mrcfile2, v, max_error):
    logger.info("calculating relative err..")
    relative_err = cryo_compare_mrc_files(mrcfile1, mrcfile2, verbose=v, max_err=max_error)
    logger.info("relative err: {}".format(relative_err))


@cli.command()
@click.argument('mrcfile', type=click.Path(exists=True))
@click.option('-o', type=click.Path(exists=False), help='output file name')
def phaseflip_mrc(filename, o=None):
    """ Apply global phase-flip to an MRC file """
    logger.info("calculating global phaseflip..")
    cryo_global_phase_flip_mrc_file(filename, o)


@cli.command('mrcopen')
@click.argument('filename', type=click.Path(exists=True))
@generator
def mrc_open_cmd(filename):
    """ Loads one or multiple images for processing.  The input parameter
        can be specified multiple times to load more than one image.
    """
    logger.info("opening {}".format(filename))
    yield mrcfile.open(filename).data


@cli.command("phaseflip")
@processor
def chained_phaseflip(stacks):
    """ Apply global phase-flip to an MRC stack """
    logger.info("calculating global phaseflip..")
    for stack in stacks:
        yield cryo_global_phase_flip_mrc_stack(stack)


@cli.command("save")
@click.option('-o', type=click.Path(exists=False), default='output.mrc',
              help='output file name')
@processor
def chained_phaseflip(stacks, o=None):
    """ Apply global phase-flip to an MRC stack

        TODO support multiple stacks
    """
    logger.info("saving stack {}..".format(o))
    yield mrcfile.new(o, stacks.__next__())

    try:
        stacks.__next__()
        raise NotImplementedError('aspire currently supports saving only 1 stack at a time!')
    except StopIteration:
        pass


cli.add_command(classify)
cli.add_command(preprocess)
cli.add_command(compare_stacks)
cli.add_command(phaseflip_mrc)
cli.add_command(chained_phaseflip)
cli.add_command(mrc_open_cmd)


if __name__ == "__main__":
    cli()
