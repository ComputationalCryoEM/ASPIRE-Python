#!/usr/bin/env python3

import functools
import time
import click

from aspire.class_averaging.averaging import ClassAverages
from aspire.common.logger import logger
from aspire.utils.compare_stacks import cryo_compare_mrc_files
from aspire.utils.mrc_utils import cryo_global_phase_flip_mrc_file


def timer(func):
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        t0_process = time.process_time()
        t0_wall = time.time()
        func(*args, **kwargs)
        logger.info("Finished in process time: {} sec".format(time.process_time() - t0_process))
        logger.info("Finished in wall time: {} sec".format(time.time() - t0_wall))

    return decorator


@click.group(chain=True, invoke_without_command=True)
def cli():
    pass


@cli.resultcallback()
def process_pipeline(processors, input):
    iterator = (x.rstrip('\r\n') for x in input)
    for processor in processors:
        iterator = processor(iterator)
    for item in iterator:
        click.echo(item)


@click.command()
@click.argument('mrcfile')
def preprocess(mrcfile):
    # TODO add preprocessor flow
    logger.info('preprocessing {}..'.format(mrcfile))
    raise NotImplementedError("Preprocessor isn't support yet. Stay tuned!")


@click.command()
@click.argument('mrcfile', type=click.Path(exists=True))
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
def classify(mrcfile, o, avg_nn, classification_nn, k_vdm_in, k_vdm_out):
    # TODO route optional args to the algoritm
    def processor(iterator):
        for line in iterator:
            yield line.upper()
        logger.info('classifying..')
        ClassAverages.run(mrcfile, o, n_nbor=classification_nn, nn_avg=avg_nn)


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
def phaseflip(subcommand_args):
    """ Apply global phase-flip to an MRC file """
    logger.info("calculating global phaseflip..")
    cryo_global_phase_flip_mrc_file(subcommand_args.mrcfile, subcommand_args.o)


cli.add_command(classify)
cli.add_command(preprocess)
cli.add_command(compare_stacks)
cli.add_command(phaseflip)


if __name__ == "__main__":
    cli()
