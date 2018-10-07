#!/usr/bin/env python3
import os
import sys
import click
import mrcfile


from aspire.class_averaging.averaging import ClassAverages
from aspire.common.logger import logger
from aspire.preprocessor import cryo_global_phase_flip_mrc_stack
from aspire.utils.compare_stacks import cryo_compare_mrc_files
from aspire.utils.mrc_utils import cryo_global_phase_flip_mrc_file


class PipedObj:
    def __init__(self, mrc_file, debug, verbose):
        self.stack = mrcfile.open(mrc_file).data
        self.debug = debug
        self.verbose = verbose


pass_obj = click.make_pass_decorator(PipedObj, ensure=True)


@click.group(chain=False)
def cli1():
    pass


@click.group(chain=True)
@click.option('--debug/--no-debug', default=False)
@click.option('-v', default=0)
@click.argument('input_mrc')
@click.pass_context
def cli2(ctx, input_mrc, debug, v):
    logger.setLevel(debug)
    ctx.obj = PipedObj(input_mrc, debug, v)


@cli1.command('preprocess')
@click.argument('mrcfile')
def preprocess_mrc(mrcfile):
    # TODO add preprocessor flow
    logger.info('preprocessing {}..'.format(mrcfile))
    raise NotImplementedError("Preprocessor isn't support yet. Stay tuned!")


@cli1.command()
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


@cli1.command()
@click.argument('mrcfile1', type=click.Path(exists=True))
@click.argument('mrcfile2', type=click.Path(exists=True))
@click.option('-v', default=0, help='verbosity (0-None 1-progress bar 2-each 100 3-each projection')
@click.option('--max-error', default=None,
              help='if given, raise an error once the err is bigger than given value')
def compare_stacks(mrcfile1, mrcfile2, v, max_error):
    logger.info("calculating relative err..")
    relative_err = cryo_compare_mrc_files(mrcfile1, mrcfile2, verbose=v, max_err=max_error)
    logger.info("relative err: {}".format(relative_err))


@cli1.command()
@click.argument('mrcfile', type=click.Path(exists=True))
@click.option('-o', type=click.Path(exists=False), help='output file name')
def phaseflip_mrc(filename, o=None):
    """ Apply global phase-flip to an MRC file """
    logger.info("calculating global phaseflip..")
    cryo_global_phase_flip_mrc_file(filename, o)


@cli2.command("phaseflip")
@pass_obj
def phaseflip_stack(ctx_obj):
    """ Apply global phase-flip to an MRC stack """
    logger.debug("calculating global phaseflip..")
    ctx_obj.stack = cryo_global_phase_flip_mrc_stack(ctx_obj.stack)


@cli2.command("save")
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


cli1.add_command(classify)
cli1.add_command(preprocess_mrc)
cli1.add_command(compare_stacks)
cli1.add_command(phaseflip_mrc)

cli2.add_command(phaseflip_stack)
cli2.add_command(chained_save_stack)


# cli = click.CommandCollection(sources=[cli1, cli2])

if __name__ == "__main__":
    cli2()
