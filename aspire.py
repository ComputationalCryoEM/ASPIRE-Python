#!/usr/bin/env python3

import argparse
import time

from aspire.class_averaging.averaging import ClassAverages
from aspire.logger import logger
from aspire.preprocessor.cryo_compare_stacks import cryo_compare_mrc_files


class AspireCommandParser(argparse.ArgumentParser):
    """ This class routes the aspire subcommands to their appropriate application. """

    def route_subcommand(self):
        args = self.parse_args()

        if not args.subparser_name:
            parser.print_help()
            return

        # route input args to subcommand
        t0_process = time.process_time()
        t0_wall = time.time()
        args.func(args)
        logger.debug("Finished in process time: {} sec".format(time.process_time() - t0_process))
        logger.debug("Finished in wall time: {} sec".format(time.time() - t0_wall))

    @staticmethod
    def classify(subcommand_args):
        ClassAverages.run(subcommand_args.instack, subcommand_args.outstack)

    @staticmethod
    def preprocess(subcommand_args):
        raise NotImplementedError("preprocessor isn't support yet. Stay tuned!")

    @staticmethod
    def compare_stacks(subcommand_args):
        logger.info("calculating relative err..")
        relative_err = cryo_compare_mrc_files(subcommand_args.mrcfile1,
                                              subcommand_args.mrcfile2,
                                              verbose=subcommand_args.verbose,
                                              max_err=subcommand_args.max_err)

        logger.info("relative err: {}".format(relative_err))


if __name__ == "__main__":
    # create the top-level parser
    parser = AspireCommandParser(prog='aspire')

    # add parsers for aspire sub commands
    subparsers = parser.add_subparsers(title="subcommands", dest="subparser_name")

    # configure parser for preprocessor
    preprocessor_parser = subparsers.add_parser('preprocess',
                                                help='preprocess stack before classifying')
    preprocessor_parser.set_defaults(func=parser.preprocess)

    # configure parser for classifier
    classifier_parser = subparsers.add_parser('classify', help='average classifier')
    classifier_parser.set_defaults(func=parser.classify)

    required_args = classifier_parser.add_argument_group('required arguments')
    required_args.add_argument("-i", "--instack", required=True,
                               help="Filename of MRCS stack. Images should be prewhitened.")

    classifier_parser.add_argument("-o", "--outstack", required=False, default='classified.mrcs',
                                   help=("Output stack filename of MRCS stack. "
                                         "default is 'classified.mrcs'"))

    classifier_parser.add_argument("--avg_nn", type=int,
                                   help="Number of images to average into each class. (default=50)",
                                   default=50)

    classifier_parser.add_argument("--classification_nn", type=int,
                                   help=("Number of nearest neighbors to find for each "
                                         "image during initial classification. (default=100)"),
                                   default=100)

    classifier_parser.add_argument("--K_VDM_in", type=int,
                                   help=("Number of nearest neighbors for building VDM graph."
                                         "(default=20"),
                                   default=20)

    classifier_parser.add_argument("--K_VDM_out", type=int,
                                   help=("Number of nearest neighbors to return for each image."
                                         "(default=200)"),
                                   default=200)

    # configure parser for compare-stacks
    compare_stacks_parser = subparsers.add_parser('compare-stacks',
                                                  help='Compare relative error between 2 stacks')
    compare_stacks_parser.set_defaults(func=parser.compare_stacks)

    compare_stacks_parser.add_argument("mrcfile1", help="first mrc file to compare")
    compare_stacks_parser.add_argument("mrcfile2", help="second mrc file to compare")
    compare_stacks_parser.add_argument("-v", "--verbose", type=int,
                                       help="increase output verbosity.\n"
                                            "0: silent\n"
                                            "1: show progress-bar\n"
                                            "2: print relative err every 100 images\n"
                                            "3: print relative err for each image")

    compare_stacks_parser.add_argument("--max-err", type=float,
                                       help="raise an error if relative error is "
                                            "bigger than max-err")

    # parse input args and route them to the appropriate commands
    parser.route_subcommand()
