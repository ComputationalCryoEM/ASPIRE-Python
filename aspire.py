#!/usr/bin/env python3

import argparse

from aspire.class_avrages.class_averages import ClassAverages


class AspireCommandParser(argparse.ArgumentParser):
    """ This class routes the aspire subcommands to their appropriate application. """

    def route_subcommand(self):
        args = self.parse_args()

        if not args.subparser_name:
            parser.print_help()
            return

        # route input args to subcommand
        args.func(args)

    @staticmethod
    def classify(subcommand_args):
        ClassAverages.run(subcommand_args.instack, subcommand_args.outstack)

    @staticmethod
    def preprocess(subcommand_args):
        raise NotImplementedError("preprocessor isn't support yet. Stay tuned!")


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

    parser.route_subcommand()
