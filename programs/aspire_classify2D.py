#!/opt/anaconda2/bin/python

import argparse
import aspirelib

logger = aspirelib.default_logger



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate 2D class averages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Example: {} --instack stack.mrcs".format(__file__),
        parents=[aspirelib.verbosity_parser])

    parser.add_argument("instack", help="Filename of MRCS stack. Images should be prewhitened.")
    parser.add_argument("--avg_nn", type=int,
                        help="Number of images to average into each class.",
                        default=50)
    parser.add_argument("--classification_nn", type=int,
                        help="Number of nearest neighbors to find for each image during initial classification.",
                        default=100)
    parser.add_argument("--K_VDM_in", type=int,
                        help="Number of nearest neighbors for building VDM graph.",
                        default=20)
    parser.add_argument("--K_VDM_out", type=int,
                        help="Number of nearest neighbors to return for each image.",
                        default=200)

    args = parser.parse_args()
    aspirelib.configure_logger(logger,args)
