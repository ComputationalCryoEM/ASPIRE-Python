#!/opt/anaconda2/bin/python

import sys
import argparse
import logging
import matplotlib.pyplot as plt
import mrcfile

verbosity_parser = argparse.ArgumentParser(add_help=False)
verbosity_parser.add_argument("--verbose", choices=["none", "info", "debug"],
                    default="off", help="Logging level")
verbosity_parser.add_argument("--logfile", help="log filename")

logger=logging.getLogger()

def viewstack(mrcname, numslices=16, startslice=1, nrows=4, ncols=4):

    logger.debug("Input parameters:stack=%s, numslices=%d, startslice=%d,"
                  "(nrows,ncols)=(%d,%d)",
                  mrcname, numslices, startslice, nrows, ncols)

    if numslices > nrows*ncols:
            raise ValueError("nrows*ncols should be at least numslices"
                             " to accomodate all images.")
    stack = mrcfile.mmap(mrcname,mode="r")
    logger.debug("Shape of loaded map: "\
          + str((stack.header.nx, stack.header.ny, stack.header.nz)))

    plt.figure(figsize=(4, 4))

    j = 0
    dx = 0.95/ncols
    dy = 0.95/nrows
    for col in range(ncols):
        for row in range(nrows):
            if j < numslices:
                x0 = col*1.0/ncols
                y0 = 1-row*1.0/nrows-dy
                # Flip y direction to fill image grid from top
                ax = plt.axes([x0, y0, dx, dy])
                plt.imshow(stack.data[j,...])
                plt.gray()
                plt.axis("off")
                ax.text(0.1, 0.9, str(j+startslice), fontsize=10,
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=ax.transAxes, color='red',
                        bbox=dict(facecolor='yellow', alpha=0.15))

                j += 1
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Display images from an MRCS stack.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Example: {} stack.mrcs".format(__file__),
        parents=[verbosity_parser])

    parser.add_argument("stack", help="Filename of MRCS stack")
    parser.add_argument("--numslices", type=int,
                        help="Number of slices to show",
                        default=16)
    parser.add_argument("--startslice", type=int,
                        help="Index of first image to show",
                        default=1)
    parser.add_argument("--nrows", type=int,
                        help="Number of rows in the dispay grid",
                        default=4)
    parser.add_argument("--ncols", type=int,
                        help="Number of columns in the display grid",
                        default=4)

    args = parser.parse_args()

    FORMAT = "[%(asctime)-15s][%(name)s:%(lineno)03d][%(levelname)-5s]%(message)s"
    DATEFMT = "%y-%m-%d %H:%M:%S"

    if args.verbose=="none":
        logger.propagte=False
    elif args.verbose=="info":
        logger.setLevel(logging.INFO)
    elif args.verbose=="debug":
        logger.setLevel(logging.DEBUG)

    if args.logfile is not None:
        lh = logging.FileHandler(args.logfile)
    else:
        lh = logging.StreamHandler(sys.stdout)

    lh.setFormatter(logging.Formatter(FORMAT,DATEFMT))
    logger.addHandler(lh)

    viewstack(args.stack, args.numslices, args.startslice,
              args.nrows, args.ncols)
