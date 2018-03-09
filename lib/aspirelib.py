import sys
import argparse
import logging

# Define default verbose arguments.
verbosity_parser = argparse.ArgumentParser(add_help=False)
verbosity_parser.add_argument("--verbose", choices=["none", "info", "debug"],
                              default="off", help="Logging level")
verbosity_parser.add_argument("--logfile", help="log filename")

# Define default logger.
default_logger = logging.getLogger()


def configure_logger(logger, args):
    # Set verbosity and output stream for the logger.
    # args are command line arguments that contain the verbosity level and the output (stdout/file) of the logger.
    # Configure the logger according to these parameters.

    msg_format = "[%(asctime)-15s][%(name)s:%(lineno)03d][%(levelname)-5s]%(message)s"
    date_format = "%y-%m-%d %H:%M:%S"

    if args.verbose == "none":
        logger.propagte = False
    elif args.verbose == "info":
        logger.setLevel(logging.INFO)
    elif args.verbose == "debug":
        logger.setLevel(logging.DEBUG)

    if args.logfile is not None:
        lh = logging.FileHandler(args.logfile)
    else:
        lh = logging.StreamHandler(sys.stdout)

    lh.setFormatter(logging.Formatter(msg_format, date_format))
    logger.addHandler(lh)
