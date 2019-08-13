import sys
import logging
import platform
import struct
import subprocess
import traceback


def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Handle any top-level unhandled exception.

    :param exc_type: Exception type object
    :param exc_value: Exception value object (an instance of type exc_type)
    :param exc_traceback: The Traceback object associated with exc_value (also available as exc_value.__traceback__)
    :return: On return, useful diagnostic information has been logged, and the exception re-raised.
    """

    # Are we explicitly/interactively killing a run? Just do it.
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    from aspire.utils import get_full_version
    logger = logging.getLogger('aspire')

    # Default level at which we log messages.
    # Since we'll be logging a whole bunch of diagnostic information which we do not want to clutter stdout with,
    # this should be a 'low-enough' number so we can pass below the radar w.r.t. other registered logging handlers.
    level = 5

    logger.log(level, '-----------Uncaught exception-----------')
    logger.log(level, f'Application version: {get_full_version()}')
    logger.log(level, f'Platform: {platform.platform()}')
    logger.log(level, f'Python version: {sys.version}')
    logger.log(level, f'Python 32/64 bit: {8 * struct.calcsize("P")}')

    logger.log(level, 'conda list output:')
    try:
        out = subprocess.check_output(['conda', 'list'], stderr=subprocess.STDOUT).decode('utf8')
        for line in out.split('\n'):
            logger.log(level, line)
    except:  # nopep8
        pass

    logger.log(level, 'pip freeze output:')
    try:
        out = subprocess.check_output(['pip', 'freeze'], stderr=subprocess.STDOUT).decode('utf8')
        for line in out.split('\n'):
            logger.log(5, line)
    except:  # nopep8
        pass

    # Walk through all traceback objects (oldest call -> most recent call), capturing frame/local variable information.
    logger.log(level, 'Exception Details (most recent call last)')
    frame_generator = traceback.walk_tb(exc_traceback)
    stack_summary = traceback.StackSummary.extract(frame_generator, capture_locals=True)
    frame_strings = stack_summary.format()
    for s in frame_strings:
        for line in s.split('\n'):
            logger.log(level, line)

    try:
        # re-raise the exception we got for the caller.
        raise exc_value
    finally:
        # cleanup - see https://cosmicpercolator.com/2016/01/13/exception-leaks-in-python-2-and-3/
        del exc_value, exc_traceback


# Useful Exception classes
class AspireException(Exception):
    pass


class WrongInput(AspireException):
    pass


class DimensionsIncompatible(AspireException):
    pass


class ErrorTooBig(AspireException):
    pass


class UnknownFormat(AspireException):
    pass
