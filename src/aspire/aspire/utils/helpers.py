import functools
import os
import sys
from logging import warning

from aspire.aspire.common.exceptions import DimensionsIncompatible
from aspire.aspire.common.logger import logger


def get_file_type(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    return os.path.splitext(file_path)[1]


class TupleCompare(object):
    """ Helper class to compare between all members of tuples. """

    @classmethod
    def validate_same_length(cls, a, b):
        if len(a) != len(b):
            raise DimensionsIncompatible("Can't compare tuples of different length")

    @classmethod
    def gt(cls, a, b, eq=False):
        cls.validate_same_length(a, b)
        if eq is True:
            return all([i >= j for i, j in zip(a, b)])
        else:
            return all([i > j for i, j in zip(a, b)])

    @classmethod
    def lt(cls, a, b, eq=False):
        cls.validate_same_length(a, b)
        if eq is True:
            return all([i <= j for i, j in zip(a, b)])
        else:
            return all([i < j for i, j in zip(a, b)])

    @classmethod
    def eq(cls, a, b):
        cls.validate_same_length(a, b)
        return all([i == j for i, j in zip(a, b)])


def colorize(s, color):
    # avoid colorizing on Win/iOS
    if not sys.platform.startswith('linux'):
        return s

    class Color:
        YELLOW = '\033[1;33m'
        RED = '\033[91m'
        NO_COLOR = '\033[0m'

    color = color.upper()
    if not hasattr(Color, color):
        warning(f"Unknown color! ({color})")
        return s

    return f"{getattr(Color, color)}{s}{Color.NO_COLOR}"


def yellow(s):
    return colorize(s, 'YELLOW')


def red(s):
    return colorize(s, 'RED')


def requires_binaries(*filenames):
    """ Decorator checking existing of binary files in directory 'binaries'.
        It prints an error and exits in case they don't.

        E.g.
        requires_binaries('mask1.npy', 'mask2.npy')
        def masking_stack_file(mrc_file):
            # do something with binary files mask1.npy and mask2.npy knowing they're there

    """
    def decorator(func):

        missing_binaries = set()
        for fn in filenames:
            if not os.path.exists(f'./binaries/{fn}'):
                missing_binaries.add(fn)

        if missing_binaries:
            for fn in missing_binaries:
                logger.error(f"Binary file {yellow(fn)} is required!")

            logger.error(f"Please run \'{yellow('make data')}\' and try again.")
            sys.exit(1)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper
    return decorator


def set_output_name(input_name, suffix):

    input_name = os.path.basename(input_name)
    split = input_name.rsplit(".", maxsplit=1)
    input_file_prefix = split[0]
    file_ending = '.' + split[1] if len(split) == 2 else ''
    return f'{input_file_prefix}_{suffix}{file_ending}'


def accepts(*types):
    """ Decorator matching func args and their respective
        positional args in the decorated function.

        # Example:
        @accepts(int, str)
        def test_int_func(x, y):
            return x == int(y)

    """
    def check_accepts(f):

        assert len(types) == f.__code__.co_argcount

        def new_f(*args, **kwds):
            for (a, t) in zip(args, types):
                assert isinstance(a, t), f"arg {a} does not match type {t}"

            return f(*args, **kwds)

        new_f.func_name = f.__name__

        return new_f
    return check_accepts
