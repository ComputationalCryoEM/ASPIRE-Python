import logging
import platform
import struct
import subprocess
import sys
import traceback


def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Handle any top-level unhandled exception.
    Tries to gather and log additional context information,
    then re-raises.

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

    lines = list()

    lines.append(f"Application version: {get_full_version()}")
    lines.append(f"Platform: {platform.platform()}")
    lines.append(f"Python version: {sys.version}")
    lines.append(f'Python 32/64 bit: {8 * struct.calcsize("P")}')

    lines.append("conda list output:")
    try:
        lines.extend(
            subprocess.check_output(["conda", "list"], stderr=subprocess.STDOUT)
            .decode("utf8")
            .split("\n")
        )
    except Exception:  # nopep8  # noqa: E722
        pass

    lines.append("pip freeze output:")
    try:
        lines.extend(
            subprocess.check_output(["pip", "freeze"], stderr=subprocess.STDOUT)
            .decode("utf8")
            .split("\n")
        )
    except Exception:  # nopep8  # noqa: E722
        pass

    # Walk through all traceback objects (oldest call -> most recent call), capturing frame/local variable information.
    lines.append("Exception Details (most recent call last)")
    frame_generator = traceback.walk_tb(exc_traceback)

    try:
        stack_summary = traceback.StackSummary.extract(
            frame_generator, capture_locals=True
        )
    except Exception:  # nopep8  # noqa: E722
        # The above code, while more informative, doesn't always work.
        # When it doesn't try something simpler.
        stack_summary = traceback.StackSummary.extract(
            frame_generator, capture_locals=False
        )

    frame_strings = stack_summary.format()
    for s in frame_strings:
        lines.extend(s.split("\n"))

    try:
        with open("aspire.err.log", "w") as f:
            f.write("\n".join(lines) + "\n")
    except Exception:  # nopep8  # noqa: E722
        pass

    try:
        # send to logger
        logging.critical(
            f"{exc_value}\nTraceback:\n"
            f'{"".join(traceback.format_tb(exc_traceback))}'
        )
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
