"""
Miscellaneous Utilities that relate to logging.
"""

import logging
import os.path
import subprocess
from collections import defaultdict

import tqdm as _tqdm

from aspire import config

logger = logging.getLogger(__name__)

LOGGING_LEVEL_NAMES = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def get_full_version():
    """
    Get as much version information as we can, including git info (if applicable)
    This method should never raise exceptions!

    :return: A version number in the form:
        <maj>.<min>.<bld>
            If we're running as a package distributed through setuptools
        <maj>.<min>.<bld>.<rev>
            If we're running as a 'regular' python source folder, possibly locally modified

            <rev> is one of:
                'src': The package is running as a source folder
                <git_tag> or <git_rev> or <git_rev>-dirty: A git tag or commit revision, possibly followed by a suffix
                    '-dirty' if source is modified locally
                'x':   The revision cannot be determined

    """
    import aspire

    full_version = aspire.__version__
    rev = None
    try:
        path = aspire.__path__[0]
        if os.path.isdir(path):
            # We have a package folder where we can get git information
            try:
                rev = (
                    subprocess.check_output(
                        ["git", "describe", "--tags", "--always", "--dirty"],
                        stderr=subprocess.STDOUT,
                        cwd=path,
                    )
                    .decode("utf-8")
                    .strip()
                )
            except (FileNotFoundError, subprocess.CalledProcessError):
                # no git or not a git repo? assume 'src'
                rev = "src"
    except Exception:  # nopep8  # noqa: E722
        # Something unexpected happened - rev number defaults to 'x'
        rev = "x"

    if rev is not None:
        full_version += f".{rev}"

    return full_version


def tqdm(*args, **kwargs):
    """
    Wraps `tqdm.tqdm`, applying ASPIRE configuration.

    Currently setting `aspire.config['logging']['tqdm_disable']`
    true/false will disable/enable tqdm progress bars.
    """

    disable = config["logging"]["tqdm_disable"] or (
        getConsoleLoggingLevel() not in ["DEBUG", "INFO"]
    )
    return _tqdm.tqdm(*args, **kwargs, disable=disable)


def trange(*args, **kwargs):
    """
    Wraps `tqdm.trange`, applying ASPIRE configuration.

    Currently setting `aspire.config['logging']['tqdm_disable']`
    true/false will disable/enable tqdm progress bars.
    """

    disable = config["logging"]["tqdm_disable"] or (
        getConsoleLoggingLevel() not in ["DEBUG", "INFO"]
    )
    return _tqdm.trange(*args, **kwargs, disable=disable)


def setConsoleLoggingLevel(level_name):
    """
    Dynamically sets the console logging level by setting the level of the root logger's StreamHandler to `level_name`.
    Note this will supersede the `logging.console_level` option stored in ASPIRE's configuration file.

    :param level_name: One of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
    """
    if level_name not in LOGGING_LEVEL_NAMES:
        raise ValueError(
            f"{level_name} not a recognized logging level. Must be one of {LOGGING_LEVEL_NAMES}"
        )
    # handler list is ordered according to logging.conf
    stream_handler = logging.getLogger().handlers[0]
    stream_handler.setLevel(getattr(logging, level_name))


def getConsoleLoggingLevel():
    """
    Returns the Python logging level of the root logger's StreamHandler, i.e. console output.
    This is the same as the `logging.console_level` option in ASPIRE's configuration file unless
    the console logging level has been changed dynamically during a session. (e.g. by a CLI option)

    :return: The current console logging level name as a string. One of "DEBUG". "INFO", "WARNING",
    "ERROR", "CRITICAL".
    """
    # handler list is ordered according to logging.conf
    stream_handler = logging.getLogger().handlers[0]
    return logging.getLevelName(stream_handler.level)


def setFileLoggingLevel(level_name):
    """
    Dynamically sets the log file logging level by setting the level of the root logger's FileHandler to `level_name`.
    Note this will supersede the `logging.log_file_level` option stored in ASPIRE's configuration file.

    :param level_name: One of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
    """
    if level_name not in LOGGING_LEVEL_NAMES:
        raise ValueError(
            f"{level_name} not a recognized logging level. Must be one of {LOGGING_LEVEL_NAMES}"
        )
    # handler list is ordered according to logging.conf
    file_handler = logging.getLogger().handlers[1]
    file_handler.setLevel(getattr(logging, level_name))


def getFileLoggingLevel():
    """
    Returns the Python logging level of the root logger's FileHandler, i.e. log file output.
    This is the same as the `logging.log_file_level` option in ASPIRE's configuration file
    unless the file logging level has been changed dynamically during a session. (e.g. by a CLI option)

    :return: The current file logging level name as a string. One of "DEBUG", "INFO", "WARNING",
    "ERROR", "CRITICAL".
    """
    # handler list is ordered according to logging.conf
    file_handler = logging.getLogger().handlers[1]
    return logging.getLevelName(file_handler.level)


class LogFilterByCount:
    """
    Provide a context manager for filtering repetitive log messages.
    """

    # msg_cache is intentionally shared by all instances of class.
    # msg_cache is map hash(str(msg)) ~~> count.
    msg_cache = defaultdict(int)

    def __init__(self, logger: logging.Logger, max_count: int):
        """
        Initialize context manager based on `logger` and `max_count`.

        :param logger: `Logger` instance.
        :param max_count: Global limit for count of each message
            encountered inside context.
        """

        self._logger = logger
        self._max_count = max_count

    def filter(self, record):
        """
        Increment msg_cache for `record`.

        `filter` returns True when `seen` <= `count` for this context.
        True implies the logger will pass the message.

        :param record: Log record.  Will be reduced by hash(str()),
        :return: Boolean
        """

        # Log messages can be arbitrarily long,
        #   convert to a hash(str())
        msg = hash(str(record.msg))

        # Increment seen count in cache.
        self.msg_cache[msg] += 1
        seen = self.msg_cache[msg]

        return seen <= self._max_count

    def __enter__(self):
        self._logger.addFilter(self)

    def __exit__(self, *args):
        self._logger.removeFilter(self)
