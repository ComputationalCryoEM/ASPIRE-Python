"""
Miscellaneous Utilities that relate to logging.
"""
import logging
import os.path
import subprocess

import tqdm as _tqdm

from aspire import config

logger = logging.getLogger(__name__)


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

    disable = config["logging"]["tqdm_disable"]
    return _tqdm.tqdm(*args, **kwargs, disable=disable)


def trange(*args, **kwargs):
    """
    Wraps `tqdm.trange`, applying ASPIRE configuration.

    Currently setting `aspire.config['logging']['tqdm_disable']`
    true/false will disable/enable tqdm progress bars.
    """

    disable = config["logging"]["tqdm_disable"]
    return _tqdm.trange(*args, **kwargs, disable=disable)
