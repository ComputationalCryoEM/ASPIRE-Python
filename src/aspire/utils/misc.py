"""
Miscellaneous Utilities that have no better place (yet).
"""
import hashlib
import logging
import os.path
import subprocess
from itertools import chain, combinations

import numpy as np

logger = logging.getLogger(__name__)


def abs2(x):
    """
    Compute complex modulus squared.
    """

    return x.real ** 2 + x.imag ** 2


def ensure(cond, error_message=None):
    """
    assert statements in Python are sometimes optimized away by the compiler, and are for internal testing purposes.
    For user-facing assertions, we use this simple wrapper to ensure conditions are met at relevant parts of the code.

    :param cond: Condition to be ensured
    :param error_message: An optional error message if condition is not met
    :return: If condition is met, returns nothing, otherwise raises AssertionError
    """
    if not cond:
        raise AssertionError(error_message)


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


def powerset(iterable):
    """
    Generate all subsets of an iterable. Example:

    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

    :return: Generator covering all subsets of iterable.
    """

    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def sha256sum(filename):
    """
    Return sha256 hash of filename.

    :param filename: path to file
    :return: sha256 hash as hex
    """

    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, "rb", buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])

    return h.hexdigest()


def gaussian_2d(resolution, x0=0, y0=0, sigma_x=1, sigma_y=1, peak=1, dtype=np.float64):
    """
    Returns a 2d Gaussian in a square 2d numpy array.

    Default is a centered disc of spread=peak=1.

    Note for odd resolutions center is the grid point 0,
    while for even resolutions center is first pixel (grid point 0.5).

    :param resolution: The height and width of returned array (pixels)
    :param x0: x cordinate of center (pixels)
    :param y0: y cordinate of center (pixels)
    :param sigma_x: spread in x direction
    :param sigma_y: spread in y direction
    :param peak: peak height at center
    :param dtype: dtype of returned array
    :return: Numpy array (2D)
    """

    # Construct centered mesh
    # # Lower
    Ll = -(resolution - 1) // 2 + (resolution - 1) % 2
    # # Upper
    Lu = resolution // 2 + 1
    (Y, X) = np.mgrid[Ll:Lu, Ll:Lu]

    p = (X - x0) ** 2 / (2 * sigma_x ** 2) + (Y - y0) ** 2 / (2 * sigma_y ** 2)
    return (peak * np.exp(-p)).astype(dtype, copy=False)
