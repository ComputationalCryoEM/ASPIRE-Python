"""
Miscellaneous Utilities that have no better place (yet).
"""
import hashlib
import logging
import os.path
import subprocess
from itertools import chain, combinations

import numpy as np

from aspire.utils.coor_trans import grid_1d, grid_2d, grid_3d

logger = logging.getLogger(__name__)


def abs2(x):
    """
    Compute complex modulus squared.
    """

    return x.real**2 + x.imag**2


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


def gaussian_1d(size, mu=0, sigma=1, peak=1, dtype=np.float64):
    """
    Returns a 1d Gaussian in a 1D numpy array.

    Default is a centered disc of spread=peak=1.

    :param size: The height and width of returned array (pixels)
    :param mu: mean or center (pixels)
    :param sigma: spread
    :param peak: peak height at center
    :param dtype: dtype of returned array
    :return: Numpy array (1D)
    """

    # Construct centered mesh
    g = grid_1d(size, normalized=False, dtype=dtype)

    p = (g["x"] - mu) ** 2 / (2 * sigma**2)

    return (peak * np.exp(-p)).astype(dtype, copy=False)


def gaussian_2d(size, x0=0, y0=0, sigma_x=1, sigma_y=1, peak=1, dtype=np.float64):
    """
    Returns a 2d Gaussian in a square 2d numpy array.

    Default is a centered disc of spread=peak=1.

    :param size: The height and width of returned array (pixels)
    :param x0: x coordinate of center (pixels)
    :param y0: y coordinate of center (pixels)
    :param sigma_x: spread in x direction
    :param sigma_y: spread in y direction
    :param peak: peak height at center
    :param dtype: dtype of returned array
    :return: Numpy array (2D)
    """

    # Construct centered mesh
    g = grid_2d(size, shifted=False, normalized=False, indexing="xy", dtype=dtype)

    p = (g["x"] - x0) ** 2 / (2 * sigma_x**2) + (g["y"] - y0) ** 2 / (
        2 * sigma_y**2
    )
    return (peak * np.exp(-p)).astype(dtype, copy=False)


def gaussian_3d(size, mu=(0, 0, 0), sigma=(1, 1, 1), peak=1, dtype=np.float64):
    """
    Returns a 3d Gaussian in a size-by-size-by-size 3d numpy array.

    Default is a centered volume of spread=peak=1.

    :param size: The height and width of returned array (pixels)
    :param mu: A 3-tuple indicating the center of the Gaussian
    :param sigma: A 3-tuple of spreads corresponding to mu
    :param peak: peak height at center
    :param dtype: dtype of returned array
    :return: Numpy array (3D)
    """

    # Construct centered mesh
    g = grid_3d(size, shifted=False, normalized=False, indexing="xyz", dtype=dtype)

    p = (
        (g["x"] - mu[0]) ** 2 / (2 * sigma[0] ** 2)
        + (g["y"] - mu[1]) ** 2 / (2 * sigma[1] ** 2)
        + (g["z"] - mu[2]) ** 2 / (2 * sigma[2] ** 2)
    )
    return (peak * np.exp(-p)).astype(dtype, copy=False)


def circ(size, x0=0, y0=0, radius=1, peak=1, dtype=np.float64):
    """
    Returns a 2d `circ` function in a square 2d numpy array.

    where for r = sqrt(x**2 + y**2)

    circ(x,y) = peak : 0 <= r <= radius
                0 : otherwise

    Default is a centered circle of spread=peak=1.

    :param size: The height and width of returned array (pixels)
    :param x0: x cordinate of center (pixels)
    :param y0: y cordinate of center (pixels)
    :param radius: radius of circle
    :param peak: peak height at center
    :param dtype: dtype of returned array
    :return: Numpy array (2D)
    """

    # Construct centered mesh
    g = grid_2d(size, shifted=True, normalized=False, indexing="yx", dtype=dtype)

    vals = ((g["x"] - x0) ** 2 + (g["y"] - y0) ** 2) < radius * radius
    return (peak * vals).astype(dtype)


def inverse_r(size, x0=0, y0=0, peak=1, dtype=np.float64):
    """
    Returns a 2d inverse radius function in a square 2d numpy array.

    Where inverse_r(x,y): 1/sqrt(1 + x**2 + y**2)

    Default is a centered circle of peak=1.

    :param size: The height and width of returned array (pixels)
    :param x0: x cordinate of center (pixels)
    :param y0: y cordinate of center (pixels)
    :param peak: peak height at center
    :param dtype: dtype of returned array
    :return: Numpy array (2D)
    """

    # Construct centered mesh
    g = grid_2d(size, shifted=True, normalized=False, indexing="yx", dtype=dtype)

    # Compute the denominator
    vals = np.sqrt(1 + (g["x"] - x0) ** 2 + (g["y"] - y0) ** 2)

    return (peak / vals).astype(dtype)
